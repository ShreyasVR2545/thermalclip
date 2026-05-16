"""
ThermalCLIP Evaluation Suite
==============================
Four evaluation metrics that together demonstrate cross-spectral alignment quality:

1. Cross-modal retrieval Precision@K (both directions)
2. Linear probe classification accuracy
3. t-SNE embedding space visualisation
4. Temperature prediction MAE (physics decoder)

Each metric tells a different part of the story:
    - Retrieval P@K: "can the model find the right thermal image for an RGB query?"
    - Linear probe: "do the aligned embeddings carry semantic information?"
    - t-SNE: "visual proof that the embedding space is modality-agnostic"
    - Temperature MAE: "is the physics grounding actually working?"
"""

import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast

from config import ThermalCLIPConfig
from dataset import FLIRPairedDataset
from model import ThermalCLIP
from physics import scene_temperature


# ── 1. Cross-Modal Retrieval ─────────────────────────────────────────────

@torch.no_grad()
def extract_embeddings(model, dataloader, device, use_amp=True):
    """Extract all RGB and thermal embeddings from a dataset."""
    model.eval()
    rgb_embs, thermal_embs, pred_temps, target_temps, indices = [], [], [], [], []

    for rgb, thermal, thermal_raw, idx in dataloader:
        rgb = rgb.to(device, non_blocking=True)
        thermal = thermal.to(device, non_blocking=True)
        thermal_raw = thermal_raw.to(device, non_blocking=True)

        with autocast('cuda', enabled=use_amp):
            r_emb = model.encode_rgb(rgb)
            t_emb = model.encode_thermal(thermal)
            p_temp = model.physics_decoder(t_emb)

        t_target = scene_temperature(thermal_raw)

        rgb_embs.append(r_emb.float().cpu())
        thermal_embs.append(t_emb.float().cpu())
        pred_temps.append(p_temp.float().cpu())
        target_temps.append(t_target.float().cpu())
        indices.append(idx)

    return {
        "rgb_emb": torch.cat(rgb_embs),
        "thermal_emb": torch.cat(thermal_embs),
        "pred_temp": torch.cat(pred_temps),
        "target_temp": torch.cat(target_temps),
        "indices": torch.cat(indices),
    }


def cross_modal_retrieval(rgb_emb, thermal_emb, k=5):
    """
    Compute Precision@K for cross-modal retrieval in both directions.

    RGB → Thermal: for each RGB query, retrieve top-K thermal images by
    cosine similarity. A retrieval is correct if the thermal image index
    matches the RGB query index (they are paired by construction).

    Thermal → RGB: symmetric — same metric, reversed roles.

    Args:
        rgb_emb:     (N, D) normalised RGB embeddings.
        thermal_emb: (N, D) normalised thermal embeddings.
        k:           number of top retrievals to consider.

    Returns:
        Dict with 'r2t_precision@k' and 't2r_precision@k'.
    """
    N = rgb_emb.size(0)

    # Cosine similarity matrix (N, N) — dot product since L2-normalised
    sim = rgb_emb @ thermal_emb.T  # (N, N)

    # RGB → Thermal: for each row (RGB query), top-K column indices (thermal)
    _, r2t_topk = sim.topk(k, dim=1)  # (N, K)
    # Correct if the paired index appears in top-K
    gt_indices = torch.arange(N).unsqueeze(1)  # (N, 1)
    r2t_hits = (r2t_topk == gt_indices).any(dim=1).float()
    r2t_precision = r2t_hits.mean().item()

    # Thermal → RGB: for each column (thermal query), top-K row indices (RGB)
    _, t2r_topk = sim.T.topk(k, dim=1)  # (N, K)
    t2r_hits = (t2r_topk == gt_indices).any(dim=1).float()
    t2r_precision = t2r_hits.mean().item()

    return {
        f"r2t_precision@{k}": r2t_precision,
        f"t2r_precision@{k}": t2r_precision,
    }


# ── 2. Linear Probe ─────────────────────────────────────────────────────

def load_coco_annotations(data_dir, split="train"):
    """
    Load COCO-format annotations from FLIR ADAS v2.

    Returns a dict mapping image filename → set of category IDs present.
    FLIR ADAS v2 uses 15 object classes (person, car, bicycle, dog, etc.)
    in a modified MSCOCO label format.
    """
    data_dir = Path(data_dir)

    # Try common annotation file locations
    candidates = [
        data_dir / f"images_thermal_{split}" / "coco.json",
        data_dir / f"images_thermal_{split}" / "data" / "coco.json",
        data_dir / f"thermal_{split}.json",
        data_dir / "annotations" / f"thermal_{split}.json",
    ]

    ann_file = None
    for c in candidates:
        if c.exists():
            ann_file = c
            break

    if ann_file is None:
        print(f"[Linear Probe] No annotation file found for {split} — skipping")
        return None, None

    with open(ann_file) as f:
        coco = json.load(f)

    # Build image_id → filename mapping
    id_to_file = {img["id"]: img["file_name"] for img in coco["images"]}

    # Build filename → set of category IDs (scene-level multi-label)
    # Note: coco.json file_name includes 'data/' prefix (e.g., 'data/video-xxx.jpg')
    # so we strip it to match against bare filenames from the dataset pairs
    file_to_cats = defaultdict(set)
    for ann in coco["annotations"]:
        fname = id_to_file.get(ann["image_id"], "")
        bare_name = Path(fname).name  # strip 'data/' prefix
        file_to_cats[bare_name].add(ann["category_id"])

    # Category names
    cat_names = {c["id"]: c["name"] for c in coco.get("categories", [])}

    return file_to_cats, cat_names


def linear_probe(rgb_emb, thermal_emb, data_dir, split="val"):
    """
    Train a linear classifier on concatenated [RGB, thermal] embeddings.

    Uses scene-level labels derived from COCO bounding box annotations:
    each image's label is the dominant (most frequent) object category.
    This tests whether the aligned embedding space captures semantic content.
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import LabelEncoder
        from sklearn.metrics import accuracy_score, classification_report
    except ImportError:
        print("[Linear Probe] scikit-learn not installed — skipping")
        return None

    file_to_cats, cat_names = load_coco_annotations(data_dir, split)
    if file_to_cats is None:
        # Fallback: report that annotations weren't found
        print("[Linear Probe] Annotations not available — returning placeholder")
        return {"accuracy": None, "note": "COCO annotations not found"}

    # For each image, use the first/dominant category as the label
    # This is a coarse scene-level classification
    labels = []
    valid_indices = []
    dataset = FLIRPairedDataset(data_dir, split=split)

    for idx, (_, thermal_path) in enumerate(dataset.pairs):
        fname = thermal_path.name
        cats = file_to_cats.get(fname, set())
        if cats:
            labels.append(min(cats))  # use lowest category ID as primary label
            valid_indices.append(idx)

    if len(valid_indices) < 50:
        print(f"[Linear Probe] Only {len(valid_indices)} labelled images — skipping")
        return {"accuracy": None, "note": f"Only {len(valid_indices)} labelled images"}

    valid_indices = torch.tensor(valid_indices)
    labels = np.array(labels)

    # Concatenate RGB and thermal embeddings → 256-dim feature vector
    features = torch.cat([
        rgb_emb[valid_indices],
        thermal_emb[valid_indices],
    ], dim=1).numpy()

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(labels)

    # Train/test split (80/20) for the linear probe
    n_train = int(0.8 * len(y))
    X_train, X_test = features[:n_train], features[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"[Linear Probe] Accuracy: {acc:.4f} ({len(le.classes_)} classes, "
          f"{len(y_train)} train, {len(y_test)} test)")

    return {
        "accuracy": acc,
        "n_classes": len(le.classes_),
        "n_train": len(y_train),
        "n_test": len(y_test),
    }


# ── 3. t-SNE Visualisation ──────────────────────────────────────────────

def plot_tsne(rgb_emb, thermal_emb, results_dir, n_samples=2000, perplexity=30):
    """
    Generate t-SNE visualisation of RGB and thermal embeddings in the shared space.

    Good cross-spectral alignment = RGB and thermal embeddings of the same scene
    cluster together, forming modality-agnostic clusters rather than two separate
    modality-specific clouds.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    N = min(n_samples, rgb_emb.size(0))

    # Subsample for speed
    perm = torch.randperm(rgb_emb.size(0))[:N]
    rgb_sub = rgb_emb[perm].numpy()
    thermal_sub = thermal_emb[perm].numpy()

    # Stack both modalities for joint t-SNE
    combined = np.concatenate([rgb_sub, thermal_sub], axis=0)  # (2N, D)
    labels = np.array(["RGB"] * N + ["Thermal"] * N)

    print(f"[t-SNE] Running on {2 * N} embeddings (perplexity={perplexity})...")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000)
    coords = tsne.fit_transform(combined)  # (2N, 2)

    # Split back
    rgb_coords = coords[:N]
    thermal_coords = coords[N:]

    # ── Plot 1: Modality-coloured scatter ────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    axes[0].scatter(rgb_coords[:, 0], rgb_coords[:, 1],
                    c="#2196F3", alpha=0.4, s=8, label="RGB")
    axes[0].scatter(thermal_coords[:, 0], thermal_coords[:, 1],
                    c="#FF5722", alpha=0.4, s=8, label="Thermal")
    axes[0].set_title("t-SNE: RGB vs Thermal Embeddings", fontsize=14)
    axes[0].legend(fontsize=12, markerscale=3)
    axes[0].set_xlabel("t-SNE 1")
    axes[0].set_ylabel("t-SNE 2")
    axes[0].grid(True, alpha=0.2)

    # ── Plot 2: Paired connections (first 100 pairs) ─────────────────────
    n_lines = min(100, N)
    axes[1].scatter(rgb_coords[:, 0], rgb_coords[:, 1],
                    c="#2196F3", alpha=0.3, s=8, label="RGB")
    axes[1].scatter(thermal_coords[:, 0], thermal_coords[:, 1],
                    c="#FF5722", alpha=0.3, s=8, label="Thermal")
    for i in range(n_lines):
        axes[1].plot(
            [rgb_coords[i, 0], thermal_coords[i, 0]],
            [rgb_coords[i, 1], thermal_coords[i, 1]],
            "k-", alpha=0.1, linewidth=0.5,
        )
    axes[1].set_title("Paired Connections (same scene)", fontsize=14)
    axes[1].legend(fontsize=12, markerscale=3)
    axes[1].set_xlabel("t-SNE 1")
    axes[1].set_ylabel("t-SNE 2")
    axes[1].grid(True, alpha=0.2)

    plt.tight_layout()
    save_path = Path(results_dir) / "tsne_embeddings.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[t-SNE] Saved to {save_path}")

    return rgb_coords, thermal_coords


# ── 4. Temperature Prediction MAE ────────────────────────────────────────

def temperature_mae(pred_temp, target_temp):
    """
    Mean Absolute Error of the physics decoder's temperature predictions.

    This measures how well the thermal encoder preserves the physically
    meaningful temperature signal in its embeddings.
    """
    mae = torch.abs(pred_temp - target_temp).mean().item()
    print(f"[Physics] Temperature MAE: {mae:.2f} °C")
    return mae


# ── Full Evaluation Pipeline ─────────────────────────────────────────────

def evaluate(cfg: ThermalCLIPConfig = None, checkpoint_path: str = None):
    """Run all four evaluation metrics and save results."""
    if cfg is None:
        cfg = ThermalCLIPConfig()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = ThermalCLIP(cfg).to(device)
    if checkpoint_path is None:
        checkpoint_path = Path(cfg.checkpoint_dir) / "best_model.pt"

    if Path(checkpoint_path).exists():
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[Eval] Loaded checkpoint from {checkpoint_path} (epoch {ckpt.get('epoch', '?')})")
    else:
        print(f"[Eval] No checkpoint found at {checkpoint_path} — using random weights")

    # Load val data
    val_dataset = FLIRPairedDataset(cfg.data_dir, split="val", image_size=cfg.image_size)
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
    )

    # Extract embeddings
    print("[Eval] Extracting embeddings...")
    data = extract_embeddings(model, val_loader, device, cfg.use_amp)

    results = {}

    # 1. Cross-modal retrieval
    print("\n[Eval] Cross-modal retrieval...")
    retrieval = cross_modal_retrieval(data["rgb_emb"], data["thermal_emb"], k=cfg.retrieval_k)
    results["retrieval"] = retrieval
    for k, v in retrieval.items():
        print(f"  {k}: {v:.4f} ({v*100:.1f}%)")

    # 2. Linear probe
    print("\n[Eval] Linear probe...")
    probe = linear_probe(data["rgb_emb"], data["thermal_emb"], cfg.data_dir, split="val")
    results["linear_probe"] = probe

    # 3. t-SNE
    print("\n[Eval] t-SNE visualisation...")
    try:
        plot_tsne(
            data["rgb_emb"], data["thermal_emb"],
            cfg.results_dir, n_samples=cfg.tsne_n_samples, perplexity=cfg.tsne_perplexity,
        )
        results["tsne"] = "saved to results/tsne_embeddings.png"
    except Exception as e:
        print(f"  [Warning] t-SNE failed: {e}")
        results["tsne"] = f"failed: {e}"

    # 4. Temperature MAE
    print("\n[Eval] Temperature prediction...")
    mae = temperature_mae(data["pred_temp"], data["target_temp"])
    results["temperature_mae_celsius"] = mae

    # Save results
    results_path = Path(cfg.results_dir) / "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[Eval] All results saved to {results_path}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate ThermalCLIP")
    parser.add_argument("--data_dir", type=str, default="data/flir_adas_v2")
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    cfg = ThermalCLIPConfig(data_dir=args.data_dir)
    evaluate(cfg, args.checkpoint)
