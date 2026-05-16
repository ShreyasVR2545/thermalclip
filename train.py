"""
ThermalCLIP Training Loop
==========================
Two-phase training with mixed precision, gradient clipping, and cosine LR decay.

Phase 1 (epochs 1–5):
    Freeze the RGB encoder entirely.  The thermal encoder starts from random
    init and needs to "catch up" — training both simultaneously would let the
    already-converged RGB encoder dominate the contrastive loss, causing the
    thermal branch to learn trivial features.

Phase 2 (epochs 6–30):
    Unfreeze both encoders with differential learning rates:
    • Thermal encoder + physics decoder: full lr (3e-4)
    • RGB encoder: 0.1× lr (3e-5)
    The RGB encoder is already pretrained; a high learning rate would destroy
    its ImageNet representations.  The lower rate allows fine-tuning without
    catastrophic forgetting.

LR Schedule: linear warmup (5 epochs) → cosine decay to 0.
"""

import os
import sys
import time
import math
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

from config import ThermalCLIPConfig
from dataset import FLIRPairedDataset
from model import ThermalCLIP


def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    """Linear warmup → cosine decay to 0, matching CLIP's original schedule."""
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def freeze_rgb_encoder(model):
    """Phase 1: freeze RGB encoder so thermal encoder can catch up from random init."""
    for param in model.rgb_encoder.parameters():
        param.requires_grad = False
    print("[Phase 1] RGB encoder frozen — thermal encoder training from scratch")


def unfreeze_rgb_encoder(model):
    """Phase 2: unfreeze RGB encoder for joint fine-tuning."""
    for param in model.rgb_encoder.parameters():
        param.requires_grad = True
    print("[Phase 2] RGB encoder unfrozen — joint training with differential LR")


def build_optimizer(model, cfg, phase=1):
    """
    Build AdamW optimizer with parameter groups.

    Phase 1: only thermal encoder + physics decoder + logit_scale are trainable.
    Phase 2: add RGB encoder at reduced learning rate.
    """
    param_groups = [
        # Thermal encoder + projection head
        {
            "params": list(model.thermal_encoder.parameters()),
            "lr": cfg.lr,
            "name": "thermal_encoder",
        },
        # Physics decoder
        {
            "params": list(model.physics_decoder.parameters()),
            "lr": cfg.lr,
            "name": "physics_decoder",
        },
        # Learnable temperature (logit_scale) — part of the loss module
        {
            "params": [model.criterion.info_nce.logit_scale],
            "lr": cfg.lr,
            "name": "logit_scale",
        },
    ]

    if phase == 2:
        param_groups.append({
            "params": list(model.rgb_encoder.parameters()),
            "lr": cfg.lr * cfg.rgb_lr_factor,
            "name": "rgb_encoder",
        })

    return torch.optim.AdamW(param_groups, weight_decay=cfg.weight_decay)


def train_one_epoch(model, dataloader, optimizer, scheduler, scaler, cfg, device, epoch):
    """Single training epoch with mixed precision and gradient clipping."""
    model.train()
    total_loss = 0.0
    total_nce = 0.0
    total_physics = 0.0
    n_batches = 0

    for batch_idx, (rgb, thermal, thermal_raw, _) in enumerate(dataloader):
        rgb = rgb.to(device, non_blocking=True)
        thermal = thermal.to(device, non_blocking=True)
        thermal_raw = thermal_raw.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Mixed precision forward pass
        with autocast('cuda', enabled=cfg.use_amp):
            outputs = model(rgb, thermal, thermal_raw)

        loss = outputs["total"]

        # Mixed precision backward pass
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += outputs["total"].item()
        total_nce += outputs["info_nce"].item()
        total_physics += outputs["physics"].item()
        n_batches += 1

        if batch_idx % 50 == 0:
            logit_scale_val = model.criterion.info_nce.logit_scale.exp().item()
            print(
                f"  Epoch {epoch} [{batch_idx}/{len(dataloader)}] "
                f"loss={outputs['total'].item():.4f} "
                f"nce={outputs['info_nce'].item():.4f} "
                f"physics={outputs['physics'].item():.4f} "
                f"temp={1.0/logit_scale_val:.4f} "
                f"lr={optimizer.param_groups[0]['lr']:.6f}"
            )

    return {
        "loss": total_loss / n_batches,
        "info_nce": total_nce / n_batches,
        "physics": total_physics / n_batches,
    }


@torch.no_grad()
def validate(model, dataloader, cfg, device):
    """Validation pass — compute losses without gradients."""
    model.eval()
    total_loss = 0.0
    total_nce = 0.0
    total_physics = 0.0
    n_batches = 0

    for rgb, thermal, thermal_raw, _ in dataloader:
        rgb = rgb.to(device, non_blocking=True)
        thermal = thermal.to(device, non_blocking=True)
        thermal_raw = thermal_raw.to(device, non_blocking=True)

        with autocast('cuda', enabled=cfg.use_amp):
            outputs = model(rgb, thermal, thermal_raw)

        total_loss += outputs["total"].item()
        total_nce += outputs["info_nce"].item()
        total_physics += outputs["physics"].item()
        n_batches += 1

    return {
        "loss": total_loss / n_batches,
        "info_nce": total_nce / n_batches,
        "physics": total_physics / n_batches,
    }


def train(cfg: ThermalCLIPConfig = None):
    """Full training pipeline: two-phase training with logging and checkpoints."""
    if cfg is None:
        cfg = ThermalCLIPConfig()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ThermalCLIP] Device: {device}")
    print(f"[ThermalCLIP] Config: epochs={cfg.epochs}, batch_size={cfg.batch_size}, "
          f"lr={cfg.lr}, temperature_init={cfg.temperature}")

    # ── Data ─────────────────────────────────────────────────────────────
    train_dataset = FLIRPairedDataset(cfg.data_dir, split="train", image_size=cfg.image_size)
    val_dataset = FLIRPairedDataset(cfg.data_dir, split="val", image_size=cfg.image_size)

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
    )

    # ── Model ────────────────────────────────────────────────────────────
    model = ThermalCLIP(cfg).to(device)
    scaler = GradScaler('cuda', enabled=cfg.use_amp)

    # ── Phase 1: Freeze RGB encoder ──────────────────────────────────────
    freeze_rgb_encoder(model)
    optimizer = build_optimizer(model, cfg, phase=1)

    steps_per_epoch = len(train_loader)
    total_steps = cfg.epochs * steps_per_epoch
    warmup_steps = cfg.warmup_epochs * steps_per_epoch
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # ── Training log ─────────────────────────────────────────────────────
    history = {"train": [], "val": []}
    best_val_loss = float("inf")

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()

        # ── Phase transition: unfreeze RGB at epoch freeze_rgb_epochs + 1 ──
        if epoch == cfg.freeze_rgb_epochs + 1:
            unfreeze_rgb_encoder(model)
            # Rebuild optimizer with RGB params at lower LR
            optimizer = build_optimizer(model, cfg, phase=2)
            # Rebuild scheduler from current step
            remaining_steps = (cfg.epochs - epoch + 1) * steps_per_epoch
            warmup_remaining = 0  # no re-warmup needed
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, warmup_remaining, remaining_steps
            )

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, cfg, device, epoch
        )

        # Validate
        val_metrics = validate(model, val_loader, cfg, device)

        elapsed = time.time() - t0
        logit_scale_val = model.criterion.info_nce.logit_scale.exp().item()
        print(
            f"Epoch {epoch}/{cfg.epochs} ({elapsed:.1f}s) | "
            f"Train: loss={train_metrics['loss']:.4f} nce={train_metrics['info_nce']:.4f} "
            f"physics={train_metrics['physics']:.4f} | "
            f"Val: loss={val_metrics['loss']:.4f} nce={val_metrics['info_nce']:.4f} "
            f"physics={val_metrics['physics']:.4f} | "
            f"τ={1.0/logit_scale_val:.4f}"
        )

        history["train"].append(train_metrics)
        history["val"].append(val_metrics)

        # ── Checkpoint best model ────────────────────────────────────────
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            ckpt_path = Path(cfg.checkpoint_dir) / "best_model.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "config": cfg,
            }, ckpt_path)
            print(f"  ✓ Saved best model (val_loss={best_val_loss:.4f})")

    # ── Save final model and training history ────────────────────────────
    final_path = Path(cfg.checkpoint_dir) / "final_model.pt"
    torch.save({
        "epoch": cfg.epochs,
        "model_state_dict": model.state_dict(),
        "config": cfg,
    }, final_path)

    history_path = Path(cfg.results_dir) / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[ThermalCLIP] Training complete. History saved to {history_path}")

    # ── Plot training curves ─────────────────────────────────────────────
    try:
        plot_training_curves(history, cfg.results_dir)
    except Exception as e:
        print(f"[Warning] Could not plot training curves: {e}")

    return model, history


def plot_training_curves(history, results_dir):
    """Generate publication-quality training curves."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = range(1, len(history["train"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Total loss
    axes[0].plot(epochs, [m["loss"] for m in history["train"]], "b-", label="Train")
    axes[0].plot(epochs, [m["loss"] for m in history["val"]], "r--", label="Val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Total Loss")
    axes[0].set_title("Total Loss (InfoNCE + λ·Physics)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # InfoNCE loss
    axes[1].plot(epochs, [m["info_nce"] for m in history["train"]], "b-", label="Train")
    axes[1].plot(epochs, [m["info_nce"] for m in history["val"]], "r--", label="Val")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("InfoNCE Loss")
    axes[1].set_title("Symmetric InfoNCE Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Physics MSE
    axes[2].plot(epochs, [m["physics"] for m in history["train"]], "b-", label="Train")
    axes[2].plot(epochs, [m["physics"] for m in history["val"]], "r--", label="Val")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Temperature MSE (°C²)")
    axes[2].set_title("Physics Decoder Loss")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    # Mark phase transition
    for ax in axes:
        ax.axvline(x=5, color="gray", linestyle=":", alpha=0.5, label="Phase 1→2")

    plt.tight_layout()
    plt.savefig(Path(results_dir) / "training_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[ThermalCLIP] Training curves saved to {results_dir}/training_curves.png")


if __name__ == "__main__":
    cfg = ThermalCLIPConfig()

    # Allow overrides from command line for quick experiments
    import argparse
    parser = argparse.ArgumentParser(description="Train ThermalCLIP")
    parser.add_argument("--data_dir", type=str, default=cfg.data_dir)
    parser.add_argument("--epochs", type=int, default=cfg.epochs)
    parser.add_argument("--batch_size", type=int, default=cfg.batch_size)
    parser.add_argument("--lr", type=float, default=cfg.lr)
    parser.add_argument("--num_workers", type=int, default=cfg.num_workers)
    args = parser.parse_args()

    cfg.data_dir = args.data_dir
    cfg.epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.lr = args.lr
    cfg.num_workers = args.num_workers

    train(cfg)
