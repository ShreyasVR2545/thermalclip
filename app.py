"""
ThermalCLIP Gradio Demo
========================
Bidirectional cross-modal retrieval demo:
    • Upload RGB   → retrieve top-5 most similar thermal images
    • Upload thermal → retrieve top-5 most similar RGB images

Also shows: predicted scene temperature, similarity scores,
and query position highlighted on the t-SNE embedding space.

This is the visual proof of cross-spectral alignment.
"""

import os
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.amp import autocast

from config import ThermalCLIPConfig
from dataset import get_rgb_transform, get_thermal_transform
from model import ThermalCLIP
from physics import scene_temperature


class ThermalCLIPRetriever:
    """
    Inference wrapper: builds an in-memory index of gallery embeddings
    and performs nearest-neighbour retrieval using cosine similarity.
    """

    def __init__(self, cfg: ThermalCLIPConfig, checkpoint_path: str = None):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model
        self.model = ThermalCLIP(cfg).to(self.device)
        if checkpoint_path is None:
            checkpoint_path = Path(cfg.checkpoint_dir) / "best_model.pt"

        if Path(checkpoint_path).exists():
            ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt["model_state_dict"])
            print(f"[Demo] Loaded checkpoint from {checkpoint_path}")
        else:
            print(f"[Demo] No checkpoint found — using random weights (demo mode)")

        self.model.eval()

        # Transforms
        self.rgb_transform = get_rgb_transform(cfg.image_size, is_train=False)
        self.thermal_transform = get_thermal_transform(cfg.image_size, is_train=False)

        # Gallery data (populated by build_index)
        self.rgb_embeddings = None
        self.thermal_embeddings = None
        self.rgb_paths = []
        self.thermal_paths = []
        self.tsne_coords = None

    def build_index(self, max_gallery: int = None):
        """
        Build embedding index over the val set for retrieval.

        Uses simple cosine similarity (dot product on L2-normed vectors).
        For larger galleries, replace with FAISS for sub-linear search.
        """
        from dataset import FLIRPairedDataset

        if max_gallery is None:
            max_gallery = self.cfg.gallery_size

        dataset = FLIRPairedDataset(
            self.cfg.data_dir, split="val", image_size=self.cfg.image_size
        )

        n = min(max_gallery, len(dataset))
        print(f"[Demo] Building index over {n} val pairs...")

        rgb_embs, thermal_embs = [], []

        with torch.no_grad():
            for i in range(n):
                rgb_tensor, thermal_tensor, thermal_raw, _ = dataset[i]
                rgb_path, thermal_path = dataset.pairs[i]

                self.rgb_paths.append(str(rgb_path))
                self.thermal_paths.append(str(thermal_path))

                rgb_in = rgb_tensor.unsqueeze(0).to(self.device)
                thermal_in = thermal_tensor.unsqueeze(0).to(self.device)

                with autocast('cuda', enabled=self.cfg.use_amp):
                    r_emb = self.model.encode_rgb(rgb_in)
                    t_emb = self.model.encode_thermal(thermal_in)

                rgb_embs.append(r_emb.float().cpu())
                thermal_embs.append(t_emb.float().cpu())

                if (i + 1) % 200 == 0:
                    print(f"  Indexed {i+1}/{n}")

        self.rgb_embeddings = torch.cat(rgb_embs)      # (N, 128)
        self.thermal_embeddings = torch.cat(thermal_embs)  # (N, 128)
        print(f"[Demo] Index built: {n} pairs")

    @torch.no_grad()
    def query_rgb(self, image: Image.Image, k: int = 5):
        """
        RGB → Thermal retrieval.
        Given an RGB image, find the top-K most similar thermal images.
        """
        if self.thermal_embeddings is None:
            return [], [], None

        # Encode query
        rgb_tensor = self.rgb_transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        with autocast('cuda', enabled=self.cfg.use_amp):
            query_emb = self.model.encode_rgb(rgb_tensor).float().cpu()

        # Cosine similarity against all thermal gallery embeddings
        sims = (query_emb @ self.thermal_embeddings.T).squeeze(0)  # (N,)
        topk_sims, topk_idx = sims.topk(k)

        results = []
        for sim, idx in zip(topk_sims.tolist(), topk_idx.tolist()):
            thermal_img = Image.open(self.thermal_paths[idx])
            results.append((thermal_img, f"Score: {sim:.3f}"))

        return results, topk_sims.tolist(), query_emb

    @torch.no_grad()
    def query_thermal(self, image: Image.Image, k: int = 5):
        """
        Thermal → RGB retrieval.
        Given a thermal image, find the top-K most similar RGB images.
        """
        if self.rgb_embeddings is None:
            return [], [], None, None

        # Encode query
        thermal_gray = image.convert("L")
        thermal_tensor = self.thermal_transform(thermal_gray)
        thermal_3ch = thermal_tensor.repeat(3, 1, 1).unsqueeze(0).to(self.device)

        # Also get raw for temperature prediction
        from torchvision import transforms
        raw_tensor = transforms.Compose([
            transforms.Resize((self.cfg.image_size, self.cfg.image_size)),
            transforms.ToTensor(),
        ])(thermal_gray).unsqueeze(0).to(self.device)

        with autocast('cuda', enabled=self.cfg.use_amp):
            query_emb = self.model.encode_thermal(thermal_3ch).float()
            pred_temp = self.model.physics_decoder(query_emb).item()

        query_emb_cpu = query_emb.cpu()

        # Cosine similarity against all RGB gallery embeddings
        sims = (query_emb_cpu @ self.rgb_embeddings.T).squeeze(0)
        topk_sims, topk_idx = sims.topk(k)

        results = []
        for sim, idx in zip(topk_sims.tolist(), topk_idx.tolist()):
            rgb_img = Image.open(self.rgb_paths[idx])
            results.append((rgb_img, f"Score: {sim:.3f}"))

        return results, topk_sims.tolist(), query_emb_cpu, pred_temp


def create_demo(cfg: ThermalCLIPConfig = None, checkpoint_path: str = None):
    """Create and launch the Gradio demo interface."""
    try:
        import gradio as gr
    except ImportError:
        print("ERROR: gradio not installed. Run: pip install gradio")
        return

    if cfg is None:
        cfg = ThermalCLIPConfig()

    retriever = ThermalCLIPRetriever(cfg, checkpoint_path)

    # Build index (may take a minute)
    try:
        retriever.build_index()
    except Exception as e:
        print(f"[Demo] Could not build index: {e}")
        print("[Demo] Demo will run but retrieval won't work until data is available")

    def rgb_to_thermal(image):
        """Handle RGB → Thermal query."""
        if image is None:
            return [], "No image provided"
        results, scores, _ = retriever.query_rgb(Image.fromarray(image), k=5)
        if not results:
            return [], "Index not built — run with dataset available"
        info = f"Top-5 retrieval scores: {', '.join(f'{s:.3f}' for s in scores)}"
        gallery = [(np.array(img), caption) for img, caption in results]
        return gallery, info

    def thermal_to_rgb(image):
        """Handle Thermal → RGB query."""
        if image is None:
            return [], "No image provided"
        results, scores, _, pred_temp = retriever.query_thermal(
            Image.fromarray(image), k=5
        )
        if not results:
            return [], "Index not built — run with dataset available"
        info = (
            f"Predicted scene temperature: {pred_temp:.1f} °C\n"
            f"Top-5 retrieval scores: {', '.join(f'{s:.3f}' for s in scores)}"
        )
        gallery = [(np.array(img), caption) for img, caption in results]
        return gallery, info

    # Build Gradio interface
    with gr.Blocks(
        title="ThermalCLIP — Cross-Spectral Retrieval Demo",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            """
            # 🌡️ ThermalCLIP — Cross-Spectral Vision Alignment
            **Bidirectional retrieval between thermal infrared and RGB images.**

            Upload an RGB image to find matching thermal images, or upload a thermal
            image to find matching RGB images. The system uses a CLIP-style dual encoder
            trained with symmetric InfoNCE loss and a physics-informed auxiliary decoder.
            """
        )

        with gr.Tab("RGB → Thermal"):
            with gr.Row():
                with gr.Column(scale=1):
                    rgb_input = gr.Image(label="Upload RGB Image", type="numpy")
                    rgb_btn = gr.Button("Find Matching Thermal Images", variant="primary")
                with gr.Column(scale=2):
                    rgb_gallery = gr.Gallery(label="Top-5 Thermal Matches",
                                             columns=5, height=300)
                    rgb_info = gr.Textbox(label="Retrieval Info", lines=2)

            rgb_btn.click(rgb_to_thermal, inputs=rgb_input, outputs=[rgb_gallery, rgb_info])

        with gr.Tab("Thermal → RGB"):
            with gr.Row():
                with gr.Column(scale=1):
                    thermal_input = gr.Image(label="Upload Thermal Image", type="numpy")
                    thermal_btn = gr.Button("Find Matching RGB Images", variant="primary")
                with gr.Column(scale=2):
                    thermal_gallery = gr.Gallery(label="Top-5 RGB Matches",
                                                 columns=5, height=300)
                    thermal_info = gr.Textbox(label="Retrieval Info + Temperature", lines=3)

            thermal_btn.click(thermal_to_rgb, inputs=thermal_input,
                              outputs=[thermal_gallery, thermal_info])

        gr.Markdown(
            """
            ---
            **How it works:** ThermalCLIP uses separate ResNet-18 encoders for each
            modality, projecting into a shared 128-dim embedding space via symmetric
            InfoNCE contrastive loss. The thermal encoder also has a physics-informed
            auxiliary head that predicts scene temperature, grounded in Planck's radiation
            law and the Stefan-Boltzmann approximation for LWIR imaging.
            """
        )

    demo.launch(server_port=cfg.demo_port, share=False)
    return demo


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ThermalCLIP Gradio Demo")
    parser.add_argument("--data_dir", type=str, default="data/flir_adas_v2")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    cfg = ThermalCLIPConfig(data_dir=args.data_dir, demo_port=args.port)
    create_demo(cfg, args.checkpoint)
