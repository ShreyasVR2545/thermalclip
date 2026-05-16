"""
ThermalCLIP Configuration
=========================
All hyperparameters and paths as a frozen dataclass.
Centralised here so experiments are reproducible — change one file, not grep across ten.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ThermalCLIPConfig:
    """Master configuration for ThermalCLIP training and evaluation."""

    # ── Paths ────────────────────────────────────────────────────────────
    data_dir: str = "data/flir_adas_v2"
    results_dir: str = "results"
    checkpoint_dir: str = "checkpoints"
    kaggle_dataset: str = "samdazel/teledyne-flir-adas-thermal-dataset-v2"

    # ── Model ────────────────────────────────────────────────────────────
    backbone: str = "resnet18"
    embedding_dim: int = 128          # final shared embedding space
    projection_hidden: int = 256      # intermediate projection head width
    backbone_out_dim: int = 512       # ResNet-18 avgpool output

    # ── Training ─────────────────────────────────────────────────────────
    epochs: int = 30
    batch_size: int = 64
    num_workers: int = 4
    lr: float = 3e-4
    weight_decay: float = 0.01
    temperature: float = 0.07        # InfoNCE temperature τ
    max_grad_norm: float = 1.0       # gradient clipping threshold

    # Two-phase training
    freeze_rgb_epochs: int = 5       # phase 1: freeze RGB encoder
    rgb_lr_factor: float = 0.1       # phase 2: RGB gets lr * this factor

    # LR schedule: linear warmup → cosine decay
    warmup_epochs: int = 5

    # Physics auxiliary loss
    physics_loss_weight: float = 0.1  # λ for temperature prediction MSE

    # Mixed precision
    use_amp: bool = True

    # ── Data ─────────────────────────────────────────────────────────────
    image_size: int = 224
    # Thermal calibration constants (FLIR LWIR 8-bit → approximate °C)
    # See physics.py for the full derivation from Planck's law
    thermal_temp_min: float = -20.0   # 0 → -20 °C
    thermal_temp_max: float = 80.0    # 255 → 80 °C

    # ── Evaluation ───────────────────────────────────────────────────────
    retrieval_k: int = 5             # Precision@K for cross-modal retrieval
    tsne_perplexity: float = 30.0
    tsne_n_samples: int = 2000       # subsample for t-SNE (speed)
    faiss_nprobe: int = 10

    # ── Gradio demo ──────────────────────────────────────────────────────
    gallery_size: int = 1000         # number of val images in FAISS index
    demo_port: int = 7860

    def __post_init__(self):
        """Ensure output directories exist."""
        Path(self.results_dir).mkdir(parents=True, exist_ok=True)
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
