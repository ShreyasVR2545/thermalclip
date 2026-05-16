"""
ThermalCLIP Model
==================
Top-level module that composes both encoders, the physics decoder, and loss.

Architecture overview:
    RGB (3×224×224)  →  RGBEncoder (ResNet-18 + proj)    →  128-dim embedding ─┐
                                                                                ├─ InfoNCE
    Thermal (3×224×224) → ThermalEncoder (ResNet-18 + proj) → 128-dim embedding ─┘
                                                            └─→ PhysicsDecoder → predicted °C

The two encoders share the same projection head architecture but NOT weights.
The physics decoder operates only on the thermal embedding.
"""

import torch
import torch.nn as nn

from config import ThermalCLIPConfig
from encoders import RGBEncoder, ThermalEncoder
from physics import PhysicsDecoder, scene_temperature
from loss import ThermalCLIPLoss


class ThermalCLIP(nn.Module):
    """
    Full ThermalCLIP model: dual encoders + physics decoder + combined loss.
    """

    def __init__(self, cfg: ThermalCLIPConfig):
        super().__init__()
        self.cfg = cfg

        # ── Encoders ─────────────────────────────────────────────────────
        self.rgb_encoder = RGBEncoder(
            embedding_dim=cfg.embedding_dim,
            projection_hidden=cfg.projection_hidden,
        )
        self.thermal_encoder = ThermalEncoder(
            embedding_dim=cfg.embedding_dim,
            projection_hidden=cfg.projection_hidden,
        )

        # ── Physics auxiliary decoder (thermal branch only) ──────────────
        self.physics_decoder = PhysicsDecoder(embedding_dim=cfg.embedding_dim)

        # ── Loss ─────────────────────────────────────────────────────────
        self.criterion = ThermalCLIPLoss(
            init_temperature=cfg.temperature,
            physics_weight=cfg.physics_loss_weight,
        )

    def forward(
        self,
        rgb_images: torch.Tensor,
        thermal_images: torch.Tensor,
        thermal_raw: torch.Tensor,
    ) -> dict:
        """
        Full forward pass: encode both modalities, predict temperature, compute loss.

        Args:
            rgb_images:     (B, 3, 224, 224) ImageNet-normalised RGB.
            thermal_images: (B, 3, 224, 224) thermal (replicated channels).
            thermal_raw:    (B, 1, 224, 224) un-augmented thermal for temperature.

        Returns:
            Dict with embeddings, predictions, and loss components.
        """
        # Encode both modalities → shared 128-dim space
        rgb_emb = self.rgb_encoder(rgb_images)           # (B, 128)
        thermal_emb = self.thermal_encoder(thermal_images)  # (B, 128)

        # Physics auxiliary: predict scene temperature from thermal embedding
        pred_temp = self.physics_decoder(thermal_emb)    # (B,)

        # Ground-truth temperature proxy from raw thermal pixels
        target_temp = scene_temperature(
            thermal_raw,
            temp_min=self.cfg.thermal_temp_min,
            temp_max=self.cfg.thermal_temp_max,
        ).to(pred_temp.device)

        # Combined loss
        losses = self.criterion(rgb_emb, thermal_emb, pred_temp, target_temp)

        return {
            "rgb_emb": rgb_emb,
            "thermal_emb": thermal_emb,
            "pred_temp": pred_temp,
            "target_temp": target_temp,
            **losses,
        }

    def encode_rgb(self, rgb_images: torch.Tensor) -> torch.Tensor:
        """Encode RGB images only (for inference / retrieval)."""
        return self.rgb_encoder(rgb_images)

    def encode_thermal(self, thermal_images: torch.Tensor) -> torch.Tensor:
        """Encode thermal images only (for inference / retrieval)."""
        return self.thermal_encoder(thermal_images)

    def predict_temperature(self, thermal_images: torch.Tensor) -> torch.Tensor:
        """Encode thermal → predict scene temperature (for demo)."""
        emb = self.thermal_encoder(thermal_images)
        return self.physics_decoder(emb)
