"""
Dual Encoders for ThermalCLIP
==============================
Two separate ResNet-18 backbones with independent projection heads.

Architecture per branch:
    Image (3×224×224) → ResNet-18 (sans FC) → avgpool → 512-dim
    → Linear(512→256) → LayerNorm → Linear(256→128) → L2-norm → 128-dim embedding

Key design decision — asymmetric initialisation:
    • RGB encoder:     ImageNet-pretrained weights.  RGB images share the same
                       domain as ImageNet (reflected visible light), so transfer
                       learning is appropriate.
    • Thermal encoder: Random initialisation.  ImageNet contains zero thermal
                       imagery.  LWIR thermal cameras measure *emitted heat*
                       (governed by Planck's radiation law), not reflected light.
                       The pixel intensity distributions, texture statistics,
                       and feature hierarchies learned from ImageNet have no
                       physical relevance to thermal infrared — pretrained
                       weights would be actively counterproductive.

The projection heads share the same architecture but NOT the same weights.
Each modality needs its own learned mapping from backbone features to the
shared contrastive embedding space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ProjectionHead(nn.Module):
    """
    MLP projection head: maps backbone features to the shared embedding space.

    Linear(in→hidden) → LayerNorm → ReLU → Linear(hidden→out)

    We use LayerNorm (not BatchNorm) following the CLIP/SimCLR v2 convention:
    LayerNorm is more stable with small effective batch sizes and does not
    introduce train/eval mode discrepancies.
    """

    def __init__(self, in_dim: int = 512, hidden_dim: int = 256, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_dim) backbone features.
        Returns:
            (B, out_dim) L2-normalised embeddings.
        """
        projected = self.net(x)
        # L2-normalise so cosine similarity = dot product in embedding space
        return F.normalize(projected, p=2, dim=-1)


class RGBEncoder(nn.Module):
    """
    RGB branch: ImageNet-pretrained ResNet-18 + projection head.

    Uses pretrained weights because RGB images occupy the same visual domain
    as ImageNet — reflected visible light with standard colour statistics.
    """

    def __init__(self, embedding_dim: int = 128, projection_hidden: int = 256):
        super().__init__()
        # Load pretrained ResNet-18 backbone
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Strip the final FC layer — we replace it with our projection head.
        # ResNet-18's avgpool outputs 512-dim features.
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])  # → (B, 512, 1, 1)
        self.backbone_out_dim = 512

        self.projection = ProjectionHead(
            in_dim=self.backbone_out_dim,
            hidden_dim=projection_hidden,
            out_dim=embedding_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, 224, 224) RGB image tensor, ImageNet-normalised.
        Returns:
            (B, embedding_dim) L2-normalised embedding.
        """
        features = self.backbone(x).flatten(1)  # (B, 512)
        return self.projection(features)


class ThermalEncoder(nn.Module):
    """
    Thermal branch: ResNet-18 trained from scratch + projection head.

    NO ImageNet pretraining — and this is intentional, not an oversight.

    LWIR thermal cameras (8–14 μm wavelength) detect emitted thermal radiation,
    governed by Planck's spectral radiance law:  B(λ,T) ∝ 1/(exp(hc/λkT) − 1).
    The pixel intensity encodes *temperature*, not colour or texture as seen in
    reflected visible light.  ImageNet's learned feature hierarchy (edge detectors,
    texture filters, object parts) is built entirely from reflected-light
    statistics and would impose a harmful inductive bias on thermal features.

    The contrastive loss (InfoNCE) and physics auxiliary loss together provide
    sufficient training signal for the thermal encoder to learn meaningful
    representations from scratch within 30 epochs on 10k+ paired images.
    """

    def __init__(self, embedding_dim: int = 128, projection_hidden: int = 256):
        super().__init__()
        # Random initialisation — no pretrained weights
        backbone = models.resnet18(weights=None)

        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.backbone_out_dim = 512

        self.projection = ProjectionHead(
            in_dim=self.backbone_out_dim,
            hidden_dim=projection_hidden,
            out_dim=embedding_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, 224, 224) thermal image tensor (single channel replicated ×3),
               NOT ImageNet-normalised (raw [0, 1] range).
        Returns:
            (B, embedding_dim) L2-normalised embedding.
        """
        features = self.backbone(x).flatten(1)  # (B, 512)
        return self.projection(features)
