"""
Loss Functions for ThermalCLIP
===============================
Implements the symmetric InfoNCE contrastive loss (identical to CLIP's training
objective) and combines it with the physics-informed temperature regression loss.

Symmetric InfoNCE — why symmetric?
-----------------------------------
We want *bidirectional* cross-modal retrieval: both RGB→thermal and thermal→RGB.
A one-directional loss (e.g., only computing cross-entropy for RGB queries against
thermal keys) would bias the embedding space toward one retrieval direction.
Averaging both directions — exactly as done in Radford et al., 2021 (CLIP) —
ensures the learned similarity is symmetric: sim(rgb, thermal) = sim(thermal, rgb).

Implicit negatives via in-batch sampling
-----------------------------------------
For a batch of B paired images, each pair (rgb_i, thermal_i) is a positive.
The remaining B−1 thermal images in the batch serve as implicit negatives for
rgb_i (and vice versa).  This is memory-efficient — no explicit negative mining,
no momentum encoder, no memory bank.  Larger batch sizes → more negatives →
harder contrastive signal → better representations, which is why we use B=64.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SymmetricInfoNCELoss(nn.Module):
    """
    Symmetric InfoNCE loss — the CLIP training objective applied to thermal–RGB pairs.

    Given L2-normalised embeddings rgb_emb and thermal_emb of shape (B, D):
        logits = (rgb_emb @ thermal_emb.T) * exp(logit_scale)  → (B, B) similarity matrix
        labels = [0, 1, 2, ..., B-1]                            → diagonal = positive pairs
        loss   = (CE(logits, labels) + CE(logits.T, labels)) / 2

    Learnable temperature (following the original CLIP paper):
        The temperature τ is parameterised as logit_scale = log(1/τ), learned via
        gradient descent, and clamped to [0, log(100)] to prevent training instability.
        This is exactly the scheme from Radford et al. (2021): "The learnable temperature
        parameter τ was initialized to the equivalent of 0.07 from (Wu et al., 2018) and
        clipped to prevent scaling the logits by more than 100."
        Learning τ removes a sensitive hyperparameter and lets the model adapt the
        softmax sharpness to the difficulty of the cross-spectral discrimination task.
    """

    def __init__(self, init_temperature: float = 0.07):
        super().__init__()
        # Learnable log-scale parameter: logit_scale = log(1/τ), so exp(logit_scale) = 1/τ
        # Initialised to log(1/0.07) ≈ 2.659
        import numpy as np
        self.logit_scale = nn.Parameter(
            torch.ones([]) * np.log(1.0 / init_temperature)
        )

    def forward(
        self,
        rgb_emb: torch.Tensor,
        thermal_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute symmetric InfoNCE loss.

        Args:
            rgb_emb:     (B, D) L2-normalised RGB embeddings.
            thermal_emb: (B, D) L2-normalised thermal embeddings.

        Returns:
            Scalar loss (average of both directions).
        """
        B = rgb_emb.size(0)
        device = rgb_emb.device

        # Clamp logit_scale to [0, log(100)] for training stability (CLIP paper)
        logit_scale = self.logit_scale.exp().clamp(max=100.0)

        # Cosine similarity matrix scaled by learned temperature
        # Since embeddings are L2-normalised, dot product = cosine similarity
        logits = (rgb_emb @ thermal_emb.T) * logit_scale  # (B, B)

        # Ground truth: the diagonal entries are the positive pairs
        labels = torch.arange(B, device=device)

        # RGB → Thermal direction: for each RGB query, which thermal is the match?
        loss_r2t = F.cross_entropy(logits, labels)

        # Thermal → RGB direction: for each thermal query, which RGB is the match?
        loss_t2r = F.cross_entropy(logits.T, labels)

        # Symmetric: average both directions
        return (loss_r2t + loss_t2r) / 2.0


class ThermalCLIPLoss(nn.Module):
    """
    Combined loss: symmetric InfoNCE + physics-informed temperature regression.

    L_total = L_InfoNCE + λ · L_temp_MSE

    The physics loss provides an auxiliary training signal that anchors the
    thermal encoder's representations in the physical quantity measured by the
    LWIR sensor (scene temperature), preventing the contrastive objective from
    collapsing to a trivially aligned but physically meaningless embedding.
    """

    def __init__(self, init_temperature: float = 0.07, physics_weight: float = 0.1):
        super().__init__()
        self.info_nce = SymmetricInfoNCELoss(init_temperature)
        self.physics_weight = physics_weight
        self.mse = nn.MSELoss()

    def forward(
        self,
        rgb_emb: torch.Tensor,
        thermal_emb: torch.Tensor,
        pred_temp: torch.Tensor,
        target_temp: torch.Tensor,
    ) -> dict:
        """
        Args:
            rgb_emb:     (B, D) L2-normalised RGB embeddings.
            thermal_emb: (B, D) L2-normalised thermal embeddings.
            pred_temp:   (B,) predicted scene temperature from physics decoder.
            target_temp: (B,) pseudo-ground-truth temperature from calibration.

        Returns:
            Dict with 'total', 'info_nce', 'physics' losses for logging.
        """
        loss_nce = self.info_nce(rgb_emb, thermal_emb)
        loss_physics = self.mse(pred_temp, target_temp)
        loss_total = loss_nce + self.physics_weight * loss_physics

        return {
            "total": loss_total,
            "info_nce": loss_nce.detach(),
            "physics": loss_physics.detach(),
        }
