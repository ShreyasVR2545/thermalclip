"""
Physics-Informed Thermal Calibration
=====================================
Bridges the gap between raw 8-bit FLIR pixel values and physically meaningful
temperature estimates.

Background — Planck's Radiation Law and LWIR Imaging
-----------------------------------------------------
All objects above absolute zero emit electromagnetic radiation according to
Planck's spectral radiance law:

    B(λ, T) = (2hc² / λ⁵) · 1 / (exp(hc / λkT) - 1)

where h is Planck's constant, c the speed of light, k Boltzmann's constant,
λ the wavelength, and T the absolute temperature in Kelvin.

FLIR LWIR cameras operate in the 8–14 μm atmospheric transmission window.
At these wavelengths and typical outdoor temperatures (250–350 K), the
Stefan-Boltzmann approximation holds: total emitted power ∝ T⁴.  The camera's
microbolometer array measures this incident irradiance and the on-board ADC
maps it to a 16-bit raw count, which is then normalised to 8-bit grayscale
(0–255) in the FLIR ADAS v2 dataset.

Our linear calibration T_celsius ≈ (pixel / 255) × 100 − 20 is a first-order
approximation of this nonlinear physical relationship.  It is intentionally
simple: the goal is to give the thermal encoder a *physics-grounded training
signal* — not a radiometrically exact measurement.  Even this coarse proxy
forces the encoder to preserve the physically meaningful "how hot is this
scene?" signal in its embeddings, rather than collapsing to purely
contrastive features.

Why this matters for the model
-------------------------------
Pure contrastive learning (InfoNCE) can learn an embedding that aligns RGB
and thermal modalities by *any* shared structure — even accidental spatial
correlations.  The physics auxiliary loss adds an inductive bias: the thermal
encoder must also predict scene temperature, so its representations are
anchored in the actual physical quantity the sensor measures (emitted thermal
radiation), not just statistical co-occurrence with RGB.
"""

import torch
import torch.nn as nn


# ── Pixel → Temperature Calibration ─────────────────────────────────────

def pixel_to_celsius(pixel_value: torch.Tensor,
                     temp_min: float = -20.0,
                     temp_max: float = 80.0) -> torch.Tensor:
    """
    Convert 8-bit FLIR thermal pixel values to approximate Celsius.

    Planck's radiation law governs thermal emission:
        B(λ, T) ∝ 1 / (exp(hc / λkT) - 1)
    For LWIR cameras (8–14 μm), the Stefan-Boltzmann approximation gives
    intensity ∝ T⁴.  Our linear calibration is a first-order approximation
    of this relationship, grounding the thermal encoder in physical emission
    principles rather than purely data-driven features.

    Args:
        pixel_value: Raw 8-bit values in [0, 255] or normalised in [0, 1].
        temp_min: Temperature mapped to pixel value 0 (°C).
        temp_max: Temperature mapped to pixel value 255 (°C).

    Returns:
        Approximate scene temperature in °C.
    """
    # If values are normalised to [0, 1] (e.g. by torchvision transforms),
    # scale back to [0, 255] first.
    if pixel_value.max() <= 1.0:
        pixel_value = pixel_value * 255.0

    temp_range = temp_max - temp_min  # 100 °C span
    return (pixel_value / 255.0) * temp_range + temp_min


def scene_temperature(thermal_image: torch.Tensor,
                      temp_min: float = -20.0,
                      temp_max: float = 80.0) -> torch.Tensor:
    """
    Compute a scene-level temperature proxy by averaging pixel temperatures.

    This gives one scalar per image — a coarse but physically grounded label
    for the auxiliary regression head.

    Args:
        thermal_image: (B, C, H, W) tensor of thermal images.
        temp_min: Calibration floor (°C).
        temp_max: Calibration ceiling (°C).

    Returns:
        (B,) tensor of mean scene temperatures in °C.
    """
    # Average over spatial dims and channels → (B,)
    mean_pixel = thermal_image.mean(dim=[1, 2, 3])
    return pixel_to_celsius(mean_pixel, temp_min, temp_max)


# ── Physics-Informed Auxiliary Decoder ───────────────────────────────────

class PhysicsDecoder(nn.Module):
    """
    Lightweight MLP that predicts scene temperature from the thermal embedding.

    Architecture: 128 → 64 → 1
    Trained with MSE loss against the calibrated scene temperature proxy.

    This forces the thermal encoder's 128-dim embedding to retain information
    about the *physical* quantity the LWIR sensor measures (emitted thermal
    radiation intensity), not just whatever features happen to align with RGB
    under the contrastive objective.
    """

    def __init__(self, embedding_dim: int = 128):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, thermal_embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            thermal_embedding: (B, 128) L2-normalised thermal embedding.

        Returns:
            (B,) predicted scene temperature in °C.
        """
        return self.head(thermal_embedding).squeeze(-1)
