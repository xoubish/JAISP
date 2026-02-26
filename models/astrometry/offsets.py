"""
Synthetic astrometric offset generation and image warping.

Provides smooth offset fields for self-supervised training:
  - constant: uniform shift across the tile (simplest)
  - affine: linear field (rotation + scale + shear)
  - smooth: sum of low-frequency sinusoids (higher-order distortions)

The curriculum ramps from constant → affine → smooth as training progresses.
"""

import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Offset field generators (all return numpy arrays in arcseconds)
# ---------------------------------------------------------------------------

def generate_constant_offset(
    H: int,
    W: int,
    max_amp: float,
    rng: Optional[np.random.RandomState] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Uniform (Δx, Δy) shift in arcseconds."""
    if rng is None:
        rng = np.random.RandomState()
    dx = rng.uniform(-max_amp, max_amp)
    dy = rng.uniform(-max_amp, max_amp)
    dra = np.full((H, W), dx, dtype=np.float32)
    ddec = np.full((H, W), dy, dtype=np.float32)
    return dra, ddec


def generate_affine_offset(
    H: int,
    W: int,
    max_amp: float,
    rng: Optional[np.random.RandomState] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Linear offset field: a + b*x + c*y for each component."""
    if rng is None:
        rng = np.random.RandomState()

    y = np.linspace(-1, 1, H, dtype=np.float32)
    x = np.linspace(-1, 1, W, dtype=np.float32)
    Y, X = np.meshgrid(y, x, indexing="ij")

    # Random coefficients scaled so max offset ≈ max_amp.
    a_ra, b_ra, c_ra = rng.uniform(-1, 1, 3) * max_amp * 0.5
    a_de, b_de, c_de = rng.uniform(-1, 1, 3) * max_amp * 0.5

    dra = (a_ra + b_ra * X + c_ra * Y).astype(np.float32)
    ddec = (a_de + b_de * X + c_de * Y).astype(np.float32)
    return dra, ddec


def generate_smooth_offset(
    H: int,
    W: int,
    max_amp: float,
    n_modes: int = 5,
    rng: Optional[np.random.RandomState] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sum of low-frequency 2D sinusoids — naturally smooth and slowly varying."""
    if rng is None:
        rng = np.random.RandomState()

    y = np.linspace(0, 1, H, dtype=np.float64)
    x = np.linspace(0, 1, W, dtype=np.float64)
    Y, X = np.meshgrid(y, x, indexing="ij")

    dra = np.zeros((H, W), dtype=np.float64)
    ddec = np.zeros((H, W), dtype=np.float64)

    for _ in range(n_modes):
        # Low spatial frequency: 0.5–3 cycles across the field.
        fx = rng.uniform(0.5, 3.0)
        fy = rng.uniform(0.5, 3.0)
        phase1 = rng.uniform(0, 2 * math.pi)
        phase2 = rng.uniform(0, 2 * math.pi)
        amp = rng.uniform(0, max_amp / n_modes)

        dra += amp * np.sin(2 * math.pi * fx * X + phase1) * np.cos(2 * math.pi * fy * Y + phase2)
        ddec += amp * np.cos(2 * math.pi * fx * X + rng.uniform(0, math.pi)) * np.sin(
            2 * math.pi * fy * Y + phase2
        )

    return dra.astype(np.float32), ddec.astype(np.float32)


def generate_offset_field(
    H: int,
    W: int,
    max_amp: float = 0.5,
    mode: str = "smooth",
    rng: Optional[np.random.RandomState] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a synthetic offset field.

    Args:
        H, W: spatial dimensions (typically Rubin pixel grid)
        max_amp: maximum offset amplitude in arcseconds
        mode: 'constant', 'affine', or 'smooth'
        rng: random state for reproducibility

    Returns:
        dra, ddec: [H, W] arrays in arcseconds
    """
    if mode == "constant":
        return generate_constant_offset(H, W, max_amp, rng)
    elif mode == "affine":
        return generate_affine_offset(H, W, max_amp, rng)
    elif mode == "smooth":
        return generate_smooth_offset(H, W, max_amp, rng=rng)
    else:
        raise ValueError(f"Unknown offset mode: {mode}")


def sample_offset_mode(
    epoch: int,
    curriculum_epochs: int = 10,
    rng: Optional[np.random.RandomState] = None,
) -> str:
    """
    Curriculum sampling: start simple, increase complexity.

    epoch 1..curriculum/3       → mostly constant
    epoch curriculum/3..2/3     → mostly affine
    epoch 2/3..end              → mostly smooth
    After curriculum_epochs      → target distribution (0.2/0.3/0.5)
    """
    if rng is None:
        rng = np.random.RandomState()

    if epoch >= curriculum_epochs:
        p_const, p_affine, p_smooth = 0.2, 0.3, 0.5
    else:
        alpha = epoch / max(1, curriculum_epochs)
        p_const = max(0.1, 0.7 - 0.5 * alpha)
        p_affine = 0.2 + 0.1 * alpha
        p_smooth = 1.0 - p_const - p_affine

    r = rng.rand()
    if r < p_const:
        return "constant"
    elif r < p_const + p_affine:
        return "affine"
    return "smooth"


# ---------------------------------------------------------------------------
# Image warping
# ---------------------------------------------------------------------------

def apply_offset_to_image(
    image: torch.Tensor,
    dra: torch.Tensor,
    ddec: torch.Tensor,
    pixel_scale: float,
) -> torch.Tensor:
    """
    Warp an image by applying an astrometric offset field.

    The image content is shifted so that a source at sky position (α, δ)
    now appears at position (α - Δα, δ - Δδ) in the image.
    The model's job is to predict (Δα, Δδ) to undo this.

    Args:
        image: [B, 1, H, W]
        dra:   [H, W] or [B, 1, H, W] offset in arcseconds
        ddec:  [H, W] or [B, 1, H, W] offset in arcseconds
        pixel_scale: arcsec/pixel of this image

    Returns:
        warped image [B, 1, H, W]
    """
    B, C, H, W = image.shape
    device = image.device

    # Convert arcseconds → pixels.
    if isinstance(dra, np.ndarray):
        dra = torch.from_numpy(dra).to(device)
    if isinstance(ddec, np.ndarray):
        ddec = torch.from_numpy(ddec).to(device)

    dx_pix = dra.float() / pixel_scale   # ΔRA*  → pixels in x
    dy_pix = ddec.float() / pixel_scale   # ΔDec → pixels in y

    # Ensure [H, W] shape.
    if dx_pix.dim() == 4:
        dx_pix = dx_pix[0, 0]
    if dy_pix.dim() == 4:
        dy_pix = dy_pix[0, 0]

    # Resize offset to image size if needed (offset may be on a different grid).
    if dx_pix.shape[0] != H or dx_pix.shape[1] != W:
        dx_pix = F.interpolate(
            dx_pix.unsqueeze(0).unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False
        ).squeeze(0).squeeze(0)
        dy_pix = F.interpolate(
            dy_pix.unsqueeze(0).unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False
        ).squeeze(0).squeeze(0)

    # Base sampling grid in normalized coordinates [-1, 1].
    yy = torch.linspace(-1, 1, H, device=device, dtype=torch.float32)
    xx = torch.linspace(-1, 1, W, device=device, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")

    # Shift: sample at (x + dx, y + dy) → content appears shifted by (-dx, -dy).
    dx_norm = dx_pix * 2.0 / W
    dy_norm = dy_pix * 2.0 / H

    grid = torch.stack([grid_x + dx_norm, grid_y + dy_norm], dim=-1)  # [H, W, 2]
    grid = grid.unsqueeze(0).expand(B, -1, -1, -1)

    return F.grid_sample(image, grid, mode="bilinear", padding_mode="border", align_corners=True)


def resample_offset_to_grid(
    dra: np.ndarray,
    ddec: np.ndarray,
    target_h: int,
    target_w: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Resample offset field to a target grid size (e.g. common token grid)."""
    dra_t = torch.from_numpy(dra).float().unsqueeze(0).unsqueeze(0)
    ddec_t = torch.from_numpy(ddec).float().unsqueeze(0).unsqueeze(0)

    dra_out = F.interpolate(dra_t, size=(target_h, target_w), mode="bilinear", align_corners=False)
    ddec_out = F.interpolate(ddec_t, size=(target_h, target_w), mode="bilinear", align_corners=False)

    return dra_out.squeeze(0), ddec_out.squeeze(0)  # [1, Ht, Wt]
