"""
Batched stamp extraction and local sky background estimation.

Key design: extract N stamps from a [C, H, W] tile in a single F.grid_sample
call — no tile duplication, O(N·S²) memory.

Usage
-----
    stamps = extract_stamps(tile, positions_px, stamp_size=21)
    # tile: [C, H, W] tensor, positions_px: [N, 2] (x, y)
    # returns [N, C, S, S]

    bg = estimate_local_background(stamps, inner_radius=7.0, outer_radius=9.5)
    # returns [N, C] sky level per source per band
"""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def extract_stamps(
    tile: torch.Tensor,          # [C, H, W]
    positions_px: torch.Tensor,  # [N, 2]  (x, y) in pixel coords
    stamp_size: int = 21,
    chunk_size: Optional[int] = None,
) -> torch.Tensor:
    """
    Extract N postage stamps from a multi-band tile using a single grid_sample.

    The trick: lay out an (N×S, S) sampling grid so grid_sample sees it as a
    single "image" → [1, C, N*S, S] output, then reshape to [N, C, S, S].

    Parameters
    ----------
    tile         : [C, H, W] float32 — the full tile (on GPU)
    positions_px : [N, 2] float32 — (x, y) centres in pixel coordinates
    stamp_size   : side length of each stamp (must be odd)
    chunk_size   : process this many sources at a time (None = all at once).
                   Set to ~16384 to cap GPU memory at ~200 MB for 10 bands.

    Returns
    -------
    stamps : [N, C, S, S] float32
    """
    C, H, W = tile.shape
    S = stamp_size
    N = positions_px.shape[0]
    device = tile.device
    positions_px = positions_px.to(device)

    if chunk_size is None or N <= chunk_size:
        return _extract_stamps_chunk(tile, positions_px, S, C, H, W)

    out = torch.empty(N, C, S, S, dtype=torch.float32, device=device)
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        out[start:end] = _extract_stamps_chunk(
            tile, positions_px[start:end], S, C, H, W
        )
    return out


def _extract_stamps_chunk(
    tile: torch.Tensor,
    positions_px: torch.Tensor,
    S: int,
    C: int,
    H: int,
    W: int,
) -> torch.Tensor:
    n = positions_px.shape[0]
    device = tile.device

    # Sub-pixel offsets for a stamp centred at each source
    half = (S - 1) / 2.0
    offsets = torch.arange(S, dtype=torch.float32, device=device) - half  # [S]

    # Build sampling grid: for each source, S rows of S columns
    # x positions: pos_x + offset  → normalised to [-1, 1]
    # y positions: pos_y + row_offset → normalised to [-1, 1]
    xs = positions_px[:, 0]  # [n]
    ys = positions_px[:, 1]  # [n]

    # [n, S] each
    x_coords = xs.unsqueeze(-1) + offsets.unsqueeze(0)   # [n, S]
    y_coords = ys.unsqueeze(-1) + offsets.unsqueeze(0)   # [n, S]

    # Normalise to [-1, 1] (grid_sample convention)
    x_norm = (x_coords / (W - 1)) * 2.0 - 1.0  # [n, S]
    y_norm = (y_coords / (H - 1)) * 2.0 - 1.0  # [n, S]

    # Build grid of shape [1, n*S, S, 2]
    # For each source i, rows i*S..(i+1)*S sample y_coords[i] × x_coords all
    # x varies along the last dim (columns), y is fixed per row
    x_grid = x_norm.unsqueeze(-1).expand(n, S, S)         # [n, S, S]  y_row × x_col? no:
    # Careful: x varies over columns, y over rows within each stamp
    # x_norm[i] has S values — one per column; y_norm[i] has S values — one per row
    y_grid = y_norm.unsqueeze(-1).expand(n, S, S)          # [n, S_row, S_col] each row = same y
    x_grid = x_norm.unsqueeze(-2).expand(n, S, S)          # [n, S_row, S_col] each col = same x

    # Stack → [n, S, S, 2], reshape → [1, n*S, S, 2]
    grid = torch.stack([x_grid, y_grid], dim=-1)            # [n, S, S, 2]
    grid = grid.view(1, n * S, S, 2)

    # tile → [1, C, H, W]
    tile4d = tile.unsqueeze(0)

    # Sample: output [1, C, n*S, S]
    sampled = F.grid_sample(
        tile4d,
        grid,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True,
    )  # [1, C, n*S, S]

    # Reshape → [n, C, S, S]
    return sampled.squeeze(0).view(C, n, S, S).permute(1, 0, 2, 3)


def estimate_local_background(
    stamps: torch.Tensor,       # [N, C, S, S]
    inner_radius: float = 7.0,
    outer_radius: float = 9.5,
) -> torch.Tensor:
    """
    Estimate per-source per-band sky background from an annulus.

    Pixels in `inner_radius < r <= outer_radius` (measured from stamp centre)
    form the sky annulus.  Returns the median of those pixels per (source, band).

    Parameters
    ----------
    stamps       : [N, C, S, S]
    inner_radius : inner annulus radius in pixels
    outer_radius : outer annulus radius in pixels

    Returns
    -------
    bg : [N, C] float32
    """
    N, C, S, _ = stamps.shape
    device = stamps.device
    half = (S - 1) / 2.0

    y, x = torch.meshgrid(
        torch.arange(S, dtype=torch.float32, device=device),
        torch.arange(S, dtype=torch.float32, device=device),
        indexing='ij',
    )
    r = torch.sqrt((x - half) ** 2 + (y - half) ** 2)
    annulus_mask = (r > inner_radius) & (r <= outer_radius)  # [S, S]
    n_pix = annulus_mask.sum().item()

    if n_pix == 0:
        # Fall back: use outer 1px border
        border = torch.zeros(S, S, dtype=torch.bool, device=device)
        border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
        annulus_mask = border
        n_pix = annulus_mask.sum().item()

    # Extract annulus pixels: [N, C, n_pix]
    idx = annulus_mask.flatten().nonzero(as_tuple=True)[0]
    flat = stamps.view(N, C, S * S)
    sky_pix = flat[:, :, idx]   # [N, C, n_pix]

    # Median over annulus pixels
    bg, _ = sky_pix.median(dim=-1)  # [N, C]
    return bg
