"""Explicit weighted control-grid solver for smooth astrometric fields.

Changes from v1:
  - _basis_for_points is fully vectorized (no Python loop over anchors).
  - Adaptive per-node anchor regularization: nodes far from any source get a
    stronger ridge pull toward zero, preventing edge/corner drift in sparse
    regions. Controlled by anchor_lambda (base strength) and
    anchor_radius_px (locality scale).
  - evaluate_control_grid_mesh returns a 'coverage' map: the minimum distance
    (in VIS pixels) from each mesh point to the nearest anchor source. This
    lets downstream code know which parts of the concordance field are
    data-driven vs. pure regularization.
  - auto_grid_shape picks grid resolution based on anchor count, avoiding
    the underdetermined regime when sources are sparse.
"""

from typing import Dict, Optional, Tuple

import numpy as np

try:
    import torch as _torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


# ---------------------------------------------------------------------------
# Grid node coordinates
# ---------------------------------------------------------------------------

def _control_grid_nodes(vis_shape: Tuple[int, int], grid_shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    h, w = int(vis_shape[0]), int(vis_shape[1])
    gy, gx = int(grid_shape[0]), int(grid_shape[1])
    x_nodes = np.linspace(0.0, max(0.0, float(w - 1)), gx, dtype=np.float64)
    y_nodes = np.linspace(0.0, max(0.0, float(h - 1)), gy, dtype=np.float64)
    return x_nodes, y_nodes


# ---------------------------------------------------------------------------
# Bilinear basis matrix (vectorized)
# ---------------------------------------------------------------------------

def _basis_for_points(xy: np.ndarray, x_nodes: np.ndarray, y_nodes: np.ndarray) -> np.ndarray:
    """Build the [N, gx*gy] bilinear interpolation basis matrix.

    Fully vectorized -- no Python loop over points.
    """
    xy = np.asarray(xy, dtype=np.float64)
    x_nodes = np.asarray(x_nodes, dtype=np.float64)
    y_nodes = np.asarray(y_nodes, dtype=np.float64)
    gx = x_nodes.size
    gy = y_nodes.size
    n = xy.shape[0]

    xs = xy[:, 0]
    ys = xy[:, 1]

    # Find the left/bottom cell index for each point.
    ix = np.searchsorted(x_nodes, xs, side='right').astype(np.int64) - 1
    iy = np.searchsorted(y_nodes, ys, side='right').astype(np.int64) - 1
    ix = np.clip(ix, 0, max(0, gx - 2))
    iy = np.clip(iy, 0, max(0, gy - 2))

    # Clipped upper neighbor indices.
    ix1 = np.minimum(ix + 1, gx - 1)
    iy1 = np.minimum(iy + 1, gy - 1)

    # Fractional position within cell.
    x0 = x_nodes[ix]
    x1 = x_nodes[ix1]
    y0 = y_nodes[iy]
    y1 = y_nodes[iy1]

    dx = x1 - x0
    dy = y1 - y0
    # Avoid division by zero for degenerate 1-node grids.
    dx = np.where(dx == 0.0, 1.0, dx)
    dy = np.where(dy == 0.0, 1.0, dy)

    tx = (xs - x0) / dx
    ty = (ys - y0) / dy

    w00 = (1.0 - tx) * (1.0 - ty)
    w10 = tx * (1.0 - ty)
    w01 = (1.0 - tx) * ty
    w11 = tx * ty

    # Flat node indices for the four corners.
    idx00 = iy * gx + ix
    idx10 = iy * gx + ix1
    idx01 = iy1 * gx + ix
    idx11 = iy1 * gx + ix1

    # Build basis matrix via scatter-add.
    row = np.arange(n)
    a = np.zeros((n, gx * gy), dtype=np.float64)
    np.add.at(a, (row, idx00), w00)
    np.add.at(a, (row, idx10), w10)
    np.add.at(a, (row, idx01), w01)
    np.add.at(a, (row, idx11), w11)
    return a


# ---------------------------------------------------------------------------
# Smoothness (finite-difference) regularization rows
# ---------------------------------------------------------------------------

def _smoothness_rows(grid_shape: Tuple[int, int]) -> np.ndarray:
    gy, gx = int(grid_shape[0]), int(grid_shape[1])
    n_nodes = gx * gy

    n_horiz = gy * max(0, gx - 1)
    n_vert = max(0, gy - 1) * gx
    total = n_horiz + n_vert
    if total == 0:
        return np.zeros((0, n_nodes), dtype=np.float64)

    d = np.zeros((total, n_nodes), dtype=np.float64)
    row = 0
    for iy in range(gy):
        for ix in range(gx - 1):
            d[row, iy * gx + ix] = -1.0
            d[row, iy * gx + ix + 1] = 1.0
            row += 1
    for iy in range(gy - 1):
        for ix in range(gx):
            d[row, iy * gx + ix] = -1.0
            d[row, (iy + 1) * gx + ix] = 1.0
            row += 1
    return d


# ---------------------------------------------------------------------------
# Adaptive per-node anchor weights
# ---------------------------------------------------------------------------

def _adaptive_anchor_weights(
    x_nodes: np.ndarray,
    y_nodes: np.ndarray,
    vis_xy: np.ndarray,
    weights: np.ndarray,
    anchor_lambda: float,
    anchor_radius_px: float,
) -> np.ndarray:
    """Compute per-node ridge regularization strength.

    Nodes close to many high-weight sources get the base anchor_lambda.
    Nodes far from any source get a much stronger pull toward zero.

    The effective per-node lambda is:

        lambda_i = anchor_lambda * max(1, base_support / local_support_i)

    where local_support_i = sum of Gaussian-weighted data weights near node i.
    """
    gx = x_nodes.size
    gy = y_nodes.size
    n_nodes = gx * gy

    # Node positions on a flat grid.
    node_yy, node_xx = np.meshgrid(y_nodes, x_nodes, indexing='ij')
    node_xy = np.stack([node_xx.ravel(), node_yy.ravel()], axis=1)  # [n_nodes, 2]

    vis_xy = np.asarray(vis_xy, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64).ravel()

    if vis_xy.shape[0] == 0 or anchor_radius_px <= 0:
        return np.full(n_nodes, np.sqrt(max(0.0, anchor_lambda * 10.0)), dtype=np.float64)

    sigma = max(1.0, float(anchor_radius_px))
    # dists[i, j] = squared distance from node i to source j
    ddx = node_xy[:, 0:1] - vis_xy[:, 0:1].T  # [n_nodes, n_sources]
    ddy = node_xy[:, 1:2] - vis_xy[:, 1:2].T
    dist_sq = ddx ** 2 + ddy ** 2
    kern = np.exp(-0.5 * dist_sq / (sigma ** 2))
    # Weighted support at each node.
    local_support = (kern * weights[None, :]).sum(axis=1)  # [n_nodes]

    # Use median support as the reference "well-constrained" level.
    base_support = max(float(np.median(local_support)), 1e-10)
    ratio = np.clip(base_support / np.maximum(local_support, 1e-10), 1.0, 100.0)

    # Return sqrt(per-node lambda) since the solver uses it as a row weight.
    return np.sqrt(anchor_lambda * ratio)


# ---------------------------------------------------------------------------
# Auto grid shape selection
# ---------------------------------------------------------------------------

def auto_grid_shape(
    n_anchors: int,
    default: Tuple[int, int] = (12, 12),
    min_shape: Tuple[int, int] = (4, 4),
) -> Tuple[int, int]:
    """Choose grid resolution so #nodes <= n_anchors / 2.

    This avoids the heavily underdetermined regime where the regularizer
    does most of the work.  With 50 sources, a 12x12 grid (144 nodes)
    means ~0.35 sources per node per axis.  This function drops to 6x6
    or 4x4 as needed.

    Returns default if there are enough sources.
    """
    target_nodes = max(int(min_shape[0]) * int(min_shape[1]),
                       n_anchors // 2)
    gy, gx = int(default[0]), int(default[1])
    while gy * gx > target_nodes and (gy > min_shape[0] or gx > min_shape[1]):
        if gy >= gx and gy > min_shape[0]:
            gy -= 1
        elif gx > min_shape[1]:
            gx -= 1
        else:
            break
    return (max(gy, int(min_shape[0])), max(gx, int(min_shape[1])))


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------

def solve_control_grid_field(
    vis_xy: np.ndarray,
    offsets_arcsec: np.ndarray,
    weights: np.ndarray,
    vis_shape: Tuple[int, int],
    grid_shape: Tuple[int, int] = (12, 12),
    smooth_lambda: float = 1e-2,
    anchor_lambda: float = 1e-3,
    anchor_radius_px: float = 0.0,
) -> Dict[str, np.ndarray]:
    """Solve for control-grid coefficients via weighted least-squares.

    Parameters
    ----------
    vis_xy : (N, 2) source positions in VIS pixel coordinates.
    offsets_arcsec : (N, 2) predicted (DRA*, DDec) offsets in arcsec.
    weights : (N,) per-source weights (typically 1/sigma^2).
    vis_shape : (H, W) of the VIS image.
    grid_shape : (gy, gx) control grid resolution.
    smooth_lambda : Tikhonov smoothness weight on finite differences.
    anchor_lambda : Ridge regularization base strength.
        Default raised from 1e-4 to 1e-3.  Nodes far from sources get
        adaptively stronger regularization when anchor_radius_px > 0.
    anchor_radius_px : Gaussian scale (VIS pixels) for adaptive anchor.
        If > 0, nodes far from any source get stronger ridge pull.
        Recommended: ~2x the mean grid cell spacing.
        If 0 (default), falls back to uniform ridge (backward compatible).

    Returns
    -------
    dict with x_nodes, y_nodes, dra_nodes, ddec_nodes, grid_shape.
    """
    vis_xy = np.asarray(vis_xy, dtype=np.float64)
    offsets_arcsec = np.asarray(offsets_arcsec, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if vis_xy.shape[0] < 4:
        raise ValueError('Need at least 4 anchors to solve a control grid field.')

    x_nodes, y_nodes = _control_grid_nodes(vis_shape, grid_shape)
    a = _basis_for_points(vis_xy, x_nodes, y_nodes)
    sqrt_w = np.sqrt(np.clip(weights, 1e-6, None))[:, None]
    aw = a * sqrt_w
    d = _smoothness_rows(grid_shape)

    # Anchor regularization -- adaptive or uniform.
    n_nodes = int(grid_shape[0]) * int(grid_shape[1])
    if anchor_radius_px > 0:
        per_node_sqrt_lam = _adaptive_anchor_weights(
            x_nodes, y_nodes, vis_xy, weights, anchor_lambda, anchor_radius_px)
        anchor = np.diag(per_node_sqrt_lam)
    else:
        anchor = np.sqrt(max(0.0, float(anchor_lambda))) * np.eye(n_nodes, dtype=np.float64)

    parts = [aw]
    if d.size:
        parts.append(np.sqrt(max(0.0, float(smooth_lambda))) * d)
    parts.append(anchor)
    design = np.concatenate(parts, axis=0)

    # Build RHS for both axes at once: [M, 2]
    rhs_data = offsets_arcsec * sqrt_w  # [N, 2]
    pad_smooth = np.zeros((d.shape[0], 2), dtype=np.float64) if d.size else np.zeros((0, 2), dtype=np.float64)
    pad_anchor = np.zeros((n_nodes, 2), dtype=np.float64)
    rhs = np.concatenate([rhs_data, pad_smooth, pad_anchor], axis=0)  # [M, 2]

    # Solve via GPU (fast) or CPU fallback.
    use_gpu = _HAS_TORCH and _torch.cuda.is_available() and design.shape[0] > 10000
    if use_gpu:
        design_t = _torch.from_numpy(design).float().cuda()
        rhs_t = _torch.from_numpy(rhs).float().cuda()
        sol_t = _torch.linalg.lstsq(design_t, rhs_t).solution
        sol = sol_t.cpu().numpy().astype(np.float64)
        del design_t, rhs_t, sol_t
        _torch.cuda.empty_cache()
    else:
        sol, _, _, _ = np.linalg.lstsq(design, rhs, rcond=None)

    coeffs = [sol[:, k].reshape(grid_shape) for k in range(2)]

    return {
        'x_nodes': x_nodes.astype(np.float32),
        'y_nodes': y_nodes.astype(np.float32),
        'dra_nodes': coeffs[0].astype(np.float32),
        'ddec_nodes': coeffs[1].astype(np.float32),
        'grid_shape': np.asarray(grid_shape, dtype=np.int32),
    }


# ---------------------------------------------------------------------------
# Mesh evaluation
# ---------------------------------------------------------------------------

def _evaluate_nodes_at_points(
    x_nodes: np.ndarray,
    y_nodes: np.ndarray,
    values: np.ndarray,
    xy: np.ndarray,
) -> np.ndarray:
    a = _basis_for_points(xy, np.asarray(x_nodes, dtype=np.float64), np.asarray(y_nodes, dtype=np.float64))
    vec = np.asarray(values, dtype=np.float64).reshape(-1)
    return (a @ vec).astype(np.float32)


def evaluate_control_grid_mesh(
    field: Dict[str, np.ndarray],
    vis_shape: Tuple[int, int],
    dstep: int = 8,
    anchor_xy: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Evaluate the solved field on a regular mesh.

    Parameters
    ----------
    field : output of solve_control_grid_field.
    vis_shape : (H, W) of the VIS image.
    dstep : sampling step in VIS pixels.
    anchor_xy : (N, 2) source positions used in the solve.
        If provided, returns a 'coverage' map = min distance from each
        mesh point to the nearest anchor, in VIS pixels.

    Returns
    -------
    dict with x_mesh, y_mesh, dra, ddec, and optionally 'coverage'.
    """
    h, w = int(vis_shape[0]), int(vis_shape[1])
    y_mesh = np.arange(0, h, int(max(1, dstep)), dtype=np.float64)
    x_mesh = np.arange(0, w, int(max(1, dstep)), dtype=np.float64)
    yy, xx = np.meshgrid(y_mesh, x_mesh, indexing='ij')
    xy = np.stack([xx.ravel(), yy.ravel()], axis=1)

    dra = _evaluate_nodes_at_points(field['x_nodes'], field['y_nodes'], field['dra_nodes'], xy)
    ddec = _evaluate_nodes_at_points(field['x_nodes'], field['y_nodes'], field['ddec_nodes'], xy)
    out = {
        'x_mesh': x_mesh.astype(np.float32),
        'y_mesh': y_mesh.astype(np.float32),
        'dra': dra.reshape(y_mesh.size, x_mesh.size).astype(np.float32),
        'ddec': ddec.reshape(y_mesh.size, x_mesh.size).astype(np.float32),
    }

    if anchor_xy is not None and len(anchor_xy) > 0:
        anchor_xy = np.asarray(anchor_xy, dtype=np.float64)
        n_mesh = xy.shape[0]
        min_dist = np.full(n_mesh, np.inf, dtype=np.float64)
        chunk = 4096
        for i in range(0, n_mesh, chunk):
            mesh_chunk = xy[i:i + chunk]  # [C, 2]
            d2 = ((mesh_chunk[:, None, :] - anchor_xy[None, :, :]) ** 2).sum(axis=2)
            min_dist[i:i + chunk] = np.sqrt(d2.min(axis=1))
        out['coverage'] = min_dist.reshape(y_mesh.size, x_mesh.size).astype(np.float32)

    return out
