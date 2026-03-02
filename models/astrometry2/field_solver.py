"""Explicit weighted control-grid solver for smooth astrometric fields."""

from typing import Dict, Tuple

import numpy as np


def _control_grid_nodes(vis_shape: Tuple[int, int], grid_shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    h, w = int(vis_shape[0]), int(vis_shape[1])
    gy, gx = int(grid_shape[0]), int(grid_shape[1])
    x_nodes = np.linspace(0.0, max(0.0, float(w - 1)), gx, dtype=np.float64)
    y_nodes = np.linspace(0.0, max(0.0, float(h - 1)), gy, dtype=np.float64)
    return x_nodes, y_nodes


def _basis_for_points(xy: np.ndarray, x_nodes: np.ndarray, y_nodes: np.ndarray) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float64)
    gx = x_nodes.size
    gy = y_nodes.size
    n = xy.shape[0]
    a = np.zeros((n, gx * gy), dtype=np.float64)

    for i, (x, y) in enumerate(xy):
        ix = int(np.searchsorted(x_nodes, x, side='right') - 1)
        iy = int(np.searchsorted(y_nodes, y, side='right') - 1)
        ix = int(np.clip(ix, 0, max(0, gx - 2)))
        iy = int(np.clip(iy, 0, max(0, gy - 2)))

        x0 = x_nodes[ix]
        x1 = x_nodes[min(ix + 1, gx - 1)]
        y0 = y_nodes[iy]
        y1 = y_nodes[min(iy + 1, gy - 1)]
        tx = 0.0 if x1 == x0 else float((x - x0) / (x1 - x0))
        ty = 0.0 if y1 == y0 else float((y - y0) / (y1 - y0))

        w00 = (1.0 - tx) * (1.0 - ty)
        w10 = tx * (1.0 - ty)
        w01 = (1.0 - tx) * ty
        w11 = tx * ty

        def idx(ixv: int, iyv: int) -> int:
            return iyv * gx + ixv

        a[i, idx(ix, iy)] += w00
        a[i, idx(min(ix + 1, gx - 1), iy)] += w10
        a[i, idx(ix, min(iy + 1, gy - 1))] += w01
        a[i, idx(min(ix + 1, gx - 1), min(iy + 1, gy - 1))] += w11
    return a


def _smoothness_rows(grid_shape: Tuple[int, int]) -> np.ndarray:
    gy, gx = int(grid_shape[0]), int(grid_shape[1])
    rows = []

    def idx(ix: int, iy: int) -> int:
        return iy * gx + ix

    for iy in range(gy):
        for ix in range(gx - 1):
            row = np.zeros((gx * gy,), dtype=np.float64)
            row[idx(ix, iy)] = -1.0
            row[idx(ix + 1, iy)] = 1.0
            rows.append(row)
    for iy in range(gy - 1):
        for ix in range(gx):
            row = np.zeros((gx * gy,), dtype=np.float64)
            row[idx(ix, iy)] = -1.0
            row[idx(ix, iy + 1)] = 1.0
            rows.append(row)
    if not rows:
        return np.zeros((0, gx * gy), dtype=np.float64)
    return np.stack(rows, axis=0)


def solve_control_grid_field(
    vis_xy: np.ndarray,
    offsets_arcsec: np.ndarray,
    weights: np.ndarray,
    vis_shape: Tuple[int, int],
    grid_shape: Tuple[int, int] = (12, 12),
    smooth_lambda: float = 1e-2,
) -> Dict[str, np.ndarray]:
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
    if d.size:
        reg = np.sqrt(max(0.0, float(smooth_lambda))) * d
        design = np.concatenate([aw, reg], axis=0)
    else:
        design = aw

    coeffs = []
    for k in range(2):
        yw = offsets_arcsec[:, k:k+1] * sqrt_w
        rhs = yw[:, 0]
        if d.size:
            rhs = np.concatenate([rhs, np.zeros((reg.shape[0],), dtype=np.float64)], axis=0)
        sol, _, _, _ = np.linalg.lstsq(design, rhs, rcond=None)
        coeffs.append(sol.reshape(grid_shape))

    return {
        'x_nodes': x_nodes.astype(np.float32),
        'y_nodes': y_nodes.astype(np.float32),
        'dra_nodes': coeffs[0].astype(np.float32),
        'ddec_nodes': coeffs[1].astype(np.float32),
        'grid_shape': np.asarray(grid_shape, dtype=np.int32),
    }


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
) -> Dict[str, np.ndarray]:
    h, w = int(vis_shape[0]), int(vis_shape[1])
    y_mesh = np.arange(0, h, int(max(1, dstep)), dtype=np.float64)
    x_mesh = np.arange(0, w, int(max(1, dstep)), dtype=np.float64)
    yy, xx = np.meshgrid(y_mesh, x_mesh, indexing='ij')
    xy = np.stack([xx.ravel(), yy.ravel()], axis=1)

    dra = _evaluate_nodes_at_points(field['x_nodes'], field['y_nodes'], field['dra_nodes'], xy)
    ddec = _evaluate_nodes_at_points(field['x_nodes'], field['y_nodes'], field['ddec_nodes'], xy)
    return {
        'x_mesh': x_mesh.astype(np.float32),
        'y_mesh': y_mesh.astype(np.float32),
        'dra': dra.reshape(y_mesh.size, x_mesh.size).astype(np.float32),
        'ddec': ddec.reshape(y_mesh.size, x_mesh.size).astype(np.float32),
    }
