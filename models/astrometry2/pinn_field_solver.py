"""
Physics-Informed Neural Network (PINN) concordance field solver.

Instead of generic smoothness regularisation (Tikhonov on a grid, or
weight-decay on an MLP), this solver encodes physical constraints on
the distortion field via automatic differentiation through the network:

  1. **Curl-free** — pure optical distortions are gradient fields (they
     derive from a scalar optical-path-length potential).  The residual
     WCS field should be approximately irrotational:
       ∂(dRA)/∂Dec − ∂(dDec)/∂RA ≈ 0

  2. **Laplacian smoothness** — the field varies slowly.  Instead of
     penalising finite-difference neighbours on a grid, we penalise
     the Laplacian of the field at collocation points sampled *anywhere*
     on the sky — no grid discretisation needed.

  3. **Band consistency** — Rubin-Euclid geometric distortions are
     achromatic.  Differential chromatic refraction (DCR) introduces a
     small band-dependent shift, but the bulk field is shared.  The
     PINN jointly fits all bands, separating a shared geometric field
     from per-band chromatic residuals.

Follows the same API as ``nn_field_solver.py`` so it can be used as a
drop-in replacement in ``infer_global_concordance.py``.

Reference: Raissi, Perdikaris & Karniadakis (2017), "Physics Informed
Deep Learning", arXiv:1711.10561.
"""

import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from scipy.spatial import cKDTree as KDTree


# ============================================================
# Network architecture
# ============================================================

class ConcordancePINN(nn.Module):
    """MLP that maps (x_norm, y_norm, band_embed) → (dRA*, dDec).

    Input coordinates require grad so that torch.autograd can compute
    spatial derivatives for the physics losses.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        n_layers: int = 5,
        n_bands: int = 9,
        band_embed_dim: int = 8,
    ):
        super().__init__()
        self.n_bands = n_bands
        self.band_embed_dim = band_embed_dim

        # Per-band embedding for chromatic correction.
        self.band_embed = nn.Embedding(n_bands, band_embed_dim)

        # Shared trunk: (x, y) → geometric field features.
        in_dim = 2  # normalised (x, y)
        layers = [nn.Linear(in_dim, hidden_dim), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        self.trunk = nn.Sequential(*layers)

        # Geometric head: shared across bands → (dRA, dDec).
        self.geo_head = nn.Linear(hidden_dim, 2)

        # Chromatic head: per-band residual from band embedding.
        self.chr_head = nn.Sequential(
            nn.Linear(hidden_dim + band_embed_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 2),
        )

        # Zero-init chromatic head so it starts at zero residual.
        nn.init.zeros_(self.chr_head[-1].weight)
        nn.init.zeros_(self.chr_head[-1].bias)

    def forward_geo(self, xy: torch.Tensor) -> torch.Tensor:
        """Geometric (achromatic) field only: [N, 2] → [N, 2]."""
        return self.geo_head(self.trunk(xy))

    def forward(
        self,
        xy: torch.Tensor,
        band_idx: torch.Tensor = None,
    ) -> torch.Tensor:
        """Full prediction: geometric + chromatic.

        Parameters
        ----------
        xy : [N, 2] normalised sky positions (requires_grad=True for PINN).
        band_idx : [N] integer band indices.  If None, returns geometric only.

        Returns
        -------
        pred : [N, 2] (dRA*, dDec) in normalised units.
        """
        feat = self.trunk(xy)
        geo = self.geo_head(feat)

        if band_idx is None:
            return geo

        embed = self.band_embed(band_idx)
        chr_in = torch.cat([feat, embed], dim=-1)
        chrom = self.chr_head(chr_in)
        return geo + chrom


# ============================================================
# Physics losses via automatic differentiation
# ============================================================

def _compute_physics_losses(
    model: ConcordancePINN,
    colloc_xy: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute curl and Laplacian losses at collocation points.

    Parameters
    ----------
    model : ConcordancePINN
    colloc_xy : [M, 2] collocation points with requires_grad=True

    Returns
    -------
    loss_curl : scalar — ||∂(dRA)/∂y − ∂(dDec)/∂x||²
    loss_lapl : scalar — ||∇²(dRA)||² + ||∇²(dDec)||²
    """
    colloc_xy = colloc_xy.requires_grad_(True)
    pred = model.forward_geo(colloc_xy)  # [M, 2]
    dra = pred[:, 0]   # dRA*
    ddec = pred[:, 1]  # dDec

    # First derivatives via autograd.
    grad_dra = torch.autograd.grad(
        dra, colloc_xy, grad_outputs=torch.ones_like(dra),
        create_graph=True,
    )[0]  # [M, 2]
    grad_ddec = torch.autograd.grad(
        ddec, colloc_xy, grad_outputs=torch.ones_like(ddec),
        create_graph=True,
    )[0]  # [M, 2]

    dra_dx = grad_dra[:, 0]    # ∂(dRA)/∂x
    dra_dy = grad_dra[:, 1]    # ∂(dRA)/∂y
    ddec_dx = grad_ddec[:, 0]  # ∂(dDec)/∂x
    ddec_dy = grad_ddec[:, 1]  # ∂(dDec)/∂y

    # Curl: ∂(dRA)/∂y − ∂(dDec)/∂x ≈ 0 for irrotational field.
    curl = dra_dy - ddec_dx
    loss_curl = (curl ** 2).mean()

    # Second derivatives for Laplacian.
    dra_dxx = torch.autograd.grad(
        dra_dx, colloc_xy, grad_outputs=torch.ones_like(dra_dx),
        create_graph=True,
    )[0][:, 0]  # ∂²(dRA)/∂x²
    dra_dyy = torch.autograd.grad(
        dra_dy, colloc_xy, grad_outputs=torch.ones_like(dra_dy),
        create_graph=True,
    )[0][:, 1]  # ∂²(dRA)/∂y²
    ddec_dxx = torch.autograd.grad(
        ddec_dx, colloc_xy, grad_outputs=torch.ones_like(ddec_dx),
        create_graph=True,
    )[0][:, 0]  # ∂²(dDec)/∂x²
    ddec_dyy = torch.autograd.grad(
        ddec_dy, colloc_xy, grad_outputs=torch.ones_like(ddec_dy),
        create_graph=True,
    )[0][:, 1]  # ∂²(dDec)/∂y²

    lapl_dra = dra_dxx + dra_dyy
    lapl_ddec = ddec_dxx + ddec_dyy
    loss_lapl = (lapl_dra ** 2 + lapl_ddec ** 2).mean()

    return loss_curl, loss_lapl


# ============================================================
# Band consistency loss
# ============================================================

def _band_consistency_loss(
    model: ConcordancePINN,
    xy: torch.Tensor,
    band_idx: torch.Tensor,
) -> torch.Tensor:
    """Penalise variance of chromatic residuals across bands.

    For each spatial position, the chromatic residual should be small
    relative to the geometric field.  This encourages the model to put
    as much as possible into the shared geometric component.
    """
    feat = model.trunk(xy)
    # Evaluate chromatic residual for all bands at these positions.
    n = xy.shape[0]
    unique_bands = band_idx.unique()
    if unique_bands.numel() <= 1:
        return torch.tensor(0.0, device=xy.device)

    chr_preds = []
    for b in unique_bands:
        mask = band_idx == b
        if mask.sum() == 0:
            continue
        embed = model.band_embed(torch.full((mask.sum(),), b, dtype=torch.long, device=xy.device))
        chr_in = torch.cat([feat[mask], embed], dim=-1)
        chr_preds.append(model.chr_head(chr_in).mean(dim=0))  # [2]

    if len(chr_preds) < 2:
        return torch.tensor(0.0, device=xy.device)

    stacked = torch.stack(chr_preds, dim=0)  # [n_bands, 2]
    return stacked.var(dim=0).sum()


# ============================================================
# Main fitting function
# ============================================================

def fit_pinn_field(
    pos_arcsec: np.ndarray,
    offsets_arcsec: np.ndarray,
    weights: np.ndarray,
    band_indices: np.ndarray = None,
    *,
    hidden_dim: int = 128,
    n_layers: int = 5,
    n_bands: int = 9,
    band_embed_dim: int = 8,
    n_steps: int = 5000,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    lambda_curl: float = 1.0,
    lambda_lapl: float = 0.1,
    lambda_band: float = 0.1,
    n_collocation: int = 10000,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> Tuple[ConcordancePINN, dict]:
    """Fit a physics-informed concordance field.

    Parameters
    ----------
    pos_arcsec : [N, 2] tangent-plane positions in arcsec
    offsets_arcsec : [N, 2] (dRA*, dDec) measured offsets in arcsec
    weights : [N] per-source weights (1/σ²)
    band_indices : [N] integer band index per source (0..n_bands-1).
        If None, all sources are treated as one band (no chromatic split).
    hidden_dim : MLP hidden width
    n_layers : MLP depth
    n_bands : total number of bands (for embedding table)
    band_embed_dim : band embedding dimension
    n_steps : Adam optimisation steps
    lr : initial learning rate
    weight_decay : L2 on network weights (additional to physics losses)
    lambda_curl : weight for curl-free physics loss
    lambda_lapl : weight for Laplacian smoothness loss
    lambda_band : weight for band consistency loss
    n_collocation : number of collocation points for physics losses
    device : torch device
    verbose : print progress

    Returns
    -------
    model : ConcordancePINN (CPU, eval mode)
    meta : dict with normalisation constants and training stats
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    N = pos_arcsec.shape[0]
    has_bands = band_indices is not None

    # ── Normalisation ───────────────────────────────────────────────────
    pos_min = pos_arcsec.min(axis=0).astype(np.float32)
    pos_max = pos_arcsec.max(axis=0).astype(np.float32)
    pos_scale = np.maximum(pos_max - pos_min, 1e-6)
    pos_norm = (pos_arcsec.astype(np.float32) - pos_min) / pos_scale * 2.0 - 1.0

    off_scale = float(np.percentile(np.abs(offsets_arcsec), 95)) or 1.0
    off_norm = (offsets_arcsec / off_scale).astype(np.float32)

    # ── Tensors ─────────────────────────────────────────────────────────
    xy = torch.tensor(pos_norm, dtype=torch.float32, device=device)
    tgt = torch.tensor(off_norm, dtype=torch.float32, device=device)
    w = torch.tensor(
        (weights / (weights.mean() + 1e-10)).astype(np.float32),
        device=device,
    )
    if has_bands:
        band_t = torch.tensor(band_indices.astype(np.int64), device=device)
    else:
        band_t = None

    # ── Model ───────────────────────────────────────────────────────────
    model = ConcordancePINN(
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        n_bands=n_bands,
        band_embed_dim=band_embed_dim,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps, eta_min=1e-6)

    # ── Collocation point range (slightly beyond data extent) ───────────
    margin = 0.1  # 10% beyond data range in normalised coords
    colloc_lo = -1.0 - margin
    colloc_hi = 1.0 + margin

    best_loss = float('inf')
    best_state = None
    loss_history = []

    for step in range(1, n_steps + 1):
        optimizer.zero_grad()

        # Data loss: weighted MSE on measured offsets.
        pred = model(xy, band_idx=band_t)
        residual = pred - tgt
        loss_data = (w.unsqueeze(-1) * residual ** 2).mean()

        # Physics losses at random collocation points.
        colloc = torch.rand(n_collocation, 2, device=device) * (colloc_hi - colloc_lo) + colloc_lo
        loss_curl, loss_lapl = _compute_physics_losses(model, colloc)

        # Band consistency loss.
        loss_band = torch.tensor(0.0, device=device)
        if has_bands and lambda_band > 0:
            loss_band = _band_consistency_loss(model, xy, band_t)

        # Total PINN loss.
        loss = (
            loss_data
            + lambda_curl * loss_curl
            + lambda_lapl * loss_lapl
            + lambda_band * loss_band
        )

        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_val = float(loss.item())
        loss_history.append(loss_val)

        if loss_val < best_loss:
            best_loss = loss_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if verbose and (step % 500 == 0 or step == 1):
            print(
                f'  PINN step {step:5d}/{n_steps} | '
                f'loss={loss_val:.6f} '
                f'data={loss_data.item():.6f} '
                f'curl={loss_curl.item():.6f} '
                f'lapl={loss_lapl.item():.6f} '
                f'band={loss_band.item():.6f}'
            )

    # Restore best model.
    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.cpu().eval()

    meta = {
        'pos_min': pos_min,
        'pos_max': pos_max,
        'pos_scale': pos_scale,
        'off_scale': off_scale,
        'n_steps': n_steps,
        'best_loss': best_loss,
        'loss_history': np.array(loss_history, dtype=np.float32),
        'lambda_curl': lambda_curl,
        'lambda_lapl': lambda_lapl,
        'lambda_band': lambda_band,
        'n_collocation': n_collocation,
    }

    if verbose:
        pred_final = model(xy.cpu(), band_idx=band_t.cpu() if band_t is not None else None).detach().numpy()
        pred_arcsec = pred_final * off_scale
        resid = offsets_arcsec - pred_arcsec
        resid_mag = np.hypot(resid[:, 0], resid[:, 1]) * 1000.0
        print(f'  PINN fit: median residual = {np.median(resid_mag):.1f} mas, '
              f'p68 = {np.percentile(resid_mag, 68):.1f} mas')

    return model, meta


# ============================================================
# Mesh evaluation (same interface as nn_field_solver)
# ============================================================

def evaluate_pinn_mesh(
    model: ConcordancePINN,
    meta: dict,
    field_h: int,
    field_w: int,
    dstep: int = 1,
    pos_arcsec_anchors: Optional[np.ndarray] = None,
    band_idx: Optional[int] = None,
    batch_size: int = 65536,
) -> dict:
    """Evaluate the PINN on a regular grid.

    Returns the same dict format as other solvers:
      {'dra': [H, W], 'ddec': [H, W], 'coverage': [H, W] or None,
       'x_mesh': [W], 'y_mesh': [H]}

    Parameters
    ----------
    band_idx : if not None, evaluate for this specific band (geometric + chromatic).
               If None, evaluate geometric field only.
    """
    pos_min = meta['pos_min']
    pos_scale = meta['pos_scale']
    off_scale = meta['off_scale']

    x_mesh = np.arange(0, field_w, dstep, dtype=np.float32)
    y_mesh = np.arange(0, field_h, dstep, dtype=np.float32)
    xx, yy = np.meshgrid(x_mesh, y_mesh)
    grid_pos = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)

    grid_norm = (grid_pos - pos_min) / pos_scale * 2.0 - 1.0

    preds = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(grid_norm), batch_size):
            chunk = torch.tensor(grid_norm[i:i + batch_size], dtype=torch.float32)
            if band_idx is not None:
                b = torch.full((chunk.shape[0],), band_idx, dtype=torch.long)
                preds.append(model(chunk, band_idx=b).numpy())
            else:
                preds.append(model.forward_geo(chunk).numpy())
    pred_np = np.concatenate(preds, axis=0) * off_scale

    mesh_shape = (len(y_mesh), len(x_mesh))
    dra = pred_np[:, 0].reshape(mesh_shape).astype(np.float32)
    ddec = pred_np[:, 1].reshape(mesh_shape).astype(np.float32)

    coverage = None
    if pos_arcsec_anchors is not None:
        tree = KDTree(pos_arcsec_anchors)
        dists, _ = tree.query(grid_pos)
        coverage = dists.reshape(mesh_shape).astype(np.float32)

    return {
        'dra': dra,
        'ddec': ddec,
        'coverage': coverage,
        'x_mesh': x_mesh,
        'y_mesh': y_mesh,
    }
