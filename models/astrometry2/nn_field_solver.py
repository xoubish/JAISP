"""Neural-network based global concordance field solver.

Fits a small MLP  (x_arcsec, y_arcsec) → (ΔRA*, ΔDec)  directly to the
per-source predictions instead of using the control-grid least-squares solve.

Advantages over the control grid
---------------------------------
- No grid resolution hyperparameter — smoothness comes from the architecture
  and weight-decay regularisation, not from the grid cell count.
- SiLU activations produce infinitely differentiable interpolants; no
  aliasing artefacts at tile boundaries.
- Scales continuously: the same model can represent a 50-source field or
  a 50 000-source field without changing any parameter.

The mesh output format is identical to evaluate_control_grid_mesh so all
downstream code (FITS writing, plotting, GlobalConcordanceMap) is unchanged.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.spatial import KDTree


# ──────────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────────

class _ResBlock(nn.Module):
    """Pre-activation residual block: SiLU → Linear → SiLU → Linear + skip."""

    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim),
            nn.SiLU(), nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class DistortionMLP(nn.Module):
    """
    Compact MLP: normalised (x, y) sky position → (ΔRA*, ΔDec) in arcsec.

    Architecture
    ------------
    - Input:  2-d normalised tangent-plane position (both in [-1, 1])
    - Hidden: n_layers × hidden_dim neurons, SiLU activation.
              When n_layers >= 4, uses residual blocks for stable
              gradient flow through deeper networks.
    - Geometric head: hidden → 2-d offset (shared across bands)
    - Chromatic head (optional, when n_bands > 0):
          (hidden, band_embed) → 2-d correction (per-band residual)

    The chromatic head is initialised to zero so the model starts as a
    pure geometric field and gradually learns band-dependent corrections.
    """

    def __init__(self, hidden_dim: int = 64, n_layers: int = 4,
                 n_bands: int = 0, band_embed_dim: int = 8):
        super().__init__()
        # Stem: project 2-d input to hidden_dim
        self.stem = nn.Sequential(nn.Linear(2, hidden_dim), nn.SiLU())

        # Hidden layers: use residual blocks for >= 4 layers
        if n_layers >= 4:
            # Each ResBlock counts as 2 layers; fill remainder with plain
            n_res = (n_layers - 1) // 2
            n_plain = (n_layers - 1) - n_res * 2
            blocks: list[nn.Module] = [_ResBlock(hidden_dim) for _ in range(n_res)]
            for _ in range(n_plain):
                blocks += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
            self.hidden = nn.Sequential(*blocks)
        else:
            layers: list[nn.Module] = []
            for _ in range(n_layers - 1):
                layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
            self.hidden = nn.Sequential(*layers)

        self.geo_head = nn.Linear(hidden_dim, 2)

        # Chromatic path (optional)
        self.n_bands = n_bands
        if n_bands > 0:
            self.band_embed = nn.Embedding(n_bands, band_embed_dim)
            self.chrom_head = nn.Sequential(
                nn.Linear(hidden_dim + band_embed_dim, hidden_dim // 2),
                nn.SiLU(),
                nn.Linear(hidden_dim // 2, 2),
            )
            # Zero-init so chromatic correction starts at zero
            nn.init.zeros_(self.chrom_head[-1].weight)
            nn.init.zeros_(self.chrom_head[-1].bias)

    def forward(self, xy: torch.Tensor,
                band_idx: torch.Tensor = None) -> torch.Tensor:
        h = self.hidden(self.stem(xy))
        out = self.geo_head(h)
        if self.n_bands > 0 and band_idx is not None:
            be = self.band_embed(band_idx)
            out = out + self.chrom_head(torch.cat([h, be], dim=-1))
        return out


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def fit_nn_field(
    pos_arcsec: np.ndarray,
    offsets_arcsec: np.ndarray,
    weights: np.ndarray,
    band_indices: np.ndarray = None,
    hidden_dim: int = 64,
    n_layers: int = 4,
    n_bands: int = 0,
    n_steps: int = 2000,
    lr: float = 3e-3,
    weight_decay: float = 1e-5,
    huber_delta: float = 0.0,
    grad_clip: float = 0.0,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> Tuple[DistortionMLP, dict]:
    """
    Fit a smooth MLP to per-source astrometric offset predictions.

    Parameters
    ----------
    pos_arcsec      : [N, 2] tangent-plane positions in arcsec (x, y)
    offsets_arcsec  : [N, 2] (ΔRA*, ΔDec) targets in arcsec
    weights         : [N]    per-source weights (typically 1/σ²)
    band_indices    : [N]    int band index per source (optional).
                     If provided together with n_bands > 0, the MLP learns
                     a shared geometric field plus per-band chromatic corrections.
    hidden_dim      : neurons per hidden layer
    n_layers        : number of hidden layers
    n_bands         : number of bands (0 = band-agnostic, >0 = chromatic)
    n_steps         : gradient-descent steps (Adam + cosine LR schedule)
    lr              : initial Adam learning rate
    weight_decay    : L2 regularisation — higher = smoother field
    huber_delta     : if > 0, use Huber loss with this delta (in normalised
                      offset units) instead of MSE.  More robust to outlier
                      centroids.  Typical value: 1.0–2.0 (≈ 1–2× off_scale).
                      0 = plain MSE (original behaviour).
    grad_clip       : max gradient norm for clipping (0 = no clipping)
    device          : torch device (auto-detected if None)
    verbose         : print loss every 500 steps

    Returns
    -------
    model : DistortionMLP  (CPU, eval mode)
    meta  : dict — normalisation constants + training stats
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── Input normalisation: scale to [-1, 1] ────────────────────────────────
    pos_min   = pos_arcsec.min(axis=0).astype(np.float32)
    pos_max   = pos_arcsec.max(axis=0).astype(np.float32)
    pos_scale = np.maximum(pos_max - pos_min, 1e-6)
    pos_norm  = (pos_arcsec.astype(np.float32) - pos_min) / pos_scale * 2.0 - 1.0

    # ── Target normalisation: unit scale helps the LR ────────────────────────
    off_scale = float(np.percentile(np.abs(offsets_arcsec), 95)) or 1.0
    off_norm  = (offsets_arcsec / off_scale).astype(np.float32)

    # ── Tensors ──────────────────────────────────────────────────────────────
    xy  = torch.tensor(pos_norm, dtype=torch.float32, device=device)
    tgt = torch.tensor(off_norm, dtype=torch.float32, device=device)
    w   = torch.tensor(
        (weights / weights.mean()).astype(np.float32),
        device=device,
    )
    bi_t = None
    if band_indices is not None and n_bands > 0:
        bi_t = torch.tensor(band_indices, dtype=torch.long, device=device)

    # ── Loss function ────────────────────────────────────────────────────────
    use_huber = huber_delta > 0
    if use_huber:
        huber_fn = nn.HuberLoss(reduction='none', delta=huber_delta)

    # ── Model + optimiser ────────────────────────────────────────────────────
    model = DistortionMLP(
        hidden_dim=hidden_dim, n_layers=n_layers,
        n_bands=n_bands, band_embed_dim=8,
    ).to(device)
    opt   = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=n_steps, eta_min=lr * 0.01,
    )

    # ── Training loop ────────────────────────────────────────────────────────
    model.train()
    losses = []
    best_loss = float('inf')
    best_state = None

    for step in range(n_steps):
        opt.zero_grad()
        pred_out = model(xy, band_idx=bi_t)
        if use_huber:
            loss = (w[:, None] * huber_fn(pred_out, tgt)).mean()
        else:
            loss = (w[:, None] * (pred_out - tgt) ** 2).mean()
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        sched.step()

        loss_val = float(loss)
        losses.append(loss_val)
        if loss_val < best_loss:
            best_loss = loss_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if verbose and (step + 1) % 500 == 0:
            print(f'    NN step {step+1:4d}/{n_steps}  loss={loss_val:.5f}')

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval().cpu()

    meta = {
        'pos_min':      pos_min,
        'pos_scale':    pos_scale,
        'off_scale':    off_scale,
        'n_sources':    int(len(pos_arcsec)),
        'final_loss':   float(losses[-1]),
        'best_loss':    best_loss,
        'loss_history': np.array(losses, dtype=np.float32),
        'hidden_dim':   hidden_dim,
        'n_layers':     n_layers,
        'n_steps':      n_steps,
        'weight_decay': weight_decay,
    }
    if verbose:
        print(f'  NN converged: best loss={best_loss:.5f}  '
              f'({hidden_dim}×{n_layers}, wd={weight_decay})')
    return model, meta


# ──────────────────────────────────────────────────────────────────────────────
# Mesh evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_nn_mesh(
    model: DistortionMLP,
    meta: dict,
    field_h: int,
    field_w: int,
    dstep: int = 1,
    pos_arcsec_anchors: Optional[np.ndarray] = None,
    band_idx: int = None,
    batch_size: int = 65536,
) -> dict:
    """
    Evaluate the trained MLP on a regular grid.

    Returns the same dict format as ``evaluate_control_grid_mesh``:
      {'dra': [H_mesh, W_mesh], 'ddec': [H_mesh, W_mesh],
       'coverage': [H_mesh, W_mesh] or None,
       'x_mesh': [W_mesh], 'y_mesh': [H_mesh]}

    Parameters
    ----------
    field_h, field_w        : field dimensions in arcsec (1 arcsec = 1 px here)
    dstep                   : mesh step in arcsec (same as dstep_arcsec)
    pos_arcsec_anchors      : [N, 2] training positions — used to build the
                              coverage map (min dist to nearest anchor source).
                              If None, coverage is not computed.
    batch_size              : GPU batch size for large grids
    """
    pos_min   = meta['pos_min']
    pos_scale = meta['pos_scale']
    off_scale = meta['off_scale']

    # Regular arcsec grid (same coordinate system as training)
    x_mesh = np.arange(0, field_w, dstep, dtype=np.float32)
    y_mesh = np.arange(0, field_h, dstep, dtype=np.float32)
    xx, yy = np.meshgrid(x_mesh, y_mesh)
    grid_pos = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)

    # Apply the same normalisation used during training
    grid_norm = (grid_pos - pos_min) / pos_scale * 2.0 - 1.0

    # Batched forward pass (avoids OOM on large grids)
    preds = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(grid_norm), batch_size):
            chunk = torch.tensor(grid_norm[i:i + batch_size], dtype=torch.float32)
            bi_chunk = None
            if band_idx is not None and model.n_bands > 0:
                bi_chunk = torch.full((chunk.shape[0],), band_idx, dtype=torch.long)
            preds.append(model(chunk, band_idx=bi_chunk).numpy())
    pred_np = np.concatenate(preds, axis=0) * off_scale   # → arcsec

    mesh_shape = (len(y_mesh), len(x_mesh))
    dra  = pred_np[:, 0].reshape(mesh_shape).astype(np.float32)
    ddec = pred_np[:, 1].reshape(mesh_shape).astype(np.float32)

    # Coverage: min distance (arcsec) to nearest training anchor
    coverage = None
    if pos_arcsec_anchors is not None:
        tree = KDTree(pos_arcsec_anchors)
        dists, _ = tree.query(grid_pos)
        coverage = dists.reshape(mesh_shape).astype(np.float32)

    return {
        'dra':      dra,
        'ddec':     ddec,
        'coverage': coverage,
        'x_mesh':   x_mesh,
        'y_mesh':   y_mesh,
    }
