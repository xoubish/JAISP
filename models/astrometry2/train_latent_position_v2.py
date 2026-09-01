"""Train latent-position head variants targeting the nb31 coherent-worsener bias.

Two variants (nb31: ~315 tight-core galaxies where the head is precise-but-biased):

  biaspen   Per-source bias penalty. Two independent jitters of the same source
            per batch; the dot product of their residuals is an unbiased
            estimator of ||bias||^2 (E[r1.r2] = |b|^2), penalized WITHOUT
            sigma normalization so bias cannot hide behind inflated sigma.
            Head architecture unchanged (checkpoint drop-in compatible).

  bandaware Band-aware multi-task head. Adds a wavelength embedding to the MLP
            input and an auxiliary output trained to find the query band's own
            classical centroid, while the main output still targets the VIS
            convention. Band queries/targets come from the anchors npz
            (classical quantities only — head-independent), restricted to
            training tiles.

Usage (per GPU):
  python train_latent_position_v2.py --variant biaspen   --device cuda:0 ...
  python train_latent_position_v2.py --variant bandaware --device cuda:1 ...
"""

import argparse
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

import sys
SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_DIR = SCRIPT_DIR.parent
for p in (MODELS_DIR, SCRIPT_DIR):
    sp = str(p)
    if sp in sys.path:
        sys.path.remove(sp)
    sys.path.insert(0, sp)

from astrometry2.dataset import (
    discover_tile_pairs,
    split_tile_pairs,
    local_vis_pixel_to_sky_matrix,
)
from astrometry2.latent_position_head import (
    LatentPositionHead,
    load_latent_position_head,
    extract_local_windows,
    vis_px_to_bottleneck_px,
)
from astrometry2.train_latent_position import (
    load_tile_data,
    encode_tile_features,
    compute_loss,
    compute_metrics,
    serializable_args,
)
from source_matching import safe_header_from_card_string  # noqa: F401

try:
    import wandb
except ImportError:
    wandb = None

BANDS = ['u', 'g', 'r', 'i', 'z', 'y', 'nisp_Y', 'nisp_J', 'nisp_H']
LAM = {b: l for b, l in zip(BANDS, [0.368, 0.480, 0.622, 0.754, 0.869,
                                    0.971, 1.081, 1.273, 1.773])}
LAM_VIS = 0.715


# ============================================================
# Band-aware head
# ============================================================

class BandAwareHead(LatentPositionHead):
    """LatentPositionHead + wavelength embedding + auxiliary band-center output.

    Output: (dx_main, dy_main, dx_aux, dy_aux, log_sigma) in VIS px / log-arcsec.
    main = offset to the VIS-convention canonical position;
    aux  = offset to the query band's own classical centroid.
    """

    LAM_FEATS = 8

    def __init__(self, **kw):
        super().__init__(**kw)
        feat_dim = self.head[0].in_features
        mlp_hidden = self.head[0].out_features
        self.head = nn.Sequential(
            nn.Linear(feat_dim + self.LAM_FEATS, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, 5),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        with torch.no_grad():
            self.head[-1].bias[4] = float(np.log(0.05))

    def _lam_embed(self, lam: torch.Tensor) -> torch.Tensor:
        ln = (lam - 0.35) / 1.45                       # ~[0, 1]
        ks = torch.tensor([1., 2., 4., 8.], device=lam.device, dtype=lam.dtype)
        ph = np.pi * ln.unsqueeze(1) * ks.unsqueeze(0)  # [N, 4]
        return torch.cat([torch.sin(ph), torch.cos(ph)], dim=1)  # [N, 8]

    def forward(self, bottleneck, vis_stem_features, source_positions_vis,
                pixel_to_sky, fused_hw, vis_hw, lam=None):
        N = source_positions_vis.shape[0]
        positions_bn = vis_px_to_bottleneck_px(
            source_positions_vis, self.vis_pixel_scale, self.fused_pixel_scale,
            fused_hw, vis_hw,
        )
        bn_windows = extract_local_windows(bottleneck, positions_bn, self.bottleneck_window)
        bn_vec = self._gauss_pool(self.bn_conv(bn_windows), self.bn_gauss)
        stem_windows = extract_local_windows(
            vis_stem_features, source_positions_vis, self.stem_window)
        stem_vec = self._gauss_pool(self.stem_conv(stem_windows), self.stem_gauss)

        if lam is None:
            lam = torch.full((N,), LAM_VIS, device=source_positions_vis.device,
                             dtype=source_positions_vis.dtype)
        combined = torch.cat([bn_vec, stem_vec, self._lam_embed(lam)], dim=1)
        out = self.head(combined)

        dx_px, dy_px = out[:, 0], out[:, 1]
        aux_px = out[:, 2:4]
        log_sigma = out[:, 4].clamp(-6.0, 3.0)
        pix = torch.stack([dx_px, dy_px], dim=1).unsqueeze(-1)
        pred_sky = torch.bmm(pixel_to_sky, pix).squeeze(-1)
        sigma = torch.exp(log_sigma)
        return {
            'pred_offset_arcsec': pred_sky,
            'dx_px': dx_px, 'dy_px': dy_px,
            'aux_px': aux_px,
            'log_sigma': log_sigma,
            'confidence': 1.0 / sigma.clamp_min(1e-4),
        }


# ============================================================
# Bounded / anchored variants
# ============================================================

FWHM_MAS = {'u': 900., 'g': 800., 'r': 750., 'i': 700., 'z': 680., 'y': 680.,
            'nisp_Y': 500., 'nisp_J': 500., 'nisp_H': 550., 'VIS': 160.}


def classical_sigma_px(fwhm_mas, snr):
    """Rough classical centroid noise (SITCOMTN-159: sigma ~ FWHM/SNR), in VIS px."""
    s = np.asarray(fwhm_mas, dtype=np.float32) / (2.355 * np.clip(snr, 3., None))
    return np.clip(s, 3., 400.) / 100.                       # mas -> px @ 0.1"/px


def soft_centroid(win, iters=2, sigma_px=2.0):
    """Differentiable-free iterated Gaussian-weighted centroid on raw VIS windows.

    win : [N, S, S] raw VIS pixel windows centered exactly on the query.
    Returns (offset_px [N,2] rel. to query, sigma_anchor_px [N]).
    Mimics the windowed-centroid convention the canonical labels follow.
    """
    N, S, _ = win.shape
    border = torch.cat([win[:, 0, :], win[:, -1, :], win[:, :, 0], win[:, :, -1]], dim=1)
    bg = border.median(dim=1).values
    noise = 1.4826 * (border - bg[:, None]).abs().median(dim=1).values + 1e-9
    img = (win - bg[:, None, None]).clamp_min(0.)
    ax = torch.arange(S, device=win.device, dtype=win.dtype) - (S - 1) / 2.
    yy, xx = torch.meshgrid(ax, ax, indexing='ij')
    cx = torch.zeros(N, device=win.device, dtype=win.dtype)
    cy = torch.zeros(N, device=win.device, dtype=win.dtype)
    tot = img.sum(dim=(1, 2)).clamp_min(1e-9)
    for _ in range(iters):
        w = img * torch.exp(-(((xx[None] - cx[:, None, None]) ** 2 +
                               (yy[None] - cy[:, None, None]) ** 2) / (2 * sigma_px ** 2)))
        tot = w.sum(dim=(1, 2)).clamp_min(1e-9)
        cx = ((w * xx[None]).sum(dim=(1, 2)) / tot).clamp(-4., 4.)
        cy = ((w * yy[None]).sum(dim=(1, 2)) / tot).clamp(-4., 4.)
    snr_loc = tot / (noise * 2. * sigma_px * float(np.sqrt(np.pi)))
    sig_anchor = (2.0 / snr_loc.clamp(1., 1e4)).clamp(0.02, 4.0)   # px
    return torch.stack([cx, cy], dim=1), sig_anchor


class BoundedBandAwareHead(BandAwareHead):
    """BandAwareHead whose correction is tanh-bounded by ~3x classical noise."""

    def forward(self, bottleneck, vis_stem_features, source_positions_vis,
                pixel_to_sky, fused_hw, vis_hw, lam=None, bound_px=None):
        out = super().forward(bottleneck, vis_stem_features, source_positions_vis,
                              pixel_to_sky, fused_hw, vis_hw, lam=lam)
        if bound_px is not None:
            B = bound_px.clamp(0.15, 2.5)                    # 15-250 mas
            dx = B * torch.tanh(out['dx_px'] / B)
            dy = B * torch.tanh(out['dy_px'] / B)
            pix = torch.stack([dx, dy], dim=1).unsqueeze(-1)
            out['dx_px'], out['dy_px'] = dx, dy
            out['pred_offset_arcsec'] = torch.bmm(pixel_to_sky, pix).squeeze(-1)
        return out


class AnchoredHead(BandAwareHead):
    """Convention-anchored head: prediction = VIS windowed-centroid anchor
    + residual bounded by ~3x the anchor's own noise. The anchor is the same
    algorithm family the canonical labels come from, so on bright structured
    hosts the head cannot drag the position away from the convention."""

    ANCHOR_WIN = 17
    K_BOUND = 3.0

    def forward(self, bottleneck, vis_stem_features, source_positions_vis,
                pixel_to_sky, fused_hw, vis_hw, lam=None, vis_img=None):
        out = super().forward(bottleneck, vis_stem_features, source_positions_vis,
                              pixel_to_sky, fused_hw, vis_hw, lam=lam)
        if vis_img is None:
            return out
        with torch.no_grad():
            win = extract_local_windows(
                vis_img, source_positions_vis, self.ANCHOR_WIN)[:, 0]
            anchor, sig_anchor = soft_centroid(win)
        B = (self.K_BOUND * sig_anchor).clamp(0.05, 2.5)     # 5-250 mas
        dx = anchor[:, 0] + B * torch.tanh(out['dx_px'] / B)
        dy = anchor[:, 1] + B * torch.tanh(out['dy_px'] / B)
        pix = torch.stack([dx, dy], dim=1).unsqueeze(-1)
        out['dx_px'], out['dy_px'] = dx, dy
        out['pred_offset_arcsec'] = torch.bmm(pixel_to_sky, pix).squeeze(-1)
        return out


# ============================================================
# Batch builders
# ============================================================

def _jitter(rng, n, jitter_arcsec, jitter_max_arcsec):
    jp = jitter_arcsec / 0.1
    j = rng.normal(scale=max(jp, 0.01), size=(n, 2)).astype(np.float32)
    if jitter_max_arcsec > 0:
        mp = jitter_max_arcsec / 0.1
        j = np.clip(j, -mp, mp)
    return j


def _rows(vis_xy_true, queries, vis_wcs, device):
    """Targets/Jacobians for query rows whose truth is vis_xy_true (VIS px)."""
    n = queries.shape[0]
    tgt = np.zeros((n, 2), dtype=np.float32)
    p2s = np.zeros((n, 2, 2), dtype=np.float32)
    for i in range(n):
        m = local_vis_pixel_to_sky_matrix(vis_wcs, queries[i])
        d = vis_xy_true[i] - queries[i]
        tgt[i] = m @ d.astype(np.float32)
        p2s[i] = m
    return (torch.from_numpy(queries).to(device),
            torch.from_numpy(tgt).to(device),
            torch.from_numpy(p2s).to(device))


def batch_biaspen(vis_xy, vis_wcs, args, rng, device):
    """Two independent jitters per source, stacked [r1-block, r2-block]."""
    N = vis_xy.shape[0]
    if N < 5:
        return None
    if N > args.max_sources_per_tile:
        idx = rng.choice(N, args.max_sources_per_tile, replace=False)
        vis_xy = vis_xy[idx]
        N = args.max_sources_per_tile
    qs, tgts, p2ss = [], [], []
    for _ in range(2):
        q = vis_xy + _jitter(rng, N, args.jitter_arcsec, args.jitter_max_arcsec)
        a, b, c = _rows(vis_xy, q, vis_wcs, device)
        qs.append(a); tgts.append(b); p2ss.append(c)
    return {
        'positions': torch.cat(qs), 'target_offset_arcsec': torch.cat(tgts),
        'pixel_to_sky': torch.cat(p2ss), 'n_src': N,
    }


def batch_bandaware(vis_xy, vis_snr, vis_wcs, tile_anchors, args, rng, device, vis_hw,
                    no_jitter=False):
    """VIS-jitter rows (lam=VIS) + band-centroid rows from the anchors npz.

    Also emits per-row `bound_px` = 3 x classical centroid noise for the row's
    band + S/N (used by the bounded variant; harmless to the others).
    With no_jitter=True band-row queries are the raw classical centroids
    (validation fidelity: |target| is then exactly the classical error).
    """
    jit_as = 0. if no_jitter else args.jitter_arcsec
    jit_mx = 0. if no_jitter else args.jitter_max_arcsec
    rows_q, rows_tgt, rows_p2s, rows_lam, rows_aux, rows_bnd = [], [], [], [], [], []

    N = vis_xy.shape[0]
    if N > args.max_sources_per_tile:
        idx = rng.choice(N, args.max_sources_per_tile, replace=False)
        vis_xy = vis_xy[idx]; vis_snr = vis_snr[idx]
        N = args.max_sources_per_tile
    if N >= 5:
        j = _jitter(rng, N, jit_as, jit_mx)
        q, tgt, p2s = _rows(vis_xy, vis_xy + j, vis_wcs, device)
        rows_q.append(q); rows_tgt.append(tgt); rows_p2s.append(p2s)
        rows_lam.append(torch.full((N,), LAM_VIS, device=device))
        rows_aux.append(torch.from_numpy(-j).to(device))   # aux: band==VIS centroid==truth
        rows_bnd.append(torch.from_numpy(
            3. * classical_sigma_px(FWHM_MAS['VIS'], vis_snr)).to(device))

    H, W = vis_hw
    if tile_anchors:
        for b, (ra, dec, raw, bsnr) in tile_anchors.items():
            n = len(ra)
            if n == 0:
                continue
            take = min(n, args.band_rows_per_band)
            sel = rng.choice(n, take, replace=False) if n > take else np.arange(n)
            bx, by = vis_wcs.wcs_world2pix(ra[sel], dec[sel], 0)
            bxy = np.stack([bx, by], axis=1).astype(np.float32)
            ok = ((bxy[:, 0] > 20) & (bxy[:, 0] < W - 20) &
                  (bxy[:, 1] > 20) & (bxy[:, 1] < H - 20))
            if not ok.any():
                continue
            bxy = bxy[ok]
            rr = raw[sel][ok].astype(np.float32)
            ss = bsnr[sel][ok].astype(np.float32)
            j = _jitter(rng, len(bxy), jit_as, jit_mx)
            q = bxy + j
            n2 = len(q)
            tgt = np.zeros((n2, 2), dtype=np.float32)
            p2s = np.zeros((n2, 2, 2), dtype=np.float32)
            for i in range(n2):
                m = local_vis_pixel_to_sky_matrix(vis_wcs, q[i])
                # main target: VIS label = band centroid + raw  =>  raw - m@j
                tgt[i] = rr[i] - m @ j[i]
                p2s[i] = m
            rows_q.append(torch.from_numpy(q).to(device))
            rows_tgt.append(torch.from_numpy(tgt).to(device))
            rows_p2s.append(torch.from_numpy(p2s).to(device))
            rows_lam.append(torch.full((n2,), LAM[b], device=device))
            rows_aux.append(torch.from_numpy(-j).to(device))
            rows_bnd.append(torch.from_numpy(
                3. * classical_sigma_px(FWHM_MAS[b], ss)).to(device))

    if not rows_q:
        return None
    q = torch.cat(rows_q)
    if q.shape[0] < 5:
        return None
    return {
        'positions': q,
        'target_offset_arcsec': torch.cat(rows_tgt),
        'pixel_to_sky': torch.cat(rows_p2s),
        'lam': torch.cat(rows_lam),
        'aux_target_px': torch.cat(rows_aux),
        'bound_px': torch.cat(rows_bnd).float(),
        'n_src': q.shape[0],
    }


# ============================================================
# Losses
# ============================================================

def loss_biaspen(out, batch, args):
    base = compute_loss(out, batch['target_offset_arcsec'], batch['pixel_to_sky'],
                        label_noise_floor=args.label_noise_floor)
    r = out['pred_offset_arcsec'] - batch['target_offset_arcsec']
    N = batch['n_src']
    r1, r2 = r[:N], r[N:2 * N]
    # E[r1.r2] = ||bias||^2 ; fixed normalization so sigma cannot absorb it.
    loss_bias = (r1 * r2).sum(dim=1).mean() / (0.010 ** 2)
    base['loss_bias'] = loss_bias
    base['loss_total'] = base['loss_total'] + args.bias_weight * loss_bias
    return base


def loss_bandaware(out, batch, args):
    base = compute_loss(out, batch['target_offset_arcsec'], batch['pixel_to_sky'],
                        label_noise_floor=args.label_noise_floor)
    aux = torch.nn.functional.smooth_l1_loss(
        out['aux_px'], batch['aux_target_px'], reduction='mean')
    base['loss_aux'] = aux
    base['loss_total'] = base['loss_total'] + args.aux_weight * aux
    return base


def loss_hinge(out, batch, args):
    """bandaware loss + do-no-harm hinge: penalize rows where the head lands
    MATERIALLY farther from the label than the query itself (the classical
    baseline) — the nb31 worsener criterion (res > raw+10 mas and > 20 mas) as
    a differentiable term. NOT sigma-normalized: inflating sigma does not pay
    for it."""
    base = loss_bandaware(out, batch, args)
    tgt = batch['target_offset_arcsec']
    res = torch.sqrt(((out['pred_offset_arcsec'] - tgt) ** 2).sum(dim=1) + 1e-10)
    rawq = torch.sqrt((tgt ** 2).sum(dim=1) + 1e-10)         # classical baseline error
    thresh = torch.maximum(rawq + 0.010, torch.full_like(rawq, 0.020))
    lh = torch.relu(res - thresh) / 0.010                    # units of 10 mas
    base['loss_hinge'] = lh.mean()
    base['frac_hinged'] = (lh > 0).float().mean()
    base['loss_total'] = base['loss_total'] + args.hinge_weight * lh.mean()
    return base


# ============================================================
# Anchors npz -> per-tile band training rows
# ============================================================

def load_band_anchors(npz_path, allowed_tiles):
    d = np.load(npz_path, allow_pickle=True)
    out: Dict[str, Dict[str, list]] = {}
    for b in BANDS:
        ra = np.asarray(d[f'{b}_ra'], dtype=np.float64)
        dec = np.asarray(d[f'{b}_dec'], dtype=np.float64)
        raw = np.asarray(d[f'{b}_raw'], dtype=np.float32)
        snr = np.asarray(d[f'{b}_snr'], dtype=np.float32)
        tiles = np.asarray(d[f'{b}_tiles'])
        for t in np.unique(tiles):
            if t not in allowed_tiles:
                continue
            m = tiles == t
            out.setdefault(t, {})[b] = (ra[m], dec[m], raw[m], snr[m])
    n_rows = sum(len(v[0]) for tb in out.values() for v in tb.values())
    print(f'Band anchors: {len(out)} tiles, {n_rows:,} rows (train tiles only)')
    return out


# ============================================================
# Training
# ============================================================

def make_raw_vs_head_epoch_fig(pairs, epoch, out_dir):
    """Per-epoch raw-vs-head hexbin on val-tile classical band queries."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        raw, res = pairs
        m = (raw > 0.3) & (res > 0.3)
        raw, res = raw[m], res[m]
        fig, ax = plt.subplots(figsize=(6.5, 6))
        hb = ax.hexbin(raw, res, gridsize=45, xscale='log', yscale='log',
                       bins='log', cmap='cividis', extent=(np.log10(0.5), np.log10(250),
                                                           np.log10(0.5), np.log10(250)))
        fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.02, label='N (log)')
        ax.plot([0.5, 250], [0.5, 250], 'k--', lw=1)
        ax.set_xlim(0.5, 250); ax.set_ylim(0.5, 250); ax.set_aspect('equal')
        ax.set_xlabel('raw (classical) offset [mas]')
        ax.set_ylabel('head residual [mas]')
        ax.set_title(f'epoch {epoch}: {(res < raw).mean():.0%} below 1:1, '
                     f'median {np.median(raw):.0f}$\to${np.median(res):.0f} mas', fontsize=11)
        fig.tight_layout()
        figs = Path(out_dir) / 'epoch_figs'
        figs.mkdir(exist_ok=True)
        path = figs / f'raw_vs_head_e{epoch:03d}.png'
        fig.savefig(path, dpi=110)
        plt.close(fig)
        return path
    except Exception as exc:
        print(f'[warn] epoch fig failed: {exc}')
        return None


def run_epoch_v2(split, pairs, frozen_encoder, head, optimizer, device, args, rng):
    is_train = optimizer is not None
    head.train(mode=is_train)
    agg = defaultdict(float)
    n_tiles = 0
    pairs_raw, pairs_res = [], []
    order = rng.permutation(len(pairs)) if is_train else np.arange(len(pairs))

    for idx in order:
        tile_id, rubin_path, euclid_path = pairs[idx]
        if tile_id not in args._canonical_labels_dict:
            continue
        try:
            img_t, rms_t, vis_hw, vis_wcs = load_tile_data(rubin_path, euclid_path, device)
        except Exception:
            continue
        enc_out = encode_tile_features(
            frozen_encoder, tile_id, img_t, rms_t, device,
            features_cache_dir=args._features_cache_dir,
        )
        vis_img_t = None
        if args.variant == 'anchored':
            t = img_t['euclid_VIS']
            while t.ndim < 4:
                t = t.unsqueeze(0)
            vis_img_t = torch.nan_to_num(t.float()).to(device)
        del img_t, rms_t
        entry = args._canonical_labels_dict[tile_id]
        vis_xy = np.asarray(entry['xy'], dtype=np.float32).copy()
        vis_snr = np.asarray(entry.get('snr', np.full(len(vis_xy), 20.)),
                             dtype=np.float32).copy()

        if args.variant == 'biaspen':
            batch = batch_biaspen(vis_xy, vis_wcs, args, rng, device)
        else:
            ba = (args._band_anchors if is_train else
                  getattr(args, '_band_anchors_val', {}))
            batch = batch_bandaware(
                vis_xy, vis_snr, vis_wcs, ba.get(tile_id),
                args, rng, device, vis_hw, no_jitter=not is_train,
            )
        if batch is None:
            continue

        if is_train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_train):
            kw = {}
            if args.variant in ('bandaware', 'bounded', 'anchored', 'hinge'):
                kw['lam'] = batch['lam']
            if args.variant == 'bounded':
                kw['bound_px'] = batch['bound_px']
            if args.variant == 'anchored':
                kw['vis_img'] = vis_img_t
            out = head(enc_out['bottleneck'], enc_out['vis_stem'],
                       batch['positions'], batch['pixel_to_sky'],
                       enc_out['fused_hw'], vis_hw, **kw)
            if args.variant == 'biaspen':
                losses = loss_biaspen(out, batch, args)
            elif args.variant == 'hinge':
                losses = loss_hinge(out, batch, args)
            else:
                losses = loss_bandaware(out, batch, args)
            if is_train:
                losses['loss_total'].backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(head.parameters(), args.grad_clip)
                optimizer.step()

        if (not is_train) and args.variant != 'biaspen' and 'lam' in batch:
            with torch.no_grad():
                bm = batch['lam'] != LAM_VIS
                if bm.any():
                    tg = batch['target_offset_arcsec'][bm]
                    rr = torch.sqrt((tg ** 2).sum(1) + 1e-12) * 1000.
                    rs = torch.sqrt(((out['pred_offset_arcsec'][bm] - tg) ** 2).sum(1) + 1e-12) * 1000.
                    pairs_raw.append(rr.cpu().numpy()); pairs_res.append(rs.cpu().numpy())
        metrics = compute_metrics(out, batch['target_offset_arcsec'])
        for k, v in losses.items():
            agg[k] += float(v.detach())
        for k, v in metrics.items():
            agg[k] += float(v)
        n_tiles += 1
        del enc_out, batch, out
        if args.limit_tiles and n_tiles >= args.limit_tiles:
            break

    denom = max(1, n_tiles)
    result = {k: v / denom for k, v in agg.items()}
    result['tiles'] = n_tiles
    if pairs_raw:
        result['_pairs'] = (np.concatenate(pairs_raw), np.concatenate(pairs_res))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--variant', required=True,
                   choices=['biaspen', 'bandaware', 'bounded', 'anchored', 'hinge'])
    p.add_argument('--rubin-dir', default='data/rubin_tiles_all')
    p.add_argument('--euclid-dir', default='data/euclid_tiles_all_q1')
    p.add_argument('--foundation-checkpoint',
                   default='models/checkpoints/jaisp_v11_q1_soft/checkpoint_best.pt')
    p.add_argument('--canonical-labels',
                   default='data/detection_labels/vis_refined_labels_q1_vissep.pt')
    p.add_argument('--features-cache-dir', default='data/cached_features_v11_q1')
    p.add_argument('--band-anchors',
                   default='models/checkpoints/latent_position_v11_q1_plain/'
                           'anchors_centernet_v11plain.npz')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--max-sources-per-tile', type=int, default=200)
    p.add_argument('--band-rows-per-band', type=int, default=40)
    p.add_argument('--jitter-arcsec', type=float, default=0.03)
    p.add_argument('--jitter-max-arcsec', type=float, default=0.1)
    p.add_argument('--bias-weight', type=float, default=0.1)
    p.add_argument('--aux-weight', type=float, default=0.3)
    p.add_argument('--hinge-weight', type=float, default=1.0)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--grad-clip', type=float, default=1.0)
    p.add_argument('--label-noise-floor', type=float, default=0.005)
    p.add_argument('--val-frac', type=float, default=0.15)
    p.add_argument('--val-patches', type=str, default=None,
                   help='Comma-separated patch ids held out as a spatially '
                        'disjoint val set (overrides --val-frac).')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--limit-tiles', type=int, default=0,
                   help='Debug: cap tiles per epoch (0 = all).')
    p.add_argument('--wandb-project', default='JAISP-LatentPosition')
    p.add_argument('--wandb-run-name', default='')
    p.add_argument('--wandb-mode', default='online',
                   choices=['online', 'offline', 'disabled'])
    args = p.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = discover_tile_pairs(args.rubin_dir, args.euclid_dir)
    if args.val_patches:
        ids = {x.strip() for x in args.val_patches.split(',') if x.strip()}
        train_pairs = [q for q in pairs if q[0].rsplit('_patch_', 1)[-1] not in ids]
        val_pairs = [q for q in pairs if q[0].rsplit('_patch_', 1)[-1] in ids]
        print(f'Patch-disjoint split: val patches {sorted(ids)}')
    else:
        train_pairs, val_pairs = split_tile_pairs(pairs, args.val_frac, args.seed)
    print(f'Tiles: {len(train_pairs)} train, {len(val_pairs)} val')
    args._features_cache_dir = (Path(args.features_cache_dir)
                                if args.features_cache_dir else None)

    cl = torch.load(args.canonical_labels, map_location='cpu', weights_only=False)
    args._canonical_labels_dict = cl['labels'] if 'labels' in cl else cl
    print(f'Canonical labels: {len(args._canonical_labels_dict)} tiles')

    frozen_encoder, head = load_latent_position_head(
        args.foundation_checkpoint, device=device,
    )
    if args.variant in ('bandaware', 'bounded', 'anchored', 'hinge'):
        base = head
        cls = {'bandaware': BandAwareHead, 'bounded': BoundedBandAwareHead,
               'anchored': AnchoredHead, 'hinge': BandAwareHead}[args.variant]
        head = cls(
            hidden_ch=base.bn_conv[1].in_channels,
            stem_ch=base.stem_conv[0].in_channels,
            bottleneck_out=base.bn_conv[1].out_channels,
            stem_out=base.stem_conv[2].out_channels,
            mlp_hidden=128,
            bottleneck_window=base.bottleneck_window,
            stem_window=base.stem_window,
            fused_pixel_scale=base.fused_pixel_scale,
            vis_pixel_scale=base.vis_pixel_scale,
        ).to(device)
        train_tile_ids = {t for t, _, _ in train_pairs}
        val_tile_ids = {t for t, _, _ in val_pairs}
        args._band_anchors = load_band_anchors(args.band_anchors, train_tile_ids)
        args._band_anchors_val = load_band_anchors(args.band_anchors, val_tile_ids)
    else:
        args._band_anchors = {}
        args._band_anchors_val = {}

    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)

    wandb_run = None
    if args.wandb_mode != 'disabled' and wandb is not None:
        try:
            wandb_run = wandb.init(project=args.wandb_project,
                                   name=args.wandb_run_name or None,
                                   config=serializable_args(args),
                                   mode=args.wandb_mode, dir=str(out_dir))
        except Exception as exc:
            print(f'W&B init failed: {exc}')

    rng = np.random.RandomState(args.seed)
    best_mae = float('inf')
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr = run_epoch_v2('train', train_pairs, frozen_encoder, head,
                          optimizer, device, args, rng)
        scheduler.step()
        va = run_epoch_v2('val', val_pairs, frozen_encoder, head,
                          None, device, args, rng) if val_pairs else {}
        dt = time.time() - t0
        extras = ''
        if 'loss_bias' in tr:
            extras = f'bias={tr["loss_bias"]:.3f} '
        if 'loss_aux' in tr:
            extras = f'aux={tr["loss_aux"]:.3f} '
        if 'loss_hinge' in tr:
            extras += f'hinge={tr["loss_hinge"]:.3f}({tr.get("frac_hinged",0):.1%}) '
        print(f'E{epoch:3d} | train MAE={tr.get("mae_total", 0)*1000:.1f} '
              f'σ={tr.get("sigma_median", 0)*1000:.1f} {extras}| '
              f'val MAE={va.get("mae_total", 0)*1000:.1f} '
              f'σ={va.get("sigma_median", 0)*1000:.1f} | '
              f'lr={optimizer.param_groups[0]["lr"]:.1e} | {dt:.0f}s | '
              f'{tr.get("tiles", 0)}+{va.get("tiles", 0)} tiles', flush=True)
        pairs = va.pop('_pairs', None)
        if pairs is not None:
            fig_path = make_raw_vs_head_epoch_fig(pairs, epoch, out_dir)
            va['frac_worsened'] = float((pairs[1] > pairs[0]).mean())
            va['frac_material'] = float(((pairs[1] > pairs[0] + 10.) & (pairs[1] > 20.)).mean())
        if wandb_run:
            log_d = {f'train/{k}': v for k, v in tr.items() if k != '_pairs'}
            log_d.update({f'val/{k}': v for k, v in va.items()})
            log_d['lr'] = optimizer.param_groups[0]['lr']
            if pairs is not None and fig_path is not None:
                log_d['val/raw_vs_head_hex'] = wandb.Image(str(fig_path))
            wandb_run.log(log_d, step=epoch)

        score = va.get('mae_total', tr.get('mae_total', float('inf')))
        ckpt = {
            'epoch': epoch,
            'head_state_dict': head.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_metrics': tr, 'val_metrics': va,
            'config': serializable_args(args),
            'head_class': type(head).__name__,
        }
        torch.save(ckpt, out_dir / 'latest.pt')
        if score < best_mae:
            best_mae = score
            torch.save(ckpt, out_dir / 'best.pt')
            print(f'  -> new best: MAE={score*1000:.1f} mas', flush=True)

    print(f'Done. Best val MAE: {best_mae*1000:.1f} mas | {out_dir}')
    if wandb_run:
        wandb_run.finish()


if __name__ == '__main__':
    main()
