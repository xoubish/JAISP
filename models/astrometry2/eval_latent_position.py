"""Evaluate the latent position head: align all 9 bands to VIS.

For each detected source, centroid it independently in each of the 9
non-VIS bands (6 Rubin + 3 NISP), project that position onto the VIS
frame, and ask the latent position head to correct it toward the VIS
PSF-fit centroid.  This tests per-object, per-band alignment using the
full multi-band latent representation.

Three regimes are compared per band:
  1. Raw offset: band centroid projected to VIS frame vs VIS centroid
  2. Head-corrected: head prediction applied to the projected position
  3. Theoretical floor: King-formula centroid noise at that SNR

Usage:
    cd models && python astrometry2/eval_latent_position.py \
        --rubin-dir  ../data/rubin_tiles_200 \
        --euclid-dir ../data/euclid_tiles_200 \
        --v7-checkpoint checkpoints/jaisp_v7_concat/checkpoint_best.pt \
        --head-checkpoint checkpoints/latent_position_head/best.pt \
        --output-dir checkpoints/latent_position_head/eval_cross_instrument
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_DIR = SCRIPT_DIR.parent
for p in (MODELS_DIR, SCRIPT_DIR):
    sp = str(p)
    if sp in sys.path:
        sys.path.remove(sp)
    sys.path.insert(0, sp)

from astrometry2.dataset import (
    build_full_context_detector_inputs,
    detect_sources,
    discover_tile_pairs,
    local_vis_pixel_to_sky_matrix,
    signal_mask_in_band,
    _to_float32,
    NISP_BAND_ORDER,
)
from astrometry2.latent_position_head import (
    LatentPositionHead,
    FrozenV7Encoder,
    load_latent_position_head,
)
from astrometry2.train_latent_position import load_tile_data
from source_matching import (
    RUBIN_BAND_ORDER,
    build_detection_image,
    detect_sources as detect_sources_classical,
    refine_centroids_psf_fit,
    safe_header_from_card_string,
)
from astropy.wcs import WCS


# ============================================================
# Per-band centroiding and projection to VIS
# ============================================================

def centroid_in_band_and_project(
    band_img: np.ndarray,
    band_wcs: WCS,
    vis_xy: np.ndarray,
    vis_wcs: WCS,
    refine_radius: int = 3,
    fwhm_guess: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Centroid sources in a single band and project back to VIS frame.

    Parameters
    ----------
    band_img : [H, W] image in this band
    band_wcs : WCS for this band
    vis_xy : [N, 2] VIS pixel positions of matched sources (used as seeds)
    vis_wcs : VIS WCS
    refine_radius : PSF-fit radius
    fwhm_guess : initial FWHM for Gaussian fit

    Returns
    -------
    band_xy_in_vis : [N, 2] band centroid projected to VIS pixels
    offset_arcsec : [N, 2] (VIS_centroid - band_centroid) in arcsec
    valid : [N] bool mask
    snr : [N] peak SNR in this band
    """
    N = vis_xy.shape[0]

    # Project VIS positions into the band's pixel frame
    ra, dec = vis_wcs.wcs_pix2world(vis_xy[:, 0], vis_xy[:, 1], 0)
    bx, by = band_wcs.wcs_world2pix(ra, dec, 0)
    seed_xy = np.stack([bx, by], axis=1).astype(np.float32)

    H, W = band_img.shape

    # Check which sources have signal in this band
    valid = signal_mask_in_band(band_img, seed_xy, radius=refine_radius)
    # Also check margin
    margin = refine_radius + 2
    in_bounds = (
        (seed_xy[:, 0] >= margin) & (seed_xy[:, 0] < W - margin) &
        (seed_xy[:, 1] >= margin) & (seed_xy[:, 1] < H - margin)
    )
    valid = valid & in_bounds

    # Outputs (default: no valid measurement)
    band_xy_in_vis = vis_xy.copy()  # fallback to VIS position
    offset_arcsec = np.zeros((N, 2), dtype=np.float32)
    snr = np.ones(N, dtype=np.float32)

    if not valid.any():
        return band_xy_in_vis, offset_arcsec, valid, snr

    # PSF-fit centroid in the band's native pixel frame
    refined_xy, band_snr, _ = refine_centroids_psf_fit(
        band_img, seed_xy[valid],
        radius=refine_radius, fwhm_guess=fwhm_guess,
    )
    snr[valid] = band_snr

    # Project refined band positions back to VIS frame
    r_ra, r_dec = band_wcs.wcs_pix2world(refined_xy[:, 0], refined_xy[:, 1], 0)
    rvx, rvy = vis_wcs.wcs_world2pix(r_ra, r_dec, 0)
    band_xy_in_vis[valid] = np.stack([rvx, rvy], axis=1).astype(np.float32)

    # Offset: VIS centroid - band centroid (in arcsec)
    v_ra, v_dec = vis_wcs.wcs_pix2world(vis_xy[valid, 0], vis_xy[valid, 1], 0)
    dra = (v_ra - r_ra) * np.cos(np.deg2rad(v_dec)) * 3600.0
    ddec = (v_dec - r_dec) * 3600.0
    offset_arcsec[valid] = np.stack([dra, ddec], axis=1).astype(np.float32)

    return band_xy_in_vis, offset_arcsec, valid, snr


# ============================================================
# Main evaluation
# ============================================================

def evaluate(args):
    device = torch.device(
        args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'),
    )
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    frozen_encoder, head = load_latent_position_head(
        args.v7_checkpoint, device=device,
    )
    ckpt = torch.load(args.head_checkpoint, map_location='cpu', weights_only=False)
    head.load_state_dict(ckpt['head_state_dict'])
    head.eval()
    print(f'Loaded head from {args.head_checkpoint} (epoch {ckpt.get("epoch", "?")})')

    pairs = discover_tile_pairs(args.rubin_dir, args.euclid_dir)
    print(f'Evaluating on {len(pairs)} tiles')

    # Define all 9 non-VIS bands
    rubin_bands = [f'rubin_{b}' for b in RUBIN_BAND_ORDER]  # u,g,r,i,z,y
    nisp_bands = [f'nisp_{b}' for b in NISP_BAND_ORDER]     # Y,J,H
    all_bands = rubin_bands + nisp_bands

    # Per-band accumulators
    band_results = {b: {'raw_err': [], 'head_err': [], 'snr': []} for b in all_bands}
    n_tiles = 0

    for tile_id, rubin_path, euclid_path in pairs:
        try:
            img_t, rms_t, vis_hw, vis_wcs = load_tile_data(rubin_path, euclid_path, device)
            rdata = np.load(rubin_path, allow_pickle=True)
            edata = np.load(euclid_path, allow_pickle=True)
            rubin_cube = rdata['img']
            vis_img = np.nan_to_num(_to_float32(edata['img_VIS']), nan=0.0)
            vhdr = safe_header_from_card_string(edata['wcs_VIS'].item())
            vwcs = WCS(vhdr)
            rwcs = WCS(rdata['wcs_hdr'].item())
        except Exception:
            continue

        # Detect in VIS and PSF-refine → these are the "true" positions
        vx, vy = detect_sources_classical(
            vis_img, nsig=4.0, smooth_sigma=1.2, min_dist=9, max_sources=400,
        )
        if vx.size < 10:
            continue
        vis_seed = np.stack([vx, vy], axis=1).astype(np.float32)
        vis_keep = signal_mask_in_band(vis_img, vis_seed, radius=3)
        if vis_keep.sum() < 10:
            continue
        vis_xy, vis_snr, _ = refine_centroids_psf_fit(
            vis_img, vis_seed[vis_keep], radius=3, fwhm_guess=2.5,
        )
        vis_xy = vis_xy.astype(np.float32)
        N = vis_xy.shape[0]

        # Run frozen encoder (once per tile)
        with torch.no_grad():
            enc_out = frozen_encoder.encode_tile(img_t, rms_t)
        del img_t, rms_t

        # ---- Per-band evaluation ----
        for band_name in all_bands:
            # Get the band image and WCS
            if band_name.startswith('rubin_'):
                short = band_name.split('_', 1)[1]
                bidx = RUBIN_BAND_ORDER.index(short)
                if bidx >= rubin_cube.shape[0]:
                    continue
                band_img = np.nan_to_num(_to_float32(rubin_cube[bidx]), nan=0.0)
                band_wcs = rwcs
                fwhm = 3.0  # Rubin PSF ~3 px at 0.2"/px
            elif band_name.startswith('nisp_'):
                short = band_name.split('_', 1)[1]
                img_key = f'img_{short}'
                wcs_key = f'wcs_{short}'
                if img_key not in edata or wcs_key not in edata:
                    continue
                band_img = np.nan_to_num(_to_float32(edata[img_key]), nan=0.0)
                try:
                    band_wcs = WCS(safe_header_from_card_string(edata[wcs_key].item()))
                except Exception:
                    continue
                fwhm = 2.5  # NISP PSF ~2.5 px at 0.1"/px
            else:
                continue

            # Centroid in this band, project to VIS frame
            band_xy_vis, offset_arcsec, valid, band_snr = centroid_in_band_and_project(
                band_img, band_wcs, vis_xy, vwcs,
                refine_radius=3, fwhm_guess=fwhm,
            )
            if valid.sum() < 3:
                continue

            # Raw error: distance from band centroid (in VIS frame) to VIS centroid
            raw_radial = np.sqrt((offset_arcsec[valid] ** 2).sum(axis=1))

            # Filter: clip large offsets (bad centroids / wrong matches)
            # and low-SNR sources before running the head.
            clip_mas = float(args.clip_mas)
            min_snr = float(args.min_snr)
            keep = np.ones(int(valid.sum()), dtype=bool)
            if clip_mas > 0:
                keep &= (raw_radial * 1000) < clip_mas
            if min_snr > 0:
                keep &= band_snr[valid] >= min_snr
            if keep.sum() < 3:
                continue

            raw_radial = raw_radial[keep]
            offset_valid = offset_arcsec[valid][keep]
            band_pos_valid = band_xy_vis[valid][keep]
            snr_valid = band_snr[valid][keep]

            # Head correction: predict offset from band position to canonical
            n_keep = int(keep.sum())
            pix2sky = np.zeros((n_keep, 2, 2), dtype=np.float32)
            for i in range(n_keep):
                pix2sky[i] = local_vis_pixel_to_sky_matrix(vwcs, band_pos_valid[i])

            with torch.no_grad():
                out = head(
                    enc_out['bottleneck'],
                    enc_out['vis_stem'],
                    torch.from_numpy(band_pos_valid).to(device),
                    torch.from_numpy(pix2sky).to(device),
                    enc_out['fused_hw'],
                    vis_hw,
                )

            pred_offset = out['pred_offset_arcsec'].cpu().numpy()
            residual = offset_valid - pred_offset
            head_radial = np.sqrt((residual ** 2).sum(axis=1))

            band_results[band_name]['raw_err'].extend((raw_radial * 1000).tolist())
            band_results[band_name]['head_err'].extend((head_radial * 1000).tolist())
            band_results[band_name]['snr'].extend(snr_valid.tolist())

        n_tiles += 1
        del enc_out

    # ---- Print results ----
    print(f'\n{"="*75}')
    print(f'Cross-instrument per-band alignment: {n_tiles} tiles')
    print(f'{"="*75}')
    print(f'{"Band":<12} {"N":>6}  {"Raw MAE":>8}  {"Raw med":>8}  {"Head MAE":>8}  {"Head med":>8}  {"Improve":>8}')
    print(f'{"":<12} {"":>6}  {"(mas)":>8}  {"(mas)":>8}  {"(mas)":>8}  {"(mas)":>8}  {"(%)":>8}')
    print('-' * 75)

    summary = {}
    for band_name in all_bands:
        br = band_results[band_name]
        if not br['raw_err']:
            continue
        raw = np.array(br['raw_err'])
        hd = np.array(br['head_err'])
        n = len(raw)
        raw_mae = np.mean(raw)
        raw_med = np.median(raw)
        hd_mae = np.mean(hd)
        hd_med = np.median(hd)
        improve = (1 - hd_mae / raw_mae) * 100 if raw_mae > 0 else 0

        print(f'{band_name:<12} {n:>6}  {raw_mae:>8.1f}  {raw_med:>8.1f}  {hd_mae:>8.1f}  {hd_med:>8.1f}  {improve:>7.1f}%')

        summary[band_name] = {
            'n_sources': n,
            'raw_mae_mas': float(raw_mae),
            'raw_median_mas': float(raw_med),
            'raw_p68_mas': float(np.percentile(raw, 68)),
            'head_mae_mas': float(hd_mae),
            'head_median_mas': float(hd_med),
            'head_p68_mas': float(np.percentile(hd, 68)),
            'improvement_pct': float(improve),
        }

    # Overall (all bands pooled)
    all_raw = np.concatenate([np.array(br['raw_err']) for br in band_results.values() if br['raw_err']])
    all_hd = np.concatenate([np.array(br['head_err']) for br in band_results.values() if br['head_err']])
    all_snr = np.concatenate([np.array(br['snr']) for br in band_results.values() if br['snr']])

    print('-' * 75)
    print(f'{"ALL":<12} {len(all_raw):>6}  {np.mean(all_raw):>8.1f}  {np.median(all_raw):>8.1f}  '
          f'{np.mean(all_hd):>8.1f}  {np.median(all_hd):>8.1f}  '
          f'{(1 - np.mean(all_hd)/np.mean(all_raw))*100:>7.1f}%')
    print(f'{"="*75}')

    summary['ALL'] = {
        'n_sources': len(all_raw),
        'raw_mae_mas': float(np.mean(all_raw)),
        'head_mae_mas': float(np.mean(all_hd)),
        'improvement_pct': float((1 - np.mean(all_hd) / np.mean(all_raw)) * 100),
    }

    with open(out_dir / 'results.json', 'w') as f:
        json.dump({'n_tiles': n_tiles, 'per_band': summary}, f, indent=2)
    print(f'\nResults saved to {out_dir / "results.json"}')

    _make_figure(band_results, all_bands, all_raw, all_hd, all_snr, out_dir)


def _make_figure(band_results, all_bands, all_raw, all_hd, all_snr, out_dir):
    """4-panel diagnostic."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # (0,0) Per-band MAE comparison (bar chart)
    ax = axes[0, 0]
    bands_with_data = [b for b in all_bands if band_results[b]['raw_err']]
    x = np.arange(len(bands_with_data))
    raw_maes = [np.mean(band_results[b]['raw_err']) for b in bands_with_data]
    hd_maes = [np.mean(band_results[b]['head_err']) for b in bands_with_data]
    w = 0.35
    ax.bar(x - w/2, raw_maes, w, label='Raw', color='tab:blue', alpha=0.7)
    ax.bar(x + w/2, hd_maes, w, label='Head-corrected', color='tab:red', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([b.replace('rubin_', 'R:').replace('nisp_', 'N:') for b in bands_with_data],
                       fontsize=8, rotation=45)
    ax.set_ylabel('MAE (mas)')
    ax.set_title('Per-band alignment: Raw vs Head-corrected')
    ax.legend(fontsize=8)
    ax.axhline(20, color='gray', ls=':', alpha=0.5)

    # (0,1) Overall error histogram
    ax = axes[0, 1]
    bins = np.linspace(0, min(150, np.percentile(all_raw, 98)), 50)
    ax.hist(all_raw, bins=bins, alpha=0.5, color='tab:blue',
            label=f'Raw: MAE={np.mean(all_raw):.1f} mas')
    ax.hist(all_hd, bins=bins, alpha=0.5, color='tab:red',
            label=f'Head: MAE={np.mean(all_hd):.1f} mas')
    ax.set_xlabel('Radial error (mas)')
    ax.set_ylabel('Count')
    ax.set_title('All bands pooled')
    ax.legend(fontsize=8)

    # (1,0) Error vs SNR (all bands pooled)
    ax = axes[1, 0]
    ax.scatter(all_snr, all_raw, s=2, alpha=0.15, c='tab:blue', label='Raw')
    ax.scatter(all_snr, all_hd, s=2, alpha=0.15, c='tab:red', label='Head-corrected')
    ax.set_xlabel('Band peak SNR')
    ax.set_ylabel('Radial error (mas)')
    ax.set_xscale('log')
    ax.set_ylim(0, min(200, np.percentile(all_raw, 98)))
    ax.set_title('Error vs SNR (all bands)')
    ax.legend(fontsize=8)
    ax.axhline(20, color='gray', ls=':', alpha=0.5)

    # (1,1) Binned median error vs SNR
    ax = axes[1, 1]
    snr_valid = all_snr[all_snr > 0]
    if len(snr_valid) > 50:
        snr_bins = np.logspace(np.log10(max(snr_valid.min(), 3)), np.log10(snr_valid.max()), 15)
        bin_idx = np.digitize(all_snr, snr_bins)
        centers, raw_med, hd_med = [], [], []
        for bi in range(1, len(snr_bins)):
            m = bin_idx == bi
            if m.sum() < 10:
                continue
            centers.append(np.median(all_snr[m]))
            raw_med.append(np.median(all_raw[m]))
            hd_med.append(np.median(all_hd[m]))
        if centers:
            ax.plot(centers, raw_med, 'o-', color='tab:blue', label='Raw median', ms=5)
            ax.plot(centers, hd_med, 's-', color='tab:red', label='Head median', ms=5)
            ax.set_xscale('log')
            ax.axhline(20, color='gray', ls=':', alpha=0.5, label='20 mas target')
            ax.set_xlabel('Band peak SNR')
            ax.set_ylabel('Median radial error (mas)')
            ax.set_title('Binned median error vs SNR')
            ax.legend(fontsize=8)

    fig.suptitle(f'9-band → VIS alignment  |  {len(all_raw)} source×band measurements', fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / 'cross_instrument_eval.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Figure saved to {out_dir / "cross_instrument_eval.png"}')


def main():
    p = argparse.ArgumentParser(description='Evaluate latent position head: align all 9 bands to VIS.')
    p.add_argument('--rubin-dir', type=str, required=True)
    p.add_argument('--euclid-dir', type=str, required=True)
    p.add_argument('--v7-checkpoint', type=str, required=True)
    p.add_argument('--head-checkpoint', type=str, required=True)
    p.add_argument('--output-dir', type=str,
                   default='models/checkpoints/latent_position_head/eval_cross_instrument')
    p.add_argument('--min-matches', type=int, default=10)
    p.add_argument('--clip-mas', type=float, default=200.0,
                   help='Reject sources with raw offset > this many mas before '
                        'running the head (bad centroids / wrong matches). '
                        '0 = no clipping. Default 200.')
    p.add_argument('--min-snr', type=float, default=5.0,
                   help='Minimum band-centroid peak SNR to include a source. '
                        'Low-SNR sources have unreliable centroids. Default 5.')
    p.add_argument('--device', type=str, default='')
    args = p.parse_args()
    evaluate(args)


if __name__ == '__main__':
    main()
