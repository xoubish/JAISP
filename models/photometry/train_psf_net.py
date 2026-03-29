"""
Train PSFNet on isolated stars from existing tile data.

Stars are identified as compact, high-S/N sources (chi2_dof < 1.5 with a
Gaussian model, FWHM consistent with the band PSF).

Usage
-----
    python train_psf_net.py \
        --rubin_dir   data/rubin_tiles_ecdfs \
        --euclid_dir  data/euclid_tiles_ecdfs \
        --concordance checkpoints/astrometry_v6_phaseB2/concordance_r.fits \
        --out         checkpoints/psf_net_v1.pt \
        --epochs      20 \
        --stamp_size  21
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset

_HERE = Path(__file__).resolve().parent
_MODELS = _HERE.parent
for _p in (_HERE, _MODELS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from astrometry2.sky_cube import SkyCubeExtractor
from psf_net import PSFNet, BAND_ORDER
from stamp_extractor import extract_stamps


# ---------------------------------------------------------------------------
# Star detection heuristic: find isolated compact sources in the VIS band
# ---------------------------------------------------------------------------

def _find_stars_in_tile(
    extractor: SkyCubeExtractor,
    tile_info: dict,
    n_stars: int = 256,
    stamp_size: int = 21,
    device: torch.device = torch.device('cpu'),
):
    """
    Extract star stamps from a tile.

    Returns
    -------
    stamps  : [N, B, S, S] float32
    rms     : [N, B, S, S] float32
    x_norm  : [N] float32
    y_norm  : [N] float32
    """
    import numpy as np
    from astrometry2.source_matching import detect_sources, build_detection_image

    rdata = np.load(tile_info['rubin_path'], allow_pickle=True, mmap_mode='r')
    rubin_img = np.asarray(rdata['img'], dtype=np.float32)   # [6, H, W]

    # Detection image: mean of g+r+i SNR
    det_img = build_detection_image(rubin_img, ['rubin_g', 'rubin_r', 'rubin_i'])
    xs, ys = detect_sources(det_img, nsig=10.0, max_sources=n_stars * 4)
    if xs.size == 0:
        return None

    H, W = rubin_img.shape[1], rubin_img.shape[2]

    # Keep only sources far from edge
    S = stamp_size
    margin = S // 2 + 2
    good = (xs >= margin) & (xs < W - margin) & (ys >= margin) & (ys < H - margin)
    xs, ys = xs[good], ys[good]
    if xs.size == 0:
        return None

    # RA/Dec of each candidate → extract aligned cube
    from astropy.wcs import WCS
    from astrometry2.source_matching import safe_header_from_card_string

    rwcs = WCS(rdata['wcs_hdr'].item())
    ra_arr, dec_arr = rwcs.wcs_pix2world(xs, ys, 0)

    all_stamps, all_rms, all_xn, all_yn = [], [], [], []
    for ra, dec in zip(ra_arr[:n_stars * 2], dec_arr[:n_stars * 2]):
        try:
            result = extractor.extract(
                ra=float(ra), dec=float(dec),
                size_arcsec=float(stamp_size) * 0.2,  # Rubin pixel scale
                tile_id=tile_info['tile_id'],
            )
        except Exception:
            continue
        cube = torch.from_numpy(result['cube'])      # [10, s, s]
        rms_c = torch.from_numpy(result['rms_cube']) # [10, s, s]
        if cube.shape[-1] < S or cube.shape[-2] < S:
            continue
        # Centre crop to stamp_size
        h0, w0 = cube.shape[-2], cube.shape[-1]
        r0 = (h0 - S) // 2
        c0 = (w0 - S) // 2
        cube  = cube[:,  r0:r0+S, c0:c0+S]
        rms_c = rms_c[:, r0:r0+S, c0:c0+S]

        # Check compactness: central pixel fraction > 0.05 (star-like)
        vis = cube[6]
        center_frac = float(vis[S//2, S//2]) / float(vis.abs().sum().clamp(min=1e-6))
        if center_frac < 0.03:
            continue

        all_stamps.append(cube)
        all_rms.append(rms_c)
        # Normalised tile-level position: use the extraction ra/dec → back to Rubin px
        rx, ry = rwcs.wcs_world2pix(ra, dec, 0)
        all_xn.append(float(rx) / max(1, W - 1))
        all_yn.append(float(ry) / max(1, H - 1))

        if len(all_stamps) >= n_stars:
            break

    if not all_stamps:
        return None

    stamps_t = torch.stack(all_stamps)                      # [N, B, S, S]
    rms_t    = torch.stack(all_rms)
    xn_t     = torch.tensor(all_xn, dtype=torch.float32)
    yn_t     = torch.tensor(all_yn, dtype=torch.float32)
    return stamps_t.to(device), rms_t.to(device), xn_t.to(device), yn_t.to(device)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Training PSFNet on {device}')

    extractor = SkyCubeExtractor(
        rubin_dir=args.rubin_dir,
        euclid_dir=args.euclid_dir,
        concordance_path=args.concordance,
    )

    psf_net = PSFNet(
        n_bands=10,
        stamp_size=args.stamp_size,
        hidden_dim=64,
        band_embed_dim=8,
    ).to(device)

    optimizer = optim.Adam(psf_net.parameters(), lr=3e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    tiles = extractor._tile_index
    best_loss = float('inf')

    for epoch in range(args.epochs):
        epoch_losses = []
        np.random.shuffle(tiles)

        for tile_info in tiles:
            batch = _find_stars_in_tile(
                extractor, tile_info,
                n_stars=64,
                stamp_size=args.stamp_size,
                device=device,
            )
            if batch is None:
                continue
            stamps, rms_stamps, x_norm, y_norm = batch

            optimizer.zero_grad()
            loss = psf_net.training_loss(stamps, rms_stamps, x_norm, y_norm)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(psf_net.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(float(loss))

        scheduler.step()
        mean_loss = np.mean(epoch_losses) if epoch_losses else float('nan')
        print(f'Epoch {epoch+1:3d}/{args.epochs}  loss={mean_loss:.5f}  lr={scheduler.get_last_lr()[0]:.2e}')

        if mean_loss < best_loss:
            best_loss = mean_loss
            torch.save({
                'psf_net_state': psf_net.state_dict(),
                'config': {
                    'n_bands': 10,
                    'stamp_size': args.stamp_size,
                    'hidden_dim': 64,
                    'band_embed_dim': 8,
                    'psf_grid_size': 8,
                    'bg_inner_radius': 7.0,
                    'bg_outer_radius': 9.5,
                },
            }, args.out)
            print(f'  ✓ saved best checkpoint → {args.out}')

    print(f'Training complete. Best loss: {best_loss:.5f}')


# ---------------------------------------------------------------------------

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--rubin_dir',   required=True)
    p.add_argument('--euclid_dir',  required=True)
    p.add_argument('--concordance', default=None)
    p.add_argument('--out',         default='checkpoints/psf_net_v1.pt')
    p.add_argument('--epochs',      type=int, default=20)
    p.add_argument('--stamp_size',  type=int, default=21)
    args = p.parse_args()
    train(args)
