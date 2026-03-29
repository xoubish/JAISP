"""
Train PSFNet on isolated stars from existing tile data.

Stars are identified as compact, high-S/N sources (high central pixel fraction).

Usage
-----
    python train_psf_net.py \
        --rubin_dir   data/rubin_tiles_ecdfs \
        --euclid_dir  data/euclid_tiles_ecdfs \
        --concordance checkpoints/astrometry_v6_phaseB2/concordance_r.fits \
        --out         checkpoints/psf_net_v1.pt \
        --epochs      20 \
        --stamp_size  21 \
        --wandb_project jaisp-photometry
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

_HERE = Path(__file__).resolve().parent
_MODELS = _HERE.parent
for _p in (_HERE, _MODELS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from astrometry2.sky_cube import SkyCubeExtractor
from psf_net import PSFNet, BAND_ORDER


# ---------------------------------------------------------------------------
# Star extraction
# ---------------------------------------------------------------------------

def _find_stars_in_tile(
    extractor: SkyCubeExtractor,
    tile_info: dict,
    n_stars: int = 64,
    stamp_size: int = 21,
    device: torch.device = torch.device('cpu'),
):
    """
    Extract compact-source stamps from a tile.

    Returns (stamps, rms_stamps, x_norm, y_norm) all float32 tensors, or None.
    """
    from astrometry2.source_matching import detect_sources, build_detection_image
    from astropy.wcs import WCS

    rdata = np.load(tile_info['rubin_path'], allow_pickle=True, mmap_mode='r')
    rubin_img = np.asarray(rdata['img'], dtype=np.float32)   # [6, H, W]
    H, W = rubin_img.shape[1], rubin_img.shape[2]

    det_img = build_detection_image(rubin_img, ['rubin_g', 'rubin_r', 'rubin_i'])
    xs, ys = detect_sources(det_img, nsig=10.0, max_sources=n_stars * 4)
    if xs.size == 0:
        return None

    S = stamp_size
    margin = S // 2 + 2
    good = (xs >= margin) & (xs < W - margin) & (ys >= margin) & (ys < H - margin)
    xs, ys = xs[good], ys[good]
    if xs.size == 0:
        return None

    rwcs = WCS(rdata['wcs_hdr'].item())
    ra_arr, dec_arr = rwcs.wcs_pix2world(xs, ys, 0)

    all_stamps, all_rms, all_xn, all_yn = [], [], [], []
    for ra, dec in zip(ra_arr[:n_stars * 2], dec_arr[:n_stars * 2]):
        try:
            result = extractor.extract(
                ra=float(ra), dec=float(dec),
                size_arcsec=float(S) * 0.2,
                tile_id=tile_info['tile_id'],
            )
        except Exception:
            continue
        cube  = torch.from_numpy(result['cube'])       # [10, h, w]
        rms_c = torch.from_numpy(result['rms_cube'])
        if cube.shape[-1] < S or cube.shape[-2] < S:
            continue
        h0, w0 = cube.shape[-2], cube.shape[-1]
        r0 = (h0 - S) // 2
        c0 = (w0 - S) // 2
        cube  = cube[:,  r0:r0+S, c0:c0+S]
        rms_c = rms_c[:, r0:r0+S, c0:c0+S]

        vis = cube[6]
        center_frac = float(vis[S//2, S//2]) / float(vis.abs().sum().clamp(min=1e-6))
        if center_frac < 0.03:
            continue

        all_stamps.append(cube)
        all_rms.append(rms_c)
        rx, ry = rwcs.wcs_world2pix(ra, dec, 0)
        all_xn.append(float(rx) / max(1, W - 1))
        all_yn.append(float(ry) / max(1, H - 1))

        if len(all_stamps) >= n_stars:
            break

    if not all_stamps:
        return None

    return (
        torch.stack(all_stamps).to(device),
        torch.stack(all_rms).to(device),
        torch.tensor(all_xn, dtype=torch.float32, device=device),
        torch.tensor(all_yn, dtype=torch.float32, device=device),
    )


# ---------------------------------------------------------------------------
# W&B PSF visualisation
# ---------------------------------------------------------------------------

def _psf_grid_images(psf_net: PSFNet, device: torch.device):
    """
    Render PSFs sampled at a 3×3 spatial grid for every band.

    Returns a dict  band_name → wandb.Image  showing 9 PSF stamps arranged
    in a 3×3 spatial layout (top-left = tile corner, centre = tile centre).
    """
    import wandb
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    psf_net.eval()
    with torch.no_grad():
        # 3×3 spatial positions covering the tile
        coords = torch.linspace(0.05, 0.95, 3, device=device)
        gy, gx = torch.meshgrid(coords, coords, indexing='ij')  # [3,3]
        xn = gx.reshape(-1)   # [9]
        yn = gy.reshape(-1)

        images = {}
        for bi, band in enumerate(BAND_ORDER[:psf_net.n_bands]):
            band_idx = torch.full((9,), bi, dtype=torch.long, device=device)
            psfs = psf_net(xn, yn, band_idx).cpu().numpy()  # [9, S, S]

            fig, axes = plt.subplots(3, 3, figsize=(5, 5))
            fig.suptitle(band, fontsize=9)
            vmax = psfs.max()
            for ax, psf_stamp, xi, yi in zip(axes.flat, psfs, xn.cpu(), yn.cpu()):
                ax.imshow(psf_stamp, origin='lower', cmap='inferno',
                          vmin=0, vmax=vmax, interpolation='nearest')
                ax.set_title(f'({xi:.2f},{yi:.2f})', fontsize=6)
                ax.axis('off')
            plt.tight_layout()
            images[f'psf/{band}'] = wandb.Image(fig)
            plt.close(fig)

    psf_net.train()
    return images


def _psf_radial_profiles(psf_net: PSFNet, device: torch.device):
    """
    Log radial profiles at tile centre for all bands as a single overlay plot.
    """
    import wandb
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    psf_net.eval()
    S = psf_net.stamp_size
    half = (S - 1) / 2.0
    y, x = np.mgrid[:S, :S]
    r = np.sqrt((x - half) ** 2 + (y - half) ** 2).ravel()
    order = np.argsort(r)
    r_sorted = r[order]

    fig, ax = plt.subplots(figsize=(6, 4))
    with torch.no_grad():
        xn = torch.tensor([0.5], device=device)
        yn = torch.tensor([0.5], device=device)
        for bi, band in enumerate(BAND_ORDER[:psf_net.n_bands]):
            band_idx = torch.tensor([bi], dtype=torch.long, device=device)
            psf = psf_net(xn, yn, band_idx).cpu().numpy()[0]  # [S, S]
            profile = psf.ravel()[order]
            ax.plot(r_sorted, profile / profile.max(), label=band, lw=1.2)

    ax.set_xlabel('Radius (px)')
    ax.set_ylabel('Normalised PSF')
    ax.legend(fontsize=6, ncol=2)
    ax.set_title('PSF radial profiles — tile centre')
    plt.tight_layout()
    img = wandb.Image(fig)
    plt.close(fig)
    psf_net.train()
    return img


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Training PSFNet on {device}')

    use_wandb = args.wandb_project is not None
    if use_wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run,
            config=vars(args),
        )

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

    tiles = list(extractor._tile_index)
    best_loss = float('inf')
    global_step = 0

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

            loss_val = float(loss)
            epoch_losses.append(loss_val)
            global_step += 1

            if use_wandb and global_step % args.log_every == 0:
                wandb.log({'train/loss_step': loss_val, 'step': global_step})

        scheduler.step()
        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else float('nan')
        lr_now = scheduler.get_last_lr()[0]
        print(f'Epoch {epoch+1:3d}/{args.epochs}  loss={mean_loss:.5f}  lr={lr_now:.2e}')

        if use_wandb:
            log_dict = {
                'train/loss_epoch': mean_loss,
                'train/lr': lr_now,
                'epoch': epoch + 1,
            }
            # Log PSF visualisations every N epochs (or always on first/last)
            if (epoch + 1) % args.vis_every == 0 or epoch == 0 or epoch == args.epochs - 1:
                log_dict.update(_psf_grid_images(psf_net, device))
                log_dict['psf/radial_profiles'] = _psf_radial_profiles(psf_net, device)
            wandb.log(log_dict)

        if mean_loss < best_loss:
            best_loss = mean_loss
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
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
            if use_wandb:
                wandb.run.summary['best_loss'] = best_loss

    print(f'Training complete. Best loss: {best_loss:.5f}')
    if use_wandb:
        wandb.finish()


# ---------------------------------------------------------------------------

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--rubin_dir',      required=True)
    p.add_argument('--euclid_dir',     required=True)
    p.add_argument('--concordance',    default=None)
    p.add_argument('--out',            default='checkpoints/psf_net_v1.pt')
    p.add_argument('--epochs',         type=int, default=20)
    p.add_argument('--stamp_size',     type=int, default=21)
    p.add_argument('--wandb_project',  default=None,
                   help='W&B project name; omit to disable W&B')
    p.add_argument('--wandb_run',      default=None,
                   help='W&B run name (auto-generated if omitted)')
    p.add_argument('--log_every',      type=int, default=10,
                   help='Log step-level loss every N gradient steps')
    p.add_argument('--vis_every',      type=int, default=5,
                   help='Log PSF visualisations every N epochs')
    args = p.parse_args()
    train(args)
