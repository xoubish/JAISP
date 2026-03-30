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

from psf_net import PSFNet, BAND_ORDER


# ---------------------------------------------------------------------------
# Star extraction  — tile-first, fully batched (no per-star Python loop)
# ---------------------------------------------------------------------------

def _load_tile_tensor(path: Path, key_img: str = 'img', key_var: str = 'var'):
    """Load a tile npz and return (img [C,H,W], rms [C,H,W]) float32 tensors."""
    data = np.load(path, allow_pickle=True, mmap_mode='r')
    img = np.asarray(data[key_img], dtype=np.float32)
    if key_var in data:
        var = np.asarray(data[key_var], dtype=np.float32)
        rms = np.sqrt(np.clip(var, 1e-20, None))
    else:
        # Fall back: MAD estimate per band
        rms = np.zeros_like(img)
        for bi in range(img.shape[0]):
            flat = img[bi].ravel()
            flat = flat[np.isfinite(flat)]
            mad = np.median(np.abs(flat - np.median(flat))) if len(flat) > 10 else 1.0
            rms[bi] = max(1.4826 * mad, 1e-10)
    return torch.from_numpy(img), torch.from_numpy(rms)


def _detect_star_positions(
    rubin_img: np.ndarray,   # [6, H, W]
    stamp_size: int,
    n_stars: int,
    nsig: float = 10.0,
):
    """
    Find compact high-S/N sources and return pixel (x, y) arrays.
    Returns (xs, ys) int arrays, already margin-clipped.
    """
    from astrometry2.source_matching import detect_sources, build_detection_image

    H, W = rubin_img.shape[1], rubin_img.shape[2]
    det = build_detection_image(rubin_img, ['rubin_g', 'rubin_r', 'rubin_i'])
    xs, ys = detect_sources(det, nsig=nsig, max_sources=n_stars * 4)
    if xs.size == 0:
        return xs, ys
    margin = stamp_size // 2 + 2
    good = (xs >= margin) & (xs < W - margin) & (ys >= margin) & (ys < H - margin)
    return xs[good], ys[good]


def _find_stars_in_tile(
    tile_info: dict,
    n_stars: int = 64,
    stamp_size: int = 21,
    device: torch.device = torch.device('cpu'),
):
    """
    Load tile once → detect stars → batch-extract all stamps in one GPU call.

    Returns (stamps [N,B,S,S], rms_stamps [N,B,S,S], x_norm [N], y_norm [N])
    or None if no usable stars found.

    Speed: one disk read + one F.grid_sample call per tile  (vs. N WCS+reproject
    calls in the old per-star loop — ~100× faster for 64 stars/tile).
    """
    from stamp_extractor import extract_stamps

    # --- Load Rubin tile ---
    rubin_img_t, rubin_rms_t = _load_tile_tensor(tile_info['rubin_path'])
    rubin_np = rubin_img_t.numpy()           # [6, H, W]  for detection (CPU)
    _, H, W = rubin_img_t.shape

    # --- Detect compact sources ---
    xs, ys = _detect_star_positions(rubin_np, stamp_size, n_stars)
    if xs.size == 0:
        return None

    # --- Load Euclid tile if available ---
    euclid_path = tile_info.get('euclid_path')
    if euclid_path and Path(euclid_path).exists():
        euclid_img_t, euclid_rms_t = _load_tile_tensor(euclid_path)
        # Euclid is at 0.1"/px (2× Rubin): downsample positions by 0.5
        # (Rubin px → Euclid px assuming same field centre, scale factor 2)
        # A better approach would use WCS; this is a fast approximation good
        # enough for PSF training on isolated stars.
        ex = xs * 0.5
        ey = ys * 0.5
        eH, eW = euclid_img_t.shape[1], euclid_img_t.shape[2]
        em = stamp_size // 2 + 2
        e_good = (ex >= em) & (ex < eW - em) & (ey >= em) & (ey < eH - em)
        xs, ys = xs[e_good], ys[e_good]
        ex, ey = ex[e_good], ey[e_good]
        has_euclid = True
    else:
        has_euclid = False

    if xs.size == 0:
        return None

    # Cap to n_stars
    xs, ys = xs[:n_stars], ys[:n_stars]

    # --- Batch stamp extraction (one GPU call each) ---
    positions_rubin = torch.from_numpy(
        np.stack([xs, ys], axis=1).astype(np.float32)
    )  # [N, 2]

    rubin_stamps = extract_stamps(
        rubin_img_t.to(device), positions_rubin, stamp_size
    )  # [N, 6, S, S]
    rubin_rms_s  = extract_stamps(
        rubin_rms_t.to(device), positions_rubin, stamp_size
    )  # [N, 6, S, S]

    if has_euclid:
        ex = ex[:n_stars]
        ey = ey[:n_stars]
        positions_euclid = torch.from_numpy(
            np.stack([ex, ey], axis=1).astype(np.float32)
        )
        euclid_stamps = extract_stamps(
            euclid_img_t.to(device), positions_euclid, stamp_size
        )  # [N, 4, S, S]
        euclid_rms_s  = extract_stamps(
            euclid_rms_t.to(device), positions_euclid, stamp_size
        )
        stamps     = torch.cat([rubin_stamps, euclid_stamps], dim=1)  # [N, 10, S, S]
        rms_stamps = torch.cat([rubin_rms_s,  euclid_rms_s],  dim=1)
    else:
        stamps     = rubin_stamps   # [N, 6, S, S] — only Rubin bands
        rms_stamps = rubin_rms_s

    # --- Compactness filter: keep sources where VIS (or r-band) peak > 3% of sum ---
    vis_idx = 6 if has_euclid else 2   # euclid_VIS or rubin_r
    S = stamp_size
    peak = stamps[:, vis_idx, S//2, S//2]                       # [N]
    total = stamps[:, vis_idx].abs().sum(dim=(-2, -1)).clamp(min=1e-6)
    compact = (peak / total) > 0.03
    if compact.sum() == 0:
        return None
    stamps     = stamps[compact]
    rms_stamps = rms_stamps[compact]
    xs_f = xs[compact.cpu().numpy().astype(bool)]
    ys_f = ys[compact.cpu().numpy().astype(bool)]

    x_norm = torch.from_numpy(xs_f.astype(np.float32) / max(1, W - 1)).to(device)
    y_norm = torch.from_numpy(ys_f.astype(np.float32) / max(1, H - 1)).to(device)

    return stamps, rms_stamps, x_norm, y_norm


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

    # Build tile index directly from data directories
    rubin_dir  = Path(args.rubin_dir)
    euclid_dir = Path(args.euclid_dir)
    euclid_map = {p.stem.replace('_euclid', ''): p
                  for p in euclid_dir.glob('tile_x*_y*_euclid.npz')}
    tiles = []
    for rp in sorted(rubin_dir.glob('tile_x*_y*.npz')):
        stem = rp.stem
        ep   = euclid_map.get(stem)
        tiles.append({'rubin_path': rp, 'euclid_path': ep})
    print(f'Found {len(tiles)} tiles')

    psf_net = PSFNet(
        n_bands=10,
        stamp_size=args.stamp_size,
        hidden_dim=64,
        band_embed_dim=8,
    ).to(device)

    optimizer = optim.Adam(psf_net.parameters(), lr=3e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_loss = float('inf')
    global_step = 0

    for epoch in range(args.epochs):
        epoch_losses = []
        np.random.shuffle(tiles)

        for tile_info in tiles:
            batch = _find_stars_in_tile(
                tile_info,
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
