"""Precompute frozen V7 encoder bottleneck features for all tiles.

Runs the full 10-band V7 encoder once per tile per augmentation variant
and saves the bottleneck tensor [C, H, W] to disk. CenterNet training
then loads these cached features directly — no encoder forward pass needed.

Usage
-----
    python detection/precompute_features.py \
        --rubin_dir    ../data/rubin_tiles_all \
        --euclid_dir   ../data/euclid_tiles_all \
        --encoder_ckpt ../checkpoints/jaisp_v7_tiles_all_ddp_online/checkpoint_best.pt \
        --out_dir      ../data/cached_features_v7_tiles_all \
        --n_augments   4
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_MODELS = _HERE.parent
for _p in (_HERE, _MODELS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from jaisp_foundation_v7 import JAISPFoundationV7, ALL_BANDS, RUBIN_BANDS
from jaisp_dataset_v6 import JAISPDatasetV6
from detection.detector import JAISPEncoderWrapper


def _load_encoder(encoder_ckpt, device):
    ckpt = torch.load(encoder_ckpt, map_location='cpu', weights_only=False)
    cfg = ckpt.get('config', {})
    model = JAISPFoundationV7(
        band_names               = cfg.get('band_names', ALL_BANDS),
        stem_ch                  = cfg.get('stem_ch', 64),
        hidden_ch                = cfg.get('hidden_ch', 256),
        blocks_per_stage         = cfg.get('blocks_per_stage', 2),
        transformer_depth        = cfg.get('transformer_depth', 4),
        transformer_heads        = cfg.get('transformer_heads', 8),
        fused_pixel_scale_arcsec = cfg.get('fused_pixel_scale_arcsec', 0.8),
    )
    model.load_state_dict(ckpt['model'], strict=False)
    wrapper = JAISPEncoderWrapper(model, freeze=True).to(device)
    print(f'Encoder loaded from {encoder_ckpt}')
    return wrapper


def main():
    p = argparse.ArgumentParser(description='Precompute V7 encoder features for CenterNet training.')
    p.add_argument('--rubin_dir',    required=True)
    p.add_argument('--euclid_dir',   default=None)
    p.add_argument('--encoder_ckpt', required=True)
    p.add_argument('--out_dir',      required=True,
                   help='Directory to save cached .pt feature files')
    p.add_argument('--n_augments',   type=int, default=8,
                   help='Number of augmentation variants per tile (max 8: 4 rot x 2 flip)')
    p.add_argument('--device',       default='')
    args = p.parse_args()

    device = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    encoder = _load_encoder(args.encoder_ckpt, device)

    # All 8 possible augmentation configs: (n_rot90, flip_ud, flip_lr)
    all_augs = [
        (r, fu, fl)
        for r in range(4)
        for fu in (False, True)
        for fl in (False, True)
    ][:args.n_augments]

    # Load dataset with augment=False — we'll apply augmentations manually
    ds = JAISPDatasetV6(
        rubin_dir=args.rubin_dir,
        euclid_dir=args.euclid_dir or args.rubin_dir,
        augment=False,
        load_euclid=args.euclid_dir is not None,
    )

    n_tiles = len(ds)
    print(f'Processing {n_tiles} tiles x {len(all_augs)} augments = {n_tiles * len(all_augs)} features')

    for idx in range(n_tiles):
        tile_id = ds.tiles[idx]['tile_id']
        item = ds[idx]

        for aug_idx, (n_rot, flip_ud, flip_lr) in enumerate(all_augs):
            out_path = out_dir / f'{tile_id}_aug{aug_idx}.pt'
            if out_path.exists():
                continue

            # Build band dicts with augmentation applied
            images = {}
            rms_d = {}

            for band in RUBIN_BANDS:
                if band in item['rubin']:
                    img = item['rubin'][band]['image'][0].numpy()  # [H, W]
                    rms = item['rubin'][band]['rms'][0].numpy()
                    img, rms = _apply_aug(img, rms, n_rot, flip_ud, flip_lr)
                    images[band] = torch.from_numpy(img[None, None]).to(device)
                    rms_d[band] = torch.from_numpy(rms[None, None]).to(device)

            if args.euclid_dir:
                from jaisp_foundation_v7 import ALL_BANDS
                euclid_bands = [b for b in ALL_BANDS if b.startswith('euclid')]
                for band in euclid_bands:
                    if band in item.get('euclid', {}):
                        img = item['euclid'][band]['image'][0].numpy()
                        rms = item['euclid'][band]['rms'][0].numpy()
                        img, rms = _apply_aug(img, rms, n_rot, flip_ud, flip_lr)
                        images[band] = torch.from_numpy(img[None, None]).to(device)
                        rms_d[band] = torch.from_numpy(rms[None, None]).to(device)

            with torch.no_grad():
                feats = encoder(images, rms_d)  # [1, C, H, W]

            # Save feature tensor + aug params
            torch.save({
                'features':   feats[0].cpu(),  # [C, H, W]
                'encoder_dim': int(feats.shape[1]),
                'tile_id':    tile_id,
                'aug_params': (n_rot, flip_ud, flip_lr),
                'aug_idx':    aug_idx,
            }, out_path)

        if (idx + 1) % 10 == 0 or idx == 0:
            print(f'  [{idx+1}/{n_tiles}] {tile_id}')

    print(f'Done. Features saved to {out_dir}')


def _apply_aug(img, rms, n_rot, flip_ud, flip_lr):
    """Apply augmentation to numpy arrays."""
    img = np.rot90(img, n_rot).copy()
    rms = np.rot90(rms, n_rot).copy()
    if flip_ud:
        img = np.flipud(img).copy()
        rms = np.flipud(rms).copy()
    if flip_lr:
        img = np.fliplr(img).copy()
        rms = np.fliplr(rms).copy()
    return img, rms


if __name__ == '__main__':
    main()
