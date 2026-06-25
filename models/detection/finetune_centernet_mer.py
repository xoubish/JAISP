"""Fine-tune the CenterNet detection head on real Euclid MER labels (frozen v10 foundation).

Warm-start from a production CenterNet head and train the head ONLY on clean MER VIS
detections (vis_det & !spurious, mag<cap) rendered as CenterNet heatmap targets. Uses a
spatially-disjoint split: hold out one DR1 MER tile entirely, train on the rest.

Example
-------
    PYTHONPATH=models python models/detection/finetune_centernet_mer.py \
        --mer-fits   data/edf_s_ood/catalogs_compact/mer_FINAL_q1_ECDFS_footprint.fits \
        --euclid-dir data/euclid_tiles_all --rubin-dir data/rubin_tiles_all \
        --hold-out-tile TILE101374533 --n-train 200 --epochs 6 \
        --out checkpoints/centernet_v10_MERfinetune/centernet_best.pt

Evaluate the result against the baseline on the held-out tile in
`io/17_detection_validation.ipynb` (injection-recovery + MER completeness/purity).
"""
import argparse, glob, os, sys
import numpy as np
import torch
import torch.optim as optim
from astropy.io import fits
from astropy.wcs import WCS

from jaisp_foundation_v10 import RUBIN_BANDS, EUCLID_BANDS
from detection.centernet_detector import CenterNetDetector
from detection.detector import JAISPEncoderWrapper
from detection.centernet_loss import CenterNetLoss
from load_foundation import load_foundation
from astrometry2.source_matching import safe_header_from_card_string


def build_mer_labels(mer_fits, euclid_dir, mag_cap):
    """Per-tile normalized-VIS-xy labels from the clean MER catalogue; plus tile->DR1-tile map."""
    cat = fits.open(mer_fits)[1].data
    keep = (np.asarray(cat['vis_det']) == 1) & (np.asarray(cat['spurious_flag']) != 1) \
        & (np.asarray(cat['mag_vis'], float) < mag_cap)
    MRA, MDEC = np.asarray(cat['ra'], float)[keep], np.asarray(cat['dec'], float)[keep]
    labels, tilemap = {}, {}
    for et in sorted(glob.glob(f'{euclid_dir}/*_euclid.npz')):
        stem = os.path.basename(et).replace('_euclid.npz', '')
        d = np.load(et, allow_pickle=True)
        H, W = np.asarray(d['img_VIS']).shape
        vw = WCS(safe_header_from_card_string(d['wcs_VIS'].item()))
        x, y = vw.all_world2pix(MRA, MDEC, 0)
        inb = (x >= 0) & (x < W) & (y >= 0) & (y < H)
        labels[stem] = np.c_[x[inb] / (W - 1), y[inb] / (H - 1)].astype(np.float32)
        tilemap[stem] = str(d['euclid_tile_id'])
    return labels, tilemap


def build_inputs(stem, euclid_dir, rubin_dir, device):
    ed = np.load(f'{euclid_dir}/{stem}_euclid.npz', allow_pickle=True)
    rd = np.load(f'{rubin_dir}/{stem}.npz', allow_pickle=True)
    rimg = np.asarray(rd['img'], np.float32); rvar = np.asarray(rd['var'], np.float32)
    im, rm = {}, {}
    for bi, b in enumerate(RUBIN_BANDS):
        im[b] = torch.from_numpy(np.nan_to_num(rimg[bi])[None, None].copy()).to(device)
        rm[b] = torch.from_numpy(np.maximum(np.sqrt(np.clip(rvar[bi], 1e-12, None)), 1e-10)[None, None].copy()).to(device)
    for b in EUCLID_BANDS:
        k = b.split('_', 1)[1]
        var = np.nan_to_num(np.asarray(ed[f'var_{k}'], np.float32), nan=1.0)
        im[b] = torch.from_numpy(np.nan_to_num(np.asarray(ed[f'img_{k}'], np.float32))[None, None].copy()).to(device)
        rm[b] = torch.from_numpy(np.maximum(np.sqrt(np.clip(var, 1e-12, None)), 1e-10)[None, None].copy()).to(device)
    return im, rm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mer-fits', required=True)
    ap.add_argument('--euclid-dir', default='data/euclid_tiles_all')
    ap.add_argument('--rubin-dir', default='data/rubin_tiles_all')
    ap.add_argument('--encoder-ckpt', default='models/checkpoints/jaisp_v10_warmstart/checkpoint_best.pt')
    ap.add_argument('--warm-start', default='checkpoints/centernet_v10_uncertain_synth_r2/centernet_best.pt')
    ap.add_argument('--hold-out-tile', default='TILE101374533')
    ap.add_argument('--mag-cap', type=float, default=26.0)
    ap.add_argument('--n-train', type=int, default=200)
    ap.add_argument('--epochs', type=int, default=6)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--sigma', type=float, default=2.0)
    ap.add_argument('--out', default='checkpoints/centernet_v10_MERfinetune/centernet_best.pt')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rng = np.random.default_rng(0)
    labels, tilemap = build_mer_labels(args.mer_fits, args.euclid_dir, args.mag_cap)
    train = [s for s, t in tilemap.items() if t != args.hold_out_tile and len(labels.get(s, [])) >= 5]
    rng.shuffle(train); train = train[:args.n_train]
    print(f'train tiles {len(train)} (excl held-out {args.hold_out_tile}); '
          f'median labels/tile {np.median([len(labels[s]) for s in train]):.0f}')

    foundation = load_foundation(args.encoder_ckpt, device=torch.device('cpu'), freeze=True)
    enc = JAISPEncoderWrapper(foundation, freeze=True).to(device).eval()
    model = CenterNetDetector.load(args.warm_start, encoder=enc, device=device)
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    print('trainable head params:', sum(p.numel() for p in params))
    opt = optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = CenterNetLoss(sigma=args.sigma)

    for ep in range(args.epochs):
        rng.shuffle(train); losses = []
        for stem in train:
            xy = np.asarray(labels[stem], np.float32)
            if len(xy) < 5:
                continue
            im, rm = build_inputs(stem, args.euclid_dir, args.rubin_dir, device)
            out = model(im, rm)
            L = crit(out, [torch.from_numpy(xy).to(device)])
            opt.zero_grad(); L['loss_total'].backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
            losses.append(float(L['loss_total']))
        sched.step()
        print(f'epoch {ep + 1}/{args.epochs}  loss {np.mean(losses):.4f}  (n={len(losses)})')

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    ck = torch.load(args.warm_start, map_location='cpu', weights_only=True)
    ck['state_dict'] = model.state_dict()
    torch.save(ck, args.out)
    print('saved fine-tuned ->', args.out)


if __name__ == '__main__':
    sys.path.insert(0, 'models'); sys.path.insert(0, 'models/astrometry2')
    main()
