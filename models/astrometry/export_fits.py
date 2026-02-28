"""
Export trained astrometry concordance predictions as FITS files.

Follows the data product specification:
  - Per VIS quadrant: HDUs for DRA (ΔRA*) and DDE (ΔDec)
  - Coarse mesh sampled every DSTEP VIS pixels
  - Standard keywords: DSTEP, DUNIT, INTERP, CONCRDNC
  - WCS inherited from VIS extension

Usage:
    python export_fits.py \
        --backbone-ckpt ../checkpoints/jaisp_v5/best.pt \
        --head-ckpt ../checkpoints/jaisp_astrometry/best_astrometry.pt \
        --rubin-dir ../data/rubin_tiles_ecdfs \
        --euclid-dir ../data/euclid_tiles_ecdfs \
        --output concordance_ecdfs.fits \
        --dstep 8
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

try:
    from astropy.io import fits
    HAS_ASTROPY = True
except ImportError:
    HAS_ASTROPY = False
    print("WARNING: astropy not installed — FITS export disabled. Install with: pip install astropy")

import sys
SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_DIR = SCRIPT_DIR.parent
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from jaisp_dataset_v4 import ALL_BANDS, _to_float32, _safe_sqrt_var, RUBIN_BAND_ORDER
from jaisp_foundation_v5 import JAISPFoundationV5
from head import AstrometryConcordanceHead, NonParametricConcordanceHead, PIXEL_SCALES


def load_backbone(device, args):
    model = JAISPFoundationV5(
        band_names=ALL_BANDS,
        stem_ch=64,
        embed_dim=args.embed_dim,
        proj_dim=args.proj_dim,
        depth=args.depth,
        patch_size=args.patch_size,
        shift_temp=0.07,
    ).to(device)
    if args.backbone_ckpt:
        ckpt = torch.load(args.backbone_ckpt, map_location=device)
        state = ckpt.get("model", ckpt)
        model.load_state_dict(state, strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def load_head(device, args):
    if args.head_mode == "nonparametric":
        head = NonParametricConcordanceHead(
            patch_size=args.patch_size,
            search_radius=args.search_radius,
            softmax_temp=args.softmax_temp,
            smooth_kernel=args.nonparam_smooth_kernel,
        ).to(device)
        head.eval()
        return head

    ckpt = torch.load(args.head_ckpt, map_location=device)
    head_args = ckpt.get("args", {})
    head = AstrometryConcordanceHead(
        embed_dim=head_args.get("embed_dim", args.embed_dim),
        search_radius=head_args.get("search_radius", args.search_radius),
        softmax_temp=head_args.get("softmax_temp", args.softmax_temp),
        refine_hidden=head_args.get("refine_hidden", 32),
        refine_depth=head_args.get("refine_depth", 4),
        patch_size=head_args.get("patch_size", args.patch_size),
        global_hidden=head_args.get("global_hidden", 128),
        local_hidden=head_args.get("local_hidden", 64),
        local_depth=head_args.get("local_depth", 5),
        match_dim=head_args.get("match_dim", 64),
        residual_gain_init=head_args.get("residual_gain_init", 1.0),
        use_stem_refine=head_args.get("use_stem_refine", False),
        stem_channels=head_args.get("stem_channels", 64),
        stem_hidden=head_args.get("stem_hidden", 32),
        stem_depth=head_args.get("stem_depth", 4),
        stem_stride=head_args.get("stem_stride", 4),
    ).to(device)
    missing, unexpected = head.load_state_dict(ckpt["head"], strict=False)
    if missing or unexpected:
        print(f"Head checkpoint mismatch: missing={len(missing)} unexpected={len(unexpected)}")
    head.eval()
    return head


def encode_band(backbone, image, rms, band, device):
    image = image.unsqueeze(0).to(device)
    rms = rms.unsqueeze(0).to(device)
    with torch.no_grad():
        feat = backbone.stems[band](image, rms)
        tokens, grid_size = backbone.encoder(feat)
    return feat, tokens, grid_size


@torch.no_grad()
def predict_tile(
    backbone,
    head,
    rubin_img: np.ndarray,
    rubin_rms: np.ndarray,
    vis_img: np.ndarray,
    vis_rms: np.ndarray,
    rubin_band: str,
    device: torch.device,
    dstep: int = 8,
) -> dict:
    """
    Predict concordance offset for a single tile.

    Returns dict with:
        dra:  [H_mesh, W_mesh] ΔRA* in arcseconds
        ddec: [H_mesh, W_mesh] ΔDec in arcseconds
        confidence: [H_mesh, W_mesh] peak sharpness
    """
    rubin_t = torch.from_numpy(rubin_img[None].copy()).float().to(device)
    rubin_rms_t = torch.from_numpy(rubin_rms[None].copy()).float().to(device)
    vis_t = torch.from_numpy(vis_img[None].copy()).float().to(device)
    vis_rms_t = torch.from_numpy(vis_rms[None].copy()).float().to(device)

    rubin_feat, rubin_tokens, rubin_grid = encode_band(backbone, rubin_t, rubin_rms_t, rubin_band, device)
    vis_feat, vis_tokens, vis_grid = encode_band(backbone, vis_t, vis_rms_t, "euclid_VIS", device)

    H_vis, W_vis = vis_img.shape[-2], vis_img.shape[-1]
    out = head(
        rubin_tokens=rubin_tokens,
        vis_tokens=vis_tokens,
        rubin_grid=rubin_grid,
        vis_grid=vis_grid,
        vis_image_hw=(H_vis, W_vis),
        vis_pixel_scale=PIXEL_SCALES["euclid_VIS"],
        mesh_step=dstep,
        rubin_stem=rubin_feat,
        vis_stem=vis_feat,
    )

    return {
        "dra": out["dra"][0, 0].cpu().numpy(),
        "ddec": out["ddec"][0, 0].cpu().numpy(),
        "confidence": out["confidence"][0, 0].cpu().numpy() if out["confidence"].shape[-2:] == out["dra"].shape[-2:] else
            F.interpolate(
                out["confidence"], size=out["dra"].shape[-2:], mode="bilinear", align_corners=False
            )[0, 0].cpu().numpy(),
    }


def make_concordance_hdu(
    data: np.ndarray,
    extname: str,
    dstep: int,
    rubin_band: str,
    tile_id: str,
    vis_wcs_header=None,
) -> "fits.ImageHDU":
    """Create a FITS ImageHDU for a concordance offset field."""
    hdu = fits.ImageHDU(data=data.astype(np.float32), name=extname)
    hdu.header["DSTEP"] = (dstep, "Mesh sampling step in VIS pixels")
    hdu.header["DUNIT"] = ("arcsec", "Unit of offset values")
    hdu.header["INTERP"] = ("bilinear", "Recommended interpolation method")
    hdu.header["CONCRDNC"] = (True, "This is a concordance offset extension")
    hdu.header["RBNBAND"] = (rubin_band, "Rubin band for this concordance")
    hdu.header["REFFRAME"] = ("euclid_VIS", "Reference frame")
    hdu.header["TILEID"] = (tile_id, "Source tile identifier")

    if vis_wcs_header is not None:
        for key in ["CRPIX1", "CRPIX2", "CRVAL1", "CRVAL2",
                     "CD1_1", "CD1_2", "CD2_1", "CD2_2",
                     "CTYPE1", "CTYPE2", "CUNIT1", "CUNIT2"]:
            if key in vis_wcs_header:
                val = vis_wcs_header[key]
                # Scale CRPIX for mesh step.
                if key in ("CRPIX1", "CRPIX2"):
                    val = val / dstep
                if key in ("CD1_1", "CD1_2", "CD2_1", "CD2_2"):
                    val = val * dstep
                hdu.header[key] = val

    return hdu


def export_tile(
    backbone,
    head,
    rubin_path: str,
    euclid_path: str,
    rubin_bands: list,
    device: torch.device,
    dstep: int = 8,
    tile_id: str = "",
) -> list:
    """
    Run inference on one tile for given Rubin bands, return list of HDUs.
    """
    rubin_data = np.load(rubin_path, allow_pickle=True)
    euclid_data = np.load(euclid_path, allow_pickle=True)

    if "img_VIS" not in euclid_data:
        return []

    vis_img = _to_float32(euclid_data["img_VIS"])
    vis_rms = _safe_sqrt_var(euclid_data["var_VIS"]) if "var_VIS" in euclid_data else np.ones_like(vis_img) * np.nanstd(vis_img)
    vis_img = np.nan_to_num(vis_img, nan=0.0)
    vis_rms = np.maximum(np.nan_to_num(vis_rms, nan=1.0), 1e-10)

    hdus = []
    for band in rubin_bands:
        band_key = band.split("_")[1]
        idx = RUBIN_BAND_ORDER.index(band_key)
        if idx >= rubin_data["img"].shape[0]:
            continue

        rubin_img = _to_float32(rubin_data["img"][idx])
        rubin_rms = _safe_sqrt_var(rubin_data["var"][idx]) if "var" in rubin_data else np.ones_like(rubin_img) * np.nanstd(rubin_img)
        rubin_img = np.nan_to_num(rubin_img, nan=0.0)
        rubin_rms = np.maximum(np.nan_to_num(rubin_rms, nan=1.0), 1e-10)

        frac = np.isfinite(rubin_data["img"][idx]).sum() / float(rubin_data["img"][idx].size)
        if frac < 0.3:
            continue

        result = predict_tile(
            backbone, head,
            rubin_img, rubin_rms,
            vis_img, vis_rms,
            band, device, dstep,
        )

        # Create DRA and DDE HDUs.
        ext_prefix = f"{tile_id}.{band_key}" if tile_id else band_key
        hdus.append(make_concordance_hdu(
            result["dra"], f"{ext_prefix}.DRA", dstep, band, tile_id,
        ))
        hdus.append(make_concordance_hdu(
            result["ddec"], f"{ext_prefix}.DDE", dstep, band, tile_id,
        ))

    return hdus


def main():
    parser = argparse.ArgumentParser(description="Export astrometry concordance FITS.")
    parser.add_argument("--backbone-ckpt", type=str, required=True)
    parser.add_argument("--head-mode", type=str, default="hybrid",
                        choices=["hybrid", "nonparametric"])
    parser.add_argument("--head-ckpt", type=str, default="")
    parser.add_argument("--rubin-dir", type=str, required=True)
    parser.add_argument("--euclid-dir", type=str, required=True)
    parser.add_argument("--output", type=str, default="concordance.fits")
    parser.add_argument("--dstep", type=int, default=8)
    parser.add_argument("--rubin-bands", type=str, nargs="+",
                        default=["rubin_u", "rubin_g", "rubin_r", "rubin_i", "rubin_z", "rubin_y"])
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--proj-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--search-radius", type=int, default=3)
    parser.add_argument("--softmax-temp", type=float, default=0.1)
    parser.add_argument("--nonparam-smooth-kernel", type=int, default=3)
    parser.add_argument("--use-stem-refine", action="store_true")
    parser.add_argument("--stem-channels", type=int, default=64)
    parser.add_argument("--stem-hidden", type=int, default=32)
    parser.add_argument("--stem-depth", type=int, default=4)
    parser.add_argument("--stem-stride", type=int, default=4)
    parser.add_argument("--device", type=str, default="")
    args = parser.parse_args()

    if not HAS_ASTROPY:
        print("ERROR: astropy is required for FITS export.")
        return
    if args.head_mode == "hybrid" and not args.head_ckpt:
        print("ERROR: --head-ckpt is required for head_mode=hybrid.")
        return

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    backbone = load_backbone(device, args)
    head = load_head(device, args)

    import glob, os
    rubin_files = sorted(glob.glob(os.path.join(args.rubin_dir, "tile_x*_y*.npz")))

    all_hdus = [fits.PrimaryHDU()]  # empty primary
    all_hdus[0].header["CONCRDNC"] = (True, "JAISP astrometry concordance product")
    all_hdus[0].header["DSTEP"] = (args.dstep, "Mesh sampling step in VIS pixels")
    all_hdus[0].header["DUNIT"] = ("arcsec", "Offset unit")
    all_hdus[0].header["REFFRAME"] = ("euclid_VIS", "Reference astrometric frame")
    all_hdus[0].header["INTERP"] = ("bilinear", "Recommended interpolation")

    n_tiles = 0
    for rp in rubin_files:
        tid = os.path.splitext(os.path.basename(rp))[0]
        ep = os.path.join(args.euclid_dir, f"{tid}_euclid.npz")
        if not os.path.exists(ep):
            continue

        hdus = export_tile(backbone, head, rp, ep, args.rubin_bands, device, args.dstep, tid)
        all_hdus.extend(hdus)
        n_tiles += 1
        if n_tiles % 10 == 0:
            print(f"  processed {n_tiles} tiles, {len(all_hdus)-1} HDUs")

    hdul = fits.HDUList(all_hdus)
    hdul.writeto(args.output, overwrite=True)
    print(f"\nWrote {args.output}: {n_tiles} tiles, {len(all_hdus)-1} HDUs")


if __name__ == "__main__":
    main()
