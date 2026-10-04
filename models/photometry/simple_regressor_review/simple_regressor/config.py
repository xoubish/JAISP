"""Configuration for the simple VIS flux regressor.

All knobs live here with defaults; a JSON file can override any of them. Paths are
resolved relative to the JSON file's directory (or CWD when passed directly).
"""
from dataclasses import dataclass, asdict, field
import json
from pathlib import Path


@dataclass
class Config:
    # ---- data sources ----
    # Directory searched recursively for tiles matching tile_glob (tiles_product/).
    tiles_root: str = "../../tiles_product"
    n_tiles: int = 0            # 0 = use all tiles; else take a spread-out subset
    # MER photometry FITS catalogue with VIS flux/error/mag columns.
    catalog: str = "../photometry_head/assets/catalogs/mer_FINAL_q1_ECDFS_photometry.fits"
    output_dir: str = "runs/poc"
    # ---- tile / catalogue format (change these to port to other data) ----
    tile_glob: str = "*_euclid.npz"   # filename pattern searched recursively under tiles_root
    img_key: str = "img_VIS"          # npz key: image (any linear flux unit; net learns the scale)
    var_key: str = "var_VIS"          # npz key: required variance map
    wcs_key: str = "wcs_VIS"          # npz key: WCS as FITS header string or dict
    img_plane: int = -1               # -1: img/var are 2-D; >=0: index into a [band,H,W] cube
    catalog_hdu: int = 1              # FITS HDU holding the catalogue table
    flux_zeropoint: float = 23.9      # AB mag = ZP - 2.5 log10(flux); 23.9 for microJy

    # ---- source selection ----
    mag_column: str = "mag_detection_total"
    flux_column: str = "flux_detection_total"        # microJy
    fluxerr_column: str = "fluxerr_detection_total"  # microJy
    # Selection (SNR cut) always uses the detection/Kron flux, so switching the
    # training label (flux_column) keeps the same object population.
    sel_flux_column: str = "flux_detection_total"
    sel_fluxerr_column: str = "fluxerr_detection_total"
    ra_column: str = "ra"
    dec_column: str = "dec"
    id_column: str = "object_id"
    mag_min: float = -99.0      # keep sources with mag_min < mag < mag_max
    mag_max: float = 25.5       # "bright" cut for the PoC
    snr_min: float = 0.0        # also require flux/fluxerr > snr_min (0 = off); cleaner labels
    require_vis_det: bool = True
    require_spurious_zero: bool = True
    det_quality_mask: int = 0   # e.g. 1420 to match the other heads; 0 = off (simple)

    # ---- stamp geometry ----
    stamp: int = 128            # VIS pixels (0.1"/px -> 12.8")
    edge_margin: int = 0        # extra px required beyond the stamp half-width
    valid_frac_min: float = 0.5  # reject a stamp with fewer finite pixels than this
    # Empty-centre cut (image-only, does not look at the label): S/N of the flux in a small
    # central aperture (local annulus sky). Removes catalogue entries with no source at the
    # stamp centre in our image. 0 = off.
    centre_snr_min: float = 0.0
    centre_aper_arcsec: float = 0.5
    centre_filter: str = "train"  # apply the cut to "train" only, or to "all" splits

    # ---- spatial split ----
    # "patch": hold out whole LSST patches for val/test (clean, balanced when RA is
    #          clustered by patch). "ra": RA stripes with a dropped guard band.
    split_by: str = "patch"
    val_patches: list = field(default_factory=list)   # e.g. ["patch_24"]; auto if empty
    test_patches: list = field(default_factory=list)  # e.g. ["patch_25"]; auto if empty
    train_frac: float = 0.70    # used by split_by="ra"
    val_frac: float = 0.15
    buffer_arcsec: float = 30.0
    seed: int = 12

    # ---- model input / preprocessing ----
    bin_factor: int = 2         # bin the 128px stamp by this (2 -> 64px @ 0.2"/px)
    native_pixscale: float = 0.1
    # Encoder channels: [asinh(image/input_scale), asinh(SNR), centre-prior].
    # input_scale is fitted on the training split and stored in the checkpoint.
    input_scale: float = 0.0
    centre_sigma_frac: float = 0.10   # centre-prior Gaussian sigma, in units of stamp size

    # ---- model ----
    model_version: int = 2  # 1: original CNN; 2: aperture baseline + CNN correction
    aperture_radii_arcsec: list = field(default_factory=lambda: [0.5, 1.0, 1.5])
    grad_clip: float = 5.0
    torch_threads: int = 4
    # Global pooling that turns the conv feature map into a vector for the MLP head:
    # Sum and average differ only by a constant at fixed input size.
    # gatedsum adds a learned spatial weighting.
    # Pooling. avg/max/avgmax/sum/gatedsum, or "mix" = concat(avg,max,sum,gatedsum) so
    # the head can weight whichever helps. (avgmax won an earlier study; see RESULTS.md.)
    pool: str = "avgmax"
    head_hidden: int = 128      # MLP head width (raise for a flexible/high-capacity test)
    dropout: float = 0.1

    # ---- uncertainty-aware loss (matches the existing heads' convention) ----
    f0_ujy: float = 1.0
    absolute_sigma_floor_ujy: float = 0.01
    fractional_sigma_floor: float = 0.02
    huber_delta: float = 1.0
    # ---- loss space ----
    # "mag" : Huber on Delta-mag (target = standardised ln F; pred flux always > 0)
    # "flux": Huber on (pred - MER flux)/sigma_eff (target = standardised asinh F) -- previous
    loss: str = "mag"
    mag_weighted: bool = False     # mag loss: divide by MER sigma_mag (+ floor); False = plain
    mag_sigma_floor: float = 0.02 # mag; floor added in quadrature to MER sigma_mag
    mag_huber_delta: float = 0.2  # mag; Huber transition when mag_weighted=False
    mag_loss_shape: str = "huber" # "huber" (linear tails, robust) or "power" (|r|^p, aggressive)
    mag_loss_power: float = 2.0   # p for shape="power": 2 = least squares, >2 more aggressive

    # ---- training ----
    epochs: int = 80
    batch_size: int = 128
    lr: float = 5e-4
    weight_decay: float = 1e-4
    patience: int = 12
    augment: bool = True        # random dihedral (8) flips/rotations, flux-preserving
    balanced_sampling: bool = True   # sqrt magnitude-balanced training sampler
    num_workers: int = 0
    device: str = "auto"        # auto | cpu | cuda | mps
    max_sources: int = 0        # 0 = no cap

    def resolve(self, base: Path):
        """Resolve relative paths against ``base`` (the config file's directory)."""
        for k in ("tiles_root", "catalog", "output_dir"):
            p = Path(getattr(self, k))
            if not p.is_absolute():
                setattr(self, k, str((base / p).resolve()))
        return self

    def to_json(self):
        return asdict(self)


def load_config(path=None, **overrides) -> Config:
    cfg = Config()
    base = Path.cwd()
    if path is not None:
        path = Path(path)
        base = path.parent
        data = json.loads(path.read_text())
        for k, v in data.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
            else:
                raise KeyError(f"Unknown config key: {k}")
    for k, v in overrides.items():
        if v is not None and hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg.resolve(base)
