"""MER catalogue loading and source selection for the VIS flux regressor.

Label  : cfg.flux_column / cfg.fluxerr_column  (e.g. flux_detection_total = Kron flux,
         or flux_vis_{1,2,3,4}fwhm_aper = fixed apertures on PSF-matched images).
Select : magnitude cut on cfg.mag_column and SNR cut on cfg.sel_flux_column /
         cfg.sel_fluxerr_column (detection/Kron by default). The same snr_min is also
         applied to the label flux/err, which drops rows with broken MER aperture errors.
The returned `mag` is derived from the *label* flux (cfg.flux_zeropoint - 2.5 log10 F), so magnitude
metrics and the magnitude-balanced sampler refer to the quantity being learned.
"""
from dataclasses import dataclass
import numpy as np
from astropy.io import fits



@dataclass
class Sources:
    object_id: np.ndarray
    ra: np.ndarray
    dec: np.ndarray
    flux: np.ndarray      # microJy, label (cfg.flux_column)
    fluxerr: np.ndarray   # microJy, label error (cfg.fluxerr_column)
    mag: np.ndarray       # AB mag of the label flux

    def __len__(self):
        return len(self.object_id)


def load_sources(cfg) -> Sources:
    """Load the catalogue, apply quality/magnitude/SNR selection, return the label."""
    with fits.open(cfg.catalog) as hdul:
        t = hdul[cfg.catalog_hdu].data
        names = {n.lower(): n for n in t.columns.names}

        def col(key):
            if key.lower() not in names:
                raise KeyError(f"Column '{key}' not in catalogue")
            return np.asarray(t[names[key.lower()]])

        oid = col(cfg.id_column).astype(np.int64)
        ra = col(cfg.ra_column).astype(np.float64)
        dec = col(cfg.dec_column).astype(np.float64)
        flux = col(cfg.flux_column).astype(np.float64)        # label
        ferr = col(cfg.fluxerr_column).astype(np.float64)
        sflux = col(cfg.sel_flux_column).astype(np.float64)   # selection (Kron)
        sferr = col(cfg.sel_fluxerr_column).astype(np.float64)
        smag = col(cfg.mag_column).astype(np.float64)

        with np.errstate(invalid="ignore", divide="ignore"):
            base = (np.isfinite(ra) & np.isfinite(dec) & np.isfinite(smag)
                    & np.isfinite(sflux) & (sflux > 0) & np.isfinite(sferr) & (sferr > 0))
            if cfg.require_vis_det and "vis_det" in names:
                base &= col("vis_det").astype(int) == 1
            if cfg.require_spurious_zero and "spurious_flag" in names:
                base &= col("spurious_flag").astype(int) == 0
            if cfg.det_quality_mask and "det_quality_flag" in names:
                base &= (col("det_quality_flag").astype(np.int64) & int(cfg.det_quality_mask)) == 0
            base &= (smag > cfg.mag_min) & (smag < cfg.mag_max)
            if cfg.snr_min > 0:
                base &= (sflux / sferr) > cfg.snr_min
            label_ok = np.isfinite(flux) & (flux > 0) & np.isfinite(ferr) & (ferr > 0)
            n_bad = int((base & ~label_ok).sum())
            # The same SNR cut also applies to the *label*. This removes MER rows whose
            # aperture fluxerr is broken (~1e2-1e4 uJy on ~1 uJy fluxes; 6-13% of the
            # fixed-aperture columns, none for Kron) as well as genuinely low-S/N labels.
            label_snr_ok = label_ok.copy()
            if cfg.snr_min > 0:
                label_snr_ok &= (flux / ferr) > cfg.snr_min
            n_lowsnr = int((base & label_ok & ~label_snr_ok).sum())
            n_broken = int((base & label_ok & ~label_snr_ok & (ferr > 10.0)).sum())
        keep = base & label_snr_ok
        if n_bad:
            print(f"[catalog] dropped {n_bad} selected sources with non-positive/invalid "
                  f"{cfg.flux_column} or its error")
        if n_lowsnr:
            print(f"[catalog] dropped {n_lowsnr} selected sources with label SNR "
                  f"{cfg.flux_column}/{cfg.fluxerr_column} <= {cfg.snr_min} "
                  f"({n_broken} of them have a quoted error > 10 uJy)")
        mag = cfg.flux_zeropoint - 2.5 * np.log10(flux[keep])

    return Sources(oid[keep], ra[keep], dec[keep], flux[keep], ferr[keep], mag)
