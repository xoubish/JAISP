"""Default constants used across the package."""
from __future__ import annotations

from pathlib import Path

MER_COLLECTION = "euclid_DpdMerBksMosaic"

UNWISE_PIXEL_SCALE = 2.75

# Default search radius for IRSA SIA tile discovery. Padded so a 50″ cutout
# still finds every overlapping MER tile (each tile is ~32′ across).
MER_TILE_HALF_DEG = 32.0 / 60.0 / 2.0  # tile half-side in degrees
SIA_SEARCH_PAD_DEG = 0.05

# AB magnitude zero point used by tractor.LinearPhotoCal.
AB_MAG_ZP = 22.5

# MER mosaic FLG plane: OR-combined per-frame VIS quality bits (McCracken
# et al. 2025, Q1 VIS processing, Appendix B.1). STARSIGNAL (18) and
# OBJECTS (24) mark detected-source footprints, not defects.
MER_VIS_FLAG_BITS = {
    0: "INVALID", 1: "HOT", 2: "COLD", 3: "SAT", 4: "COSMIC", 5: "GHOST",
    7: "BAD_COLUMN", 8: "BAD_CLUSTER", 9: "CR_REGION", 12: "OVRCOL",
    15: "CHARINJ", 17: "SATXTALKGHOST", 18: "STARSIGNAL", 21: "ADCMAX",
    22: "NO_DATA", 24: "OBJECTS",
}

# Coadd-unusable bits only. Per-frame defect bits (HOT, COSMIC, ...) are OR'd
# from any input frame, but the coadd used the clean frames there.
MER_VIS_BAD_BITS = (1 << 0) | (1 << 3) | (1 << 22)  # INVALID | SAT | NO_DATA

MER_NISP_BAD_BITS = (1 << 0) | (1 << 10)


def mer_bad_bits_for_band(band: str) -> int:
    """Default MER pixel veto using the originating instrument's bit map."""
    name = str(band).upper().removeprefix("NIR_")
    return MER_NISP_BAD_BITS if name in ("Y", "J", "H") else MER_VIS_BAD_BITS

# STARSIGNAL: bright-star footprints (halos, diffraction spikes). Default
# bright-star pixel mask; the masked stars themselves are not measured.
MER_VIS_STARSIGNAL = 1 << 18

# R_lambda = A_lambda / E(B-V). Euclid: Gordon et al. (2023) MW curve,
# A_lambda/A_V = 0.678/0.366/0.261/0.160 (Hunt et al. 2025, A&A 697, A9)
EXTINCTION_COEFF = {
    "VIS": 3.1 * 0.678,   # 2.102
    "Y":   3.1 * 0.366,   # 1.135
    "J":   3.1 * 0.261,   # 0.809
    "H":   3.1 * 0.160,   # 0.496
    "W1":  0.19,          # Yuan et al. 2013, MNRAS 430, 2188
    "W2":  0.15,
}

WISE_COADD_VERSION = "neo7"

DEFAULT_DATA_DIR = Path("examples/data")
DEFAULT_CUTOUT_DIR = DEFAULT_DATA_DIR / "cutouts"
DEFAULT_PSF_DIR = DEFAULT_DATA_DIR / "psf"
DEFAULT_WISE_CACHE_DIR = DEFAULT_DATA_DIR / "wise"

