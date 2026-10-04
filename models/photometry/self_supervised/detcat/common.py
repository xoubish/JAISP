"""Paths, constants and small geometry helpers shared by the detection-catalog pipeline."""
from pathlib import Path
import json
import re
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
import os
OUT = Path(os.environ.get('JAISP_DETCAT_OUT', ROOT / 'models/photometry/self_supervised/runs/detection_catalog'))
EUCLID = ('VIS', 'Y', 'J', 'H')
EUCLID_BANDS = tuple('euclid_' + b for b in EUCLID)
RUBIN_BANDS = tuple('rubin_' + b for b in 'ugrizy')
FOUNDATION = 'models/checkpoints/jaisp_v11_q1_soft/checkpoint_best.pt'
ASTROMETRY = 'models/checkpoints/latent_position_v11_anchored_v11labels_patchval25/best.pt'
LABELS = 'data/detection_labels/centernet_q1_790_vissep_v11_thresh03.pt'
PRIORS = 'models/photometry/self_supervised/runs/q1_mixture_calibrated/priors.pt'
PSF_CALIBRATION = 'models/photometry/self_supervised/runs/q1_all_bands/psf_calibration.json'
PRIOR_RUN = 'models/photometry/self_supervised/runs/q1_all_bands'
CACHE = 'data/cached_features_v11_q1'
# Same MER FLG conventions as the upstream package (euclid_phot.config) and real_mer_prepare.
VIS_BAD = (1 << 0) | (1 << 3) | (1 << 22)
NISP_BAD = (1 << 0) | (1 << 10)
STARSIGNAL = 1 << 18
RUBIN_BAD = 1 | 2 | 256
MATCH_RADIUS_ARCSEC = .5
SCENE_HALF_ARCSEC = 6.
NEIGHBOR_ARCSEC = 14.
MAX_SCENE_SOURCES = 40
CUTOUT_SIZES_ARCSEC = (116., 108.4)
# Photometry-prior partition boundaries from run.py; patch 25 is the detection/astrometry holdout.
RA_B1, RA_B2, RA_GUARD = 53.166483388, 53.220111106, .009453496771566816


def region_dir(i):
    return OUT / f'region_{i:03d}'


def tile_xy(name):
    x = int(re.search(r'x(\d+)', name)[1]); y = int(re.search(r'y(\d+)', name)[1])
    return x, y


def tile_patch(name):
    return name.split('patch_')[1]


def tangent_arcsec(sky, center):
    """East/north tangent-plane offsets in arcsec of sky[N,2] from center[2] (degrees)."""
    sky = np.atleast_2d(np.asarray(sky, float)); center = np.asarray(center, float)
    dra = (sky[:, 0] - center[0] + 180) % 360 - 180
    return np.column_stack((dra * np.cos(np.deg2rad(center[1])) * 3600, (sky[:, 1] - center[1]) * 3600))


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, default=_json_default))


def _json_default(value):
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating,)): return float(value)
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, Path): return str(value)
    raise TypeError(type(value))
