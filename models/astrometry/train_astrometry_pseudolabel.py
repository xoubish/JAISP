"""
Legacy compatibility wrapper.

The older sparse-point pseudo-label trainer now lives in
`models/astrometry/older/train_astrometry_sparse_pseudolabel.py`.
For the current neural path, use `train_astrometry_multiband_teacher.py`.
"""

import os

import numpy as np
import torch

from older.train_astrometry_sparse_pseudolabel import build_parser, train


if __name__ == "__main__":
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.environ.setdefault("PYTHONHASHSEED", str(args.seed))
    train(args)
