"""Training script for JAISP Foundation v10 — PSF-aware loss shaping.

Inherits everything from v9 (concat fusion + adversarial cross-instrument
masking) and changes the loss function to fix the PSF-profile residuals
diagnosed during v9 training:

1. **Charbonnier base loss** instead of L1.  L1 has a constant gradient,
   so the per-pixel optimum collapses to the median of the conditional
   distribution — which for a sharp PSF is broader than the truth.  This
   systematically under-predicts source cores and over-predicts halos
   (the donut/dipole patterns visible in v9 reconstruction diagnostics).
   Charbonnier ``sqrt(diff^2 + eps^2)`` is L1-like at large residuals
   (robust against noise) but L2-like near zero (sharp peaks rewarded).

2. **High-info L2 weighting** on top.  An additional L2 loss restricted
   to ``info_w > core_info_threshold`` pixels — explicitly rewards
   getting the bright-source pixels exactly right rather than smoothed.

3. **Per-band normalised loss telemetry** (already wired in v7 trainer
   for any model that returns ``pixel_loss_norm``).  v10's model does,
   so the W&B dashboard now logs ``train/band_X_norm`` per band — the
   un-RMS-multiplied pixel loss that is fair to compare across bands.

The architecture stays identical to v9.  Only the per-pixel loss formula
in JAISPFoundationV8.forward changes (gated on the ``loss_type`` and
``core_l2_weight`` parameters introduced for v10).

Usage::

    # Single GPU
    python train_jaisp_foundation_v10.py \\
        --rubin_dir  ../data/rubin_tiles_all \\
        --euclid_dir ../data/euclid_tiles_all \\
        --output_dir ./checkpoints/jaisp_v10 \\
        --fused_pixel_scale_arcsec 0.4 \\
        --crop_size_rubin 256 \\
        --p_adversarial 0.25 \\
        --loss_type charbonnier \\
        --charbonnier_eps 1e-3 \\
        --core_l2_weight 0.2 \\
        --core_info_threshold 0.5 \\
        --epochs 100 --lr 3e-4 --accum_steps 4 \\
        --wandb_project JAISP-Foundation-v10 \\
        --wandb_name v10_charb_corel2_02

    # Optional: warm-start from v9 best.pt to save training time.
    # Add: --resume ./checkpoints/jaisp_v9/checkpoint_best.pt --resume_weights_only

    # Multi-GPU
    torchrun --nproc_per_node=2 train_jaisp_foundation_v10.py \\
        --rubin_dir  ../data/rubin_tiles_all \\
        --euclid_dir ../data/euclid_tiles_all \\
        --output_dir ./checkpoints/jaisp_v10 \\
        --p_adversarial 0.25 \\
        --loss_type charbonnier \\
        --core_l2_weight 0.2 \\
        --epochs 100 --lr 3e-4 --accum_steps 2
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import jaisp_foundation_v8 as _v8
from train_jaisp_foundation_v9 import JAISPTrainerV9, build_argparser as build_v9_argparser
from train_jaisp_foundation_v7 import suppress_stdout

try:
    import wandb
except ImportError:
    wandb = None


class JAISPTrainerV10(JAISPTrainerV9):
    """V10 trainer: v9 + Charbonnier loss + optional core-L2 term.

    Reuses the V9 trainer's __init__ (which builds a v9 model with
    rubin_concat=True and sets up adversarial masking), then replaces the
    model with a v10 instance that has the loss-shaping parameters baked in.
    """

    def __init__(
        self,
        loss_type: str = "charbonnier",
        charbonnier_eps: float = 1e-3,
        core_l2_weight: float = 0.2,
        core_info_threshold: float = 0.5,
        **kwargs,
    ):
        self.loss_type = str(loss_type).lower()
        self.charbonnier_eps = float(charbonnier_eps)
        self.core_l2_weight = float(core_l2_weight)
        self.core_info_threshold = float(core_info_threshold)

        # Let v9 init build the v9 model. We then replace it with v10.
        super().__init__(**kwargs)

        if self.is_main_process:
            print(f"\nReplacing V9 model with V10 (loss_type={self.loss_type}, "
                  f"core_l2_weight={self.core_l2_weight})...")

        with suppress_stdout(not self.is_main_process):
            v10_model = _v8.JAISPFoundationV8(
                band_names=_v8.ALL_BANDS,
                stem_ch=self._v8_stem_ch,
                hidden_ch=self._v8_hidden_ch,
                blocks_per_stage=self._v8_blocks,
                transformer_depth=self._v8_tdepth,
                transformer_heads=self._v8_theads,
                fused_pixel_scale_arcsec=self._v8_fused,
                rubin_concat=True,
                loss_type=self.loss_type,
                charbonnier_eps=self.charbonnier_eps,
                core_l2_weight=self.core_l2_weight,
                core_info_threshold=self.core_info_threshold,
            ).to(self.device)

        if self.distributed:
            self.model = DDP(
                v10_model,
                device_ids=[self.local_rank] if self.device.type == "cuda" else None,
                find_unused_parameters=True,
            )
        else:
            self.model = v10_model

        # Re-create optimizer + scheduler for the v10 model.  Param count is
        # identical to v9 (the loss change adds no parameters), but optimizer
        # state must reset since DDP reconstructs the model wrapper.
        from jaisp_foundation_v6 import create_optimizer, create_scheduler
        lr = kwargs.get('lr', 3e-4)
        wd = kwargs.get('weight_decay', 0.05)
        warmup = kwargs.get('warmup_epochs', 5)
        epochs = kwargs.get('epochs', 80)
        self.optimizer = create_optimizer(self.model, lr=lr, weight_decay=wd)
        self.scheduler = create_scheduler(self.optimizer, warmup, epochs)

        # Save v10 markers in config so load_foundation picks the right path on resume.
        self.config['rubin_concat']        = True
        self.config['loss_type']           = self.loss_type
        self.config['charbonnier_eps']     = self.charbonnier_eps
        self.config['core_l2_weight']      = self.core_l2_weight
        self.config['core_info_threshold'] = self.core_info_threshold

        if self.is_main_process:
            n_params = sum(p.numel() for p in v10_model.parameters())
            print(f"  V10 params: {n_params/1e6:.1f}M  "
                  f"(adversarial p={self.p_adversarial}, "
                  f"loss={self.loss_type}, core_l2={self.core_l2_weight})")


# ============================================================
# CLI
# ============================================================

def build_argparser():
    p = build_v9_argparser()
    p.description = ('Train JAISP Foundation v10 (v9 architecture + Charbonnier '
                     'loss + core-L2 weighting for sharper source reconstruction).')
    p.set_defaults(
        output_dir='./checkpoints/jaisp_v10',
        fused_pixel_scale_arcsec=0.4,
        wandb_project='JAISP-Foundation-v10',
    )
    p.add_argument(
        '--loss_type', type=str, default='charbonnier',
        choices=['l1', 'charbonnier'],
        help="Per-pixel base loss. 'charbonnier' (default for v10) is L1-like at "
             "large residuals and L2-like near zero, encouraging sharp peak "
             "predictions while staying robust on noise.",
    )
    p.add_argument(
        '--charbonnier_eps', type=float, default=1e-3,
        help='Charbonnier smoothing parameter epsilon (default 1e-3). Smaller '
             '→ more L1-like; larger → more L2-like everywhere.',
    )
    p.add_argument(
        '--core_l2_weight', type=float, default=0.2,
        help='Weight for the additional L2 penalty on high-info pixels (default 0.2). '
             'Set to 0 to disable. Pushes the model to predict source cores exactly '
             'rather than median-broadening them.',
    )
    p.add_argument(
        '--core_info_threshold', type=float, default=0.5,
        help='InformationMap value above which a pixel is considered a "core" '
             'and gets the core-L2 penalty (default 0.5).',
    )
    return p


if __name__ == '__main__':
    args = build_argparser().parse_args()
    trainer_kwargs = vars(args).copy()
    resume = trainer_kwargs.pop('resume', None)
    resume_weights_only = trainer_kwargs.pop('resume_weights_only', False)
    trainer = JAISPTrainerV10(**trainer_kwargs)
    trainer.train(resume_from=resume, resume_weights_only=resume_weights_only)
