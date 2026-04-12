"""Training script for JAISP Foundation v8 — fine-scale bottleneck with random crop.

Thin wrapper around the v7 trainer that:
  1. Replaces JAISPFoundationV7 with V8 (configurable stream depths)
  2. Injects random cropping into _prepare_batch before context/target split

Everything else (DDP, wandb, checkpointing, visualization, loss) is
inherited from the v7 trainer unchanged.

Usage:
    # Single GPU — 256×256 Rubin crops at 0.4"/px fused scale
    python train_jaisp_foundation_v8.py \\
        --rubin_dir  ../data/rubin_tiles_all \\
        --euclid_dir ../data/euclid_tiles_all \\
        --output_dir ./checkpoints/jaisp_v8_fine \\
        --fused_pixel_scale_arcsec 0.4 \\
        --crop_size_rubin 256 \\
        --hidden_ch 256 --transformer_depth 4 --transformer_heads 8 \\
        --epochs 100 --lr 3e-4 --accum_steps 4 \\
        --wandb_project JAISP-Foundation-v8 \\
        --wandb_name v8_fused04_crop256

    # Multi-GPU
    torchrun --nproc_per_node=2 train_jaisp_foundation_v8.py \\
        --rubin_dir  ../data/rubin_tiles_all \\
        --euclid_dir ../data/euclid_tiles_all \\
        --output_dir ./checkpoints/jaisp_v8_fine \\
        --fused_pixel_scale_arcsec 0.4 \\
        --crop_size_rubin 256 \\
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

# Monkey-patch: make v7 trainer import V8 model when we construct it.
import jaisp_foundation_v8 as _v8
import jaisp_foundation_v7 as _v7
from jaisp_dataset_v8 import random_crop_sample
from train_jaisp_foundation_v7 import (
    JAISPTrainerV7,
    build_argparser as build_v7_argparser,
    suppress_stdout,
)

try:
    import wandb
except ImportError:
    wandb = None


class JAISPTrainerV8(JAISPTrainerV7):
    """V8 trainer: fine-scale bottleneck with random crop.

    After the v7 __init__ builds a V7 model, we replace it with V8.
    Then _prepare_batch applies random cropping before the split.
    """

    def __init__(self, crop_size_rubin: int = 256, **kwargs):
        self.crop_size_rubin = int(crop_size_rubin)

        # Store v8-relevant config before super().__init__
        self._v8_stem_ch = kwargs.get('stem_ch', 64)
        self._v8_hidden_ch = kwargs.get('hidden_ch', 256)
        self._v8_blocks = kwargs.get('blocks_per_stage', 2)
        self._v8_tdepth = kwargs.get('transformer_depth', 4)
        self._v8_theads = kwargs.get('transformer_heads', 8)
        self._v8_fused = kwargs.get('fused_pixel_scale_arcsec', 0.4)

        # Let v7 init build everything (model, optimizer, scheduler, wandb).
        super().__init__(**kwargs)

        # Now replace the model with V8.
        if self.is_main_process:
            print(f"\nReplacing V7 model with V8 (fused_scale={self._v8_fused}, "
                  f"crop={self.crop_size_rubin})...")

        with suppress_stdout(not self.is_main_process):
            v8_model = _v8.JAISPFoundationV8(
                band_names=_v8.ALL_BANDS,
                stem_ch=self._v8_stem_ch,
                hidden_ch=self._v8_hidden_ch,
                blocks_per_stage=self._v8_blocks,
                transformer_depth=self._v8_tdepth,
                transformer_heads=self._v8_theads,
                fused_pixel_scale_arcsec=self._v8_fused,
            ).to(self.device)

        if self.distributed:
            self.model = DDP(
                v8_model,
                device_ids=[self.local_rank] if self.device.type == "cuda" else None,
                find_unused_parameters=True,
            )
        else:
            self.model = v8_model

        # Re-create optimizer and scheduler for the new model params.
        from jaisp_foundation_v6 import create_optimizer, create_scheduler
        lr = kwargs.get('lr', 3e-4)
        wd = kwargs.get('weight_decay', 0.05)
        warmup = kwargs.get('warmup_epochs', 5)
        epochs = kwargs.get('epochs', 80)
        self.optimizer = create_optimizer(self.model, lr=lr, weight_decay=wd)
        self.scheduler = create_scheduler(self.optimizer, warmup, epochs)

        if self.is_main_process:
            depths = _v8.compute_stream_depths(self._v8_fused)
            print(f"  V8 stream depths: {depths}")
            n_params = sum(p.numel() for p in v8_model.parameters())
            print(f"  V8 params: {n_params/1e6:.1f}M")

    def _prepare_batch(self, sample: dict, rng: np.random.RandomState, force_phase_b: bool = False):
        """Apply random crop before v7's context/target split."""
        if self.crop_size_rubin > 0:
            sample = random_crop_sample(sample, self.crop_size_rubin, rng)
        return super()._prepare_batch(sample, rng, force_phase_b=force_phase_b)


# ============================================================
# CLI
# ============================================================

def build_argparser():
    p = build_v7_argparser()
    p.description = 'Train JAISP Foundation v8 (fine-scale bottleneck with random crop).'
    p.set_defaults(
        output_dir='./checkpoints/jaisp_v8_fine',
        fused_pixel_scale_arcsec=0.4,
        wandb_project='JAISP-Foundation-v8',
    )
    p.add_argument(
        '--crop_size_rubin', type=int, default=256,
        help='Random crop side in Rubin pixels (default 256). '
             'Set to 0 to disable cropping (use full tiles). '
             'With fused_scale=0.4, crop=256 gives ~128×128 bottleneck tokens '
             '(same cost as v7 at fused_scale=0.8 with full 512 tiles).',
    )
    return p


if __name__ == '__main__':
    args = build_argparser().parse_args()
    trainer_kwargs = vars(args).copy()
    resume = trainer_kwargs.pop('resume', None)
    resume_weights_only = trainer_kwargs.pop('resume_weights_only', False)
    trainer = JAISPTrainerV8(**trainer_kwargs)
    trainer.train(resume_from=resume, resume_weights_only=resume_weights_only)
