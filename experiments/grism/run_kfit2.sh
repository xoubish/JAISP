#!/bin/bash
set -u
cd /home/shemmati/Work/Projects/JAISP
F="Deprecation|FITSFixedWarning|warnings.warn|RuntimeWarning|ret = ret|_methods|JAISPFoundation|stem_ch|stream_depths|rubin_concat|loss_type|decoder channels|Loaded v11|^\s*$"
while pgrep -f "run_chai[n].sh" >/dev/null; do sleep 30; done
echo "=== kernel-fit run 2 (star PSF stamps, kernel learnable) $(date)"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /usr/bin/python3 -u experiments/grism/scene_test.py --sky-targets 0,1080,1170 --iters 100 --lam-step 13.4 --smooth 1.0 --star-psf --align-holdout --device cuda --fit-kernel --tag _kfit2 2>&1 | /usr/bin/grep --line-buffered -v -E "$F"
echo "=== comparison kfit2 $(date)"
/usr/bin/python3 experiments/grism/compare_with_sir.py --sky-targets 0,1080,1170 --dither 0 --spectra experiments/grism/scene_holdout_det23_d3_kfit2_spectra.pt --lam-step 13.4 --nshow 6 --galaxies --tag _kfit2 2>&1 | /usr/bin/grep -v -E "$F"
echo "=== all done $(date)"
