#!/bin/bash
# Chained grism runs: kernel fit (star PSF stamps, held-out alignment) -> SIR comparison ->
# foundation-stamp run with the fitted kernel -> SIR comparison.
set -u
cd /home/shemmati/Work/Projects/JAISP
F="Deprecation|FITSFixedWarning|warnings.warn|RuntimeWarning|ret = ret|_methods|JAISPFoundation|stem_ch|stream_depths|rubin_concat|loss_type|decoder channels|Loaded v11|^\s*$"
COMMON="--sky-targets 0,1080,1170 --iters 100 --lam-step 13.4 --smooth 1.0 --star-psf --align-holdout --device cuda"
echo "=== kernel-fit run (star PSF stamps) $(date)"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /usr/bin/python3 -u experiments/grism/scene_test.py $COMMON --fit-kernel --tag _kfit 2>&1 | /usr/bin/grep --line-buffered -v -E "$F"
echo "=== comparison kfit $(date)"
/usr/bin/python3 experiments/grism/compare_with_sir.py --sky-targets 0,1080,1170 --dither 0 --spectra experiments/grism/scene_holdout_det23_d3_kfit_spectra.pt --lam-step 13.4 --nshow 6 --galaxies --tag _kfit 2>&1 | /usr/bin/grep -v -E "$F"
echo "=== foundation-stamps run $(date)"
K=experiments/grism/nisps_kernel_fit_kfit.npz; [ -f $K ] || K=experiments/grism/nisps_kernel.npz
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /usr/bin/python3 -u experiments/grism/scene_test.py $COMMON --kernel $K --foundation-stamps --tag _fnd 2>&1 | /usr/bin/grep --line-buffered -v -E "$F"
echo "=== comparison fnd $(date)"
/usr/bin/python3 experiments/grism/compare_with_sir.py --sky-targets 0,1080,1170 --dither 0 --spectra experiments/grism/scene_holdout_det23_d3_fnd_spectra.pt --lam-step 13.4 --nshow 6 --galaxies --tag _fnd 2>&1 | /usr/bin/grep -v -E "$F"
echo "=== all done $(date)"
