#!/bin/bash
# A/B for the photometric shape prior: _cr (fitted kernel + CR mask) vs _sp (same + shape prior).
set -u
cd /home/shemmati/Work/Projects/JAISP
F="Deprecation|FITSFixedWarning|warnings.warn|RuntimeWarning|ret = ret|_methods|JAISPFoundation|stem_ch|stream_depths|rubin_concat|loss_type|decoder channels|Loaded v11|^\s*$"
COMMON="--sky-targets 0,1080,1170 --iters 100 --lam-step 13.4 --smooth 1.0 --star-psf --align-holdout --device cuda --kernel experiments/grism/nisps_kernel_fit_kfit2.npz"
for tag in _cr _sp; do
  extra=""; [ "$tag" = "_sp" ] && extra="--shape-prior 1.0"
  echo "=== run $tag $(date)"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /usr/bin/python3 -u experiments/grism/scene_test.py $COMMON $extra --tag $tag 2>&1 | /usr/bin/grep --line-buffered -v -E "$F"
  echo "=== comparison $tag $(date)"
  /usr/bin/python3 experiments/grism/compare_with_sir.py --sky-targets 0,1080,1170 --dither 0 --spectra experiments/grism/scene_holdout_det23_d3${tag}_spectra.pt --lam-step 13.4 --nshow 6 --galaxies --tag $tag 2>&1 | /usr/bin/grep -v -E "$F"
done
echo "=== overlap metrics $(date)"
for tag in _cr _sp; do OMP_NUM_THREADS=16 /usr/bin/python3 experiments/grism/eval_overlap.py --tag $tag --star-psf --kernel experiments/grism/nisps_kernel_fit_kfit2.npz 2>&1 | /usr/bin/grep -v -E "$F"; done
echo "=== all done $(date)"
