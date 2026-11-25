#!/bin/bash
# Foundation stamps + fitted kernel (clean pair with base2/kfit2), then the SIR comparison and overlap metrics.
set -u
cd /home/shemmati/Work/Projects/JAISP
F="Deprecation|FITSFixedWarning|warnings.warn|RuntimeWarning|ret = ret|_methods|JAISPFoundation|stem_ch|stream_depths|rubin_concat|loss_type|decoder channels|Loaded v11|^\s*$"
while pgrep -f "run_base[2].sh" >/dev/null; do sleep 30; done
echo "=== fndk run (foundation stamps, fitted kernel) $(date)"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /usr/bin/python3 -u experiments/grism/scene_test.py --sky-targets 0,1080,1170 --iters 100 --lam-step 13.4 --smooth 1.0 --star-psf --align-holdout --device cuda --kernel experiments/grism/nisps_kernel_fit_kfit2.npz --foundation-stamps --tag _fndk 2>&1 | /usr/bin/grep --line-buffered -v -E "$F"
echo "=== comparison fndk $(date)"
/usr/bin/python3 experiments/grism/compare_with_sir.py --sky-targets 0,1080,1170 --dither 0 --spectra experiments/grism/scene_holdout_det23_d3_fndk_spectra.pt --lam-step 13.4 --nshow 6 --galaxies --tag _fndk 2>&1 | /usr/bin/grep -v -E "$F"
echo "=== overlap metrics kfit2 / base2 / fndk $(date)"
OMP_NUM_THREADS=16 /usr/bin/python3 experiments/grism/eval_overlap.py --tag _kfit2 --star-psf 2>&1 | /usr/bin/grep -v -E "$F"
OMP_NUM_THREADS=16 /usr/bin/python3 experiments/grism/eval_overlap.py --tag _base2 --star-psf 2>&1 | /usr/bin/grep -v -E "$F"
OMP_NUM_THREADS=16 /usr/bin/python3 experiments/grism/eval_overlap.py --tag _fndk --star-psf --foundation-stamps --kernel experiments/grism/nisps_kernel_fit_kfit2.npz 2>&1 | /usr/bin/grep -v -E "$F"
echo "=== all done $(date)"
