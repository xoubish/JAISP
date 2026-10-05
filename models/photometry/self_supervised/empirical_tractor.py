"""Resumable, truth-blind Tractor shape/flux fitting for empirical.py exports.

Use runs/tractor_env/bin/python; intentionally no torch imports.
"""
from pathlib import Path
import argparse
import json
import hashlib
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
# Avoid the local JAISP astrometry.py shadowing upstream astrometry.util.
sys.path.insert(0, str(Path(__file__).resolve().parent/'runs/tractor_dependencies'))
from tractor_compare import fit_scene


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--run', type=Path, required=True); p.add_argument('--workers', type=int, default=8)
    p.add_argument('--count', type=int)
    a = p.parse_args(); dest = a.run/'tractor_results'; dest.mkdir(exist_ok=True)
    paths = sorted((a.run/'tractor_inputs').glob('scene_*.npz'))
    if a.count is not None: paths = paths[:a.count]
    digests={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    for path in paths:
        result=dest/(path.stem+'.json')
        if result.exists() and json.loads(result.read_text()).get('input_sha256') != digests[path]:
            raise RuntimeError(f'Input changed since fit: {path}; use a fresh run directory')
    pending = [p for p in paths if not (dest/(p.stem+'.json')).exists()]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures = {pool.submit(fit_scene, p):p for p in pending}
        for i, future in enumerate(as_completed(futures)):
            path = futures[future]; result = future.result()
            result['input_sha256']=digests[path]
            (dest/(path.stem+'.json')).write_text(json.dumps(result, indent=2))
            print(f"Tractor {path.stem}: {'FAILED' if 'error' in result else 'OK'}, {result['seconds']:.1f}s ({i+1}/{len(pending)})", flush=True)
    results = [json.loads((dest/(p.stem+'.json')).read_text()) for p in paths]
    pd.DataFrame([r for x in results for r in x.get('rows', [])]).to_csv(a.run/'tractor_fluxes.csv', index=False)
    (a.run/'tractor_failures.json').write_text(json.dumps([x for x in results if 'error' in x], indent=2))
    (a.run/'tractor_protocol.json').write_text(json.dumps(dict(
        shape_fit='Existing upstream full VIS profile ladder; every source position fixed',
        flux_fit='Native signed joint least squares plus constant sky; full blend covariance',
        truth_access='Truth read only for reporting after all fits; no donor morphology supplied',
        shared_inputs='Same scene NPZ data, variance, mask, positions and pixelized PSF as JAISP',
        failures='All failures retained; comparisons report complete matched sets and failure rates'), indent=2))
    if any('error' in x for x in results): raise RuntimeError('Tractor failures recorded in tractor_failures.json')


if __name__ == '__main__': main()
