"""Add real held-out local patches using the cached Q1 MER catalog.

The original evaluation used selected test-scene centers. This script samples
new Euclid/Rubin patch IDs from the cached local image inventory, excluding
every train/validation/test patch ID and existing real-MER region. It centers
each 20-arcsec archive cutout on a nearby unused MER source.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import torch
from astropy.table import Table
from .real_mer_fetch import OUT, fetch_region


def candidates():
    split_data = torch.load(
        'models/photometry/self_supervised/runs/q1_all_bands/scenes.pt',
        weights_only=False, map_location='cpu')
    held_patch_ids = {s['tile'] for split in split_data['splits'].values() for s in split}
    existing_patch_ids = set()
    for meta_path in OUT.glob('region_*/metadata.json'):
        try:
            existing_patch_ids.add(json.loads(meta_path.read_text())['tile'])
        except (OSError, KeyError, json.JSONDecodeError):
            pass
    cat = Table.read(OUT / 'mer_catalog.fits')
    ra = np.asarray(cat['ra'], dtype=float)
    dec = np.asarray(cat['dec'], dtype=float)
    ids = np.asarray(cat['object_id'], dtype=np.int64)
    sky = np.column_stack((ra, dec))
    rubin_ids = {p.stem for p in Path('data/rubin_tiles_all').glob('*.npz')}
    rows = []
    for path in sorted(Path('data/euclid_tiles_all_q1').glob('*.npz')):
        with np.load(path) as tile:
            patch_id = tile['tile_id'].item()
            if isinstance(patch_id, bytes):
                patch_id = patch_id.decode()
            patch_id = str(patch_id)
            if patch_id in held_patch_ids or patch_id in existing_patch_ids:
                continue
            if patch_id not in rubin_ids:
                continue
            ra0, dec0 = float(tile['ra_center']), float(tile['dec_center'])
        dra = (ra - ra0) * np.cos(np.deg2rad(dec0)) * 3600
        ddec = (dec - dec0) * 3600
        distance = np.hypot(dra, ddec)
        # Keep centers on a catalog source and avoid weakly populated patches.
        near = np.flatnonzero(distance <= 9.)
        if len(near) < 5:
            continue
        rows.append(dict(patch_id=patch_id, ra=ra0, dec=dec0,
                         nearby=int(len(near)), nearest=near[np.argmin(distance[near])],
                         source_indices=near, distance=distance))

    # Greedily spread the new cutouts across independent patch locations.
    # Favor source-rich cutouts, with ties resolved by their stable patch ID.
    rows.sort(key=lambda r: (-r['nearby'], r['patch_id']))
    selected = []
    used_ids = set()
    for row in rows:
        if any(np.hypot((row['ra']-q['ra'])*np.cos(np.deg2rad(row['dec']))*3600,
                        (row['dec']-q['dec'])*3600) < 38 for q in selected):
            continue
        candidates_here = [int(i) for i in row['source_indices'] if int(ids[i]) not in used_ids]
        if len(candidates_here) < 5:
            continue
        # Put a real MER source at the cutout center, then use the complete
        # catalog inside that native 20-arcsec cutout during preparation.
        nearest = min(candidates_here, key=lambda i: row['distance'][i])
        row['central_index'] = nearest
        selected.append(row)
        used_ids.add(int(ids[nearest]))
    return cat, selected


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--count', type=int, default=30)
    p.add_argument('--workers', type=int, default=4)
    args = p.parse_args()
    cat, selected = candidates()
    if len(selected) < args.count:
        raise RuntimeError(f'Only {len(selected)} well-separated new patches have local MER coverage')
    selected = selected[:args.count]
    first = 41
    existing_numbers = [int(p.name.split('_')[1]) for p in OUT.glob('region_*') if p.is_dir()]
    first = max([40, *existing_numbers]) + 1
    products = json.loads((OUT / 'products.json').read_text())
    tasks = []
    for j, row in enumerate(selected):
        source = row['central_index']
        scene = dict(tile=row['patch_id'], central=0,
                     sky=np.array([[float(cat['ra'][source]), float(cat['dec'][source])]]))
        tasks.append((first+j, scene, products))
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        infos = list(pool.map(fetch_region, tasks))
    record = dict(first_region=first, regions=infos,
                  selected_patch_ids=[r['patch_id'] for r in selected],
                  selected_centers=[[float(cat['ra'][r['central_index']]),
                                     float(cat['dec'][r['central_index']])] for r in selected])
    (OUT / f'expansion_{first:03d}_{first+len(infos)-1:03d}.json').write_text(json.dumps(record, indent=2))
    print(json.dumps(dict(regions=len(infos), first=first,
                          patches=len({x['tile'] for x in infos}),
                          centers=[[round(x[0], 6), round(x[1], 6)] for x in record['selected_centers']]), indent=2))


if __name__ == '__main__':
    main()
