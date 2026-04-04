#!/usr/bin/env python3
"""Validate and ingest JAISP tile tarballs into the data directory.

For each tarball (tiles_product.tar.gz, batch_*.tar.gz, ...):
  - Parses tract/patch from the directory structure inside the archive
  - Validates each Rubin tile: NO_DATA mask, band availability, image stats
  - Validates paired Euclid tile if present
  - Optionally extracts valid tiles, preserving tract/patch subdirectory layout
  - Prints a per-tile summary table and final statistics

Supported archive layouts include both older flat products such as:
    rubin_tiles_tract5063/patch00/tile_x00000_y00000.npz
    euclid_tiles_tract5063/tile_x00000_y00000_euclid.npz

and newer mixed products such as:
    tiles_product/tract_5063/patch_25/tile_x00000_y00000.npz
    tiles_product/tract_5063/patch_25/tile_x00000_y00000_euclid.npz

Usage
-----
    # Dry run: report contents without extracting
    python io/ingest_tiles.py data/tiles_product.tar.gz --dry_run

    # Validate + extract
    python io/ingest_tiles.py data/tiles_product.tar.gz --out_dir data/

    # Multiple archives
    python io/ingest_tiles.py data/batch_*.tar.gz --out_dir data/ --dry_run

Hierarchical output layout:
    data/rubin_tiles_tract5063/patch00/tile_x00256_y00512.npz
    data/euclid_tiles_tract5063/tile_x00256_y00512_euclid.npz
    data/euclid_tiles_tract5063/patch_25/tile_x00256_y00512_euclid.npz

Flat output layout (`--layout flat`):
    data/rubin_tiles_all/tile_x00256_y00512_tract5063_patch00.npz
    data/euclid_tiles_all/tile_x00256_y00512_tract5063_patch00_euclid.npz
"""

from __future__ import annotations

import argparse
import io
import sys
import tarfile
from pathlib import Path, PurePosixPath

import numpy as np

# ── Rubin bitmask planes (LSST DM convention) ──────────────────────────────
NO_DATA_BIT  = 256   # NO_DATA plane
BAD_BIT      = 1     # BAD (defect)
SAT_BIT      = 2     # SAT (saturated)
DETECTED_BIT = 32    # DETECTED

EXPECTED_RUBIN_BANDS = {"u", "g", "r", "i", "z", "y"}
EXPECTED_EUCLID_BANDS = {"VIS", "Y", "J", "H"}


# ── helpers ─────────────────────────────────────────────────────────────────

def _parse_path(archive_path: str) -> dict:
    """Extract tract, patch, tile_id, and instrument from an archive member path.

    Handles both:
      rubin_tiles_tract5063/patch00/tile_x00000_y00000.npz
      euclid_tiles_tract5063/tile_x00000_y00000_euclid.npz
      tiles_product/tract_5063/patch_25/tile_x00000_y00000.npz
      tiles_product/tract_5063/patch_25/tile_x00000_y00000_euclid.npz
    """
    p = PurePosixPath(archive_path)
    parts = p.parts           # e.g. ('rubin_tiles_tract5063', 'patch00', 'tile_x…npz')
    info = {
        'archive_path': archive_path,
        'filename': p.name,
        'tract': None,
        'patch': None,
        'tile_id': None,
        'instrument': None,
    }

    filename = p.name
    if filename.startswith('._'):
        return info

    inferred_instrument = None
    if filename.endswith('_euclid.npz'):
        inferred_instrument = 'euclid'
    elif filename.endswith('.npz'):
        inferred_instrument = 'rubin'

    # Identify instrument / tract / patch from any meaningful path component so
    # we can handle archives rooted at "data/", "tiles_product/", etc.
    for idx, part in enumerate(parts[:-1]):
        if part.startswith('rubin_tiles_tract'):
            info['instrument'] = 'rubin'
            info['tract'] = part.replace('rubin_tiles_tract', '')
            if idx + 1 < len(parts) - 1 and parts[idx + 1].startswith('patch'):
                info['patch'] = parts[idx + 1]
            break
        if part.startswith('euclid_tiles_tract'):
            info['instrument'] = 'euclid'
            info['tract'] = part.replace('euclid_tiles_tract', '')
            if idx + 1 < len(parts) - 1 and parts[idx + 1].startswith('patch'):
                info['patch'] = parts[idx + 1]
            break
        if part.startswith('rubin_tiles_ecdfs'):
            info['instrument'] = 'rubin'
            info['tract'] = 'ecdfs'
            if idx + 1 < len(parts) - 1 and parts[idx + 1].startswith('patch'):
                info['patch'] = parts[idx + 1]
            break
        if part.startswith('euclid_tiles_ecdfs'):
            info['instrument'] = 'euclid'
            info['tract'] = 'ecdfs'
            if idx + 1 < len(parts) - 1 and parts[idx + 1].startswith('patch'):
                info['patch'] = parts[idx + 1]
            break
        if part.startswith('tract_'):
            info['instrument'] = inferred_instrument
            info['tract'] = part.replace('tract_', '')
            if idx + 1 < len(parts) - 1 and parts[idx + 1].startswith('patch'):
                info['patch'] = parts[idx + 1]
            break

    if info['instrument'] is None:
        info['instrument'] = inferred_instrument

    # Tile ID from filename
    stem = p.stem
    if stem.endswith('_euclid'):
        stem = stem[:-7]
    if stem.startswith('tile_'):
        info['tile_id'] = stem

    return info


def _validate_rubin(data: np.lib.npyio.NpzFile) -> dict:
    """Validate a Rubin tile npz.  Returns a dict of findings."""
    result = {
        'valid': False,
        'bands': [],
        'missing_bands': [],
        'no_data_frac': 0.0,
        'has_data': False,
        'img_stats': {},
        'issues': [],
    }

    # Bands
    try:
        bands = list(data['bands'])
        result['bands'] = bands
        result['missing_bands'] = sorted(EXPECTED_RUBIN_BANDS - set(bands))
        if result['missing_bands']:
            result['issues'].append(f"missing bands: {result['missing_bands']}")
    except Exception as e:
        result['issues'].append(f"could not read bands: {e}")
        return result

    # Mask / NO_DATA
    try:
        mask = data['mask']
        no_data_frac = float(np.mean((mask & NO_DATA_BIT).astype(bool)))
        result['no_data_frac'] = no_data_frac
        result['has_data'] = no_data_frac < 0.99
        if no_data_frac > 0.99:
            result['issues'].append('tile is >99% NO_DATA')
    except Exception as e:
        result['issues'].append(f"could not read mask: {e}")
        return result

    # Image stats (only if has data)
    if result['has_data']:
        try:
            img = data['img']
            var = data['var']
            for i, b in enumerate(bands):
                finite_pix = img[i][np.isfinite(img[i]) & ~(mask[i].astype(bool) & NO_DATA_BIT)]
                if len(finite_pix) == 0:
                    result['issues'].append(f"band {b}: no finite non-masked pixels")
                    continue
                v_finite = var[i][np.isfinite(var[i]) & ~(mask[i].astype(bool) & NO_DATA_BIT)]
                inf_var_frac = float(np.mean(~np.isfinite(var[i])))
                result['img_stats'][b] = {
                    'mean': float(np.mean(finite_pix)),
                    'std':  float(np.std(finite_pix)),
                    'inf_var_frac': inf_var_frac,
                }
                if inf_var_frac > 0.5:
                    result['issues'].append(f"band {b}: {inf_var_frac:.0%} inf variance")
        except Exception as e:
            result['issues'].append(f"image stats failed: {e}")

    result['valid'] = result['has_data'] and len(result['issues']) == 0
    return result


def _validate_euclid(data: np.lib.npyio.NpzFile) -> dict:
    """Validate a Euclid tile npz."""
    result = {
        'valid': False,
        'bands': [],
        'missing_bands': [],
        'vis_shape': None,
        'nisp_shape': None,
        'issues': [],
    }
    keys = set(data.keys())
    present = {b for b in EXPECTED_EUCLID_BANDS if f'img_{b}' in keys}
    result['bands'] = sorted(present)
    result['missing_bands'] = sorted(EXPECTED_EUCLID_BANDS - present)
    if result['missing_bands']:
        result['issues'].append(f"missing bands: {result['missing_bands']}")

    if 'img_VIS' in keys:
        result['vis_shape'] = tuple(data['img_VIS'].shape)
    else:
        result['issues'].append('img_VIS missing')

    if 'img_Y' in keys:
        result['nisp_shape'] = tuple(data['img_Y'].shape)

    result['valid'] = len(result['issues']) == 0
    return result


# ── main logic ───────────────────────────────────────────────────────────────

def _fmt(val, width=8):
    if isinstance(val, float):
        return f'{val:{width}.3f}'
    return str(val)[:width].ljust(width)


def _flat_filename(info: dict) -> str:
    """Build a flat filename that still preserves tract/patch provenance."""
    tile_id = info['tile_id']
    tract = info['tract'] or 'unknown'
    patch = info['patch']
    patch_part = f'_{patch}' if patch else ''

    if info['instrument'] == 'euclid':
        return f'{tile_id}_tract{tract}{patch_part}_euclid.npz'
    return f'{tile_id}_tract{tract}{patch_part}.npz'


def process_archive(
    archive_path: Path,
    out_dir: Path | None,
    dry_run: bool,
    layout: str = 'hierarchical',
    nsig_data: float = 3.0,
) -> list[dict]:
    """Process one tarball. Returns list of per-tile result dicts."""
    print(f'\n{"="*70}')
    print(f'Archive: {archive_path}')
    print(f'{"="*70}')

    results = []

    with tarfile.open(archive_path, 'r:gz') as tar:
        members = tar.getmembers()
        npz_members = [m for m in members if m.name.endswith('.npz')]

        # Group by tile_id within each tract/patch
        rubin_map: dict[str, tarfile.TarInfo] = {}
        euclid_map: dict[str, tarfile.TarInfo] = {}
        matched_euclid_keys: set[str] = set()

        for m in npz_members:
            info = _parse_path(m.name)
            if info['tile_id'] is None:
                continue
            key = f"{info['tract']}/{info['patch'] or 'nopatch'}/{info['tile_id']}"
            if info['instrument'] == 'rubin':
                rubin_map[key] = (m, info)
            elif info['instrument'] == 'euclid':
                euclid_map[key] = (m, info)

        print(f'  Found {len(rubin_map)} Rubin tiles, {len(euclid_map)} Euclid tiles')
        print()

        # Print header
        print(f"  {'tile_id':<22} {'tract':<8} {'patch':<8} {'bands':<14} "
              f"{'no_data%':<10} {'has_data':<10} {'issues'}")
        print(f"  {'-'*22} {'-'*8} {'-'*8} {'-'*14} {'-'*10} {'-'*10} {'-'*30}")

        n_valid = n_invalid = n_no_data = 0

        for key in sorted(rubin_map):
            m, info = rubin_map[key]
            tile_id = info['tile_id']
            tract   = info['tract']
            patch   = info['patch'] or '-'

            # Load and validate
            f = tar.extractfile(m)
            data = np.load(io.BytesIO(f.read()), allow_pickle=True)
            r = _validate_rubin(data)

            # Prefer same-patch Euclid matches for mixed archives, but fall back
            # to flat tract-level Euclid layouts used by older products.
            euclid_keys = [f"{tract}/{info['patch'] or 'nopatch'}/{tile_id}"]
            if info['patch'] is not None:
                euclid_keys.append(f"{tract}/nopatch/{tile_id}")
            euclid_res = None
            euclid_key = None
            for candidate in euclid_keys:
                if candidate in euclid_map:
                    euclid_key = candidate
                    em, einfo = euclid_map[euclid_key]
                    ef = tar.extractfile(em)
                    edata = np.load(io.BytesIO(ef.read()), allow_pickle=True)
                    euclid_res = _validate_euclid(edata)
                    matched_euclid_keys.add(euclid_key)
                    break

            bands_str = ','.join(r['bands']) if r['bands'] else '?'
            issues_str = '; '.join(r['issues'][:2]) if r['issues'] else 'ok'
            if euclid_res and euclid_res['issues']:
                issues_str += f" | euclid: {'; '.join(euclid_res['issues'][:1])}"

            status = 'YES' if r['has_data'] else 'NO'
            print(f"  {tile_id:<22} {tract:<8} {patch:<8} {bands_str:<14} "
                  f"{r['no_data_frac']*100:<10.1f} {status:<10} {issues_str}")

            rec = {
                'archive': str(archive_path.name),
                'tile_id': tile_id,
                'tract':   tract,
                'patch':   patch,
                'ra':      float(data.get('ra_center', np.nan)),
                'dec':     float(data.get('dec_center', np.nan)),
                'bands':   r['bands'],
                'missing_bands': r['missing_bands'],
                'no_data_frac':  r['no_data_frac'],
                'has_data':      r['has_data'],
                'valid':         r['valid'],
                'issues':        r['issues'],
                'has_euclid':    euclid_res is not None,
                'euclid_valid':  euclid_res['valid'] if euclid_res else False,
                'euclid_issues': euclid_res['issues'] if euclid_res else [],
            }
            results.append(rec)

            if r['has_data']:
                n_valid += 1
            else:
                n_no_data += 1
            if r['issues'] and r['has_data']:
                n_invalid += 1

            # Extract if requested
            if not dry_run and out_dir and r['has_data']:
                _extract_tile(tar, m, info, out_dir, layout=layout)
                if euclid_res and euclid_key in euclid_map:
                    em, einfo = euclid_map[euclid_key]
                    _extract_tile(tar, em, einfo, out_dir, layout=layout)

        unmatched_euclid = sorted(set(euclid_map) - matched_euclid_keys)
        if unmatched_euclid:
            print()
            print(f'  Note: {len(unmatched_euclid)} Euclid tiles had no Rubin counterpart')
            for key in unmatched_euclid[:5]:
                print(f'    unmatched euclid: {key}')
            if len(unmatched_euclid) > 5:
                print(f'    ... {len(unmatched_euclid) - 5} more')

    print()
    print(f'  Summary: {n_valid} with data, {n_no_data} NO_DATA, '
          f'{n_invalid} with data but issues')
    return results


def _extract_tile(
    tar: tarfile.TarFile,
    member: tarfile.TarInfo,
    info: dict,
    out_dir: Path,
    layout: str = 'hierarchical',
) -> Path:
    """Extract one tile member to out_dir using the requested output layout."""
    instrument = info['instrument']
    tract      = info['tract']
    patch      = info['patch']

    if layout == 'flat':
        if instrument == 'rubin':
            subdir = out_dir / 'rubin_tiles_all'
        else:
            subdir = out_dir / 'euclid_tiles_all'
        dest_name = _flat_filename(info)
    else:
        if instrument == 'rubin':
            if tract == 'ecdfs':
                subdir = out_dir / 'rubin_tiles_ecdfs'
            else:
                subdir = out_dir / f'rubin_tiles_tract{tract}'
                if patch:
                    subdir = subdir / patch
        else:
            if tract == 'ecdfs':
                subdir = out_dir / 'euclid_tiles_ecdfs'
            else:
                subdir = out_dir / f'euclid_tiles_tract{tract}'
                if patch:
                    subdir = subdir / patch
        dest_name = Path(member.name).name

    subdir.mkdir(parents=True, exist_ok=True)
    dest = subdir / dest_name

    if dest.exists():
        return dest

    f = tar.extractfile(member)
    dest.write_bytes(f.read())
    return dest


def print_summary(all_results: list[dict]) -> None:
    print(f'\n{"="*70}')
    print('OVERALL SUMMARY')
    print(f'{"="*70}')
    total = len(all_results)
    with_data = sum(1 for r in all_results if r['has_data'])
    no_data   = total - with_data
    with_euclid = sum(1 for r in all_results if r['has_euclid'] and r['has_data'])
    full_6band = sum(1 for r in all_results if r['has_data'] and not r['missing_bands'])
    miss_z     = sum(1 for r in all_results if r['has_data'] and 'z' in r['missing_bands'])

    print(f'  Total tiles inspected : {total}')
    print(f'  Tiles with data       : {with_data}')
    print(f'  NO_DATA tiles         : {no_data}')
    print(f'  With Euclid match     : {with_euclid}')
    print(f'  Full 6-band Rubin     : {full_6band}')
    print(f'  Missing z-band        : {miss_z}')

    # By tract
    tracts = sorted({r['tract'] for r in all_results})
    print(f'\n  Breakdown by tract:')
    for tr in tracts:
        sub = [r for r in all_results if r['tract'] == tr]
        nd = sum(1 for r in sub if r['has_data'])
        eu = sum(1 for r in sub if r['has_euclid'] and r['has_data'])
        bds = sub[0]['bands'] if sub else []
        print(f'    tract {tr:<8}: {nd:3d}/{len(sub):3d} with data, '
              f'{eu:3d} have Euclid, bands={bds}')

    ra_vals  = [r['ra']  for r in all_results if r['has_data'] and np.isfinite(r['ra'])]
    dec_vals = [r['dec'] for r in all_results if r['has_data'] and np.isfinite(r['dec'])]
    if ra_vals:
        print(f'\n  RA  range : {min(ra_vals):.4f} – {max(ra_vals):.4f} deg')
        print(f'  Dec range : {min(dec_vals):.4f} – {max(dec_vals):.4f} deg')


def main():
    p = argparse.ArgumentParser(description='Validate and ingest JAISP tile tarballs.')
    p.add_argument('archives', nargs='+', help='One or more .tar.gz files')
    p.add_argument('--out_dir', default=None,
                   help='Root data directory to extract into (default: dry-run only)')
    p.add_argument('--layout', choices=['hierarchical', 'flat'], default='hierarchical',
                   help='Extraction layout for written files (default: hierarchical)')
    p.add_argument('--dry_run', action='store_true',
                   help='Validate without extracting (default if --out_dir not given)')
    args = p.parse_args()

    dry_run = args.dry_run or args.out_dir is None
    out_dir = Path(args.out_dir) if args.out_dir else None

    if dry_run:
        print('[DRY RUN — no files will be written]')

    all_results = []
    for archive in args.archives:
        path = Path(archive)
        if not path.exists():
            print(f'[warn] not found: {path}', file=sys.stderr)
            continue
        results = process_archive(path, out_dir, dry_run, layout=args.layout)
        all_results.extend(results)

    if all_results:
        print_summary(all_results)


if __name__ == '__main__':
    main()
