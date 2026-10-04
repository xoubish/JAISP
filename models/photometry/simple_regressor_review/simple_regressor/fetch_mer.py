"""Fetch Q1 MER photometry from IRSA TAP over an existing catalog footprint.

Queries small RA stripes, checks TAP overflow, deduplicates object IDs and records
ADQL/provenance. The bounding box can include sources outside the image tiles;
prepare performs the actual WCS footprint selection. Public data, no credentials.
"""
import argparse
from datetime import datetime, timezone
from io import BytesIO
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen
import xml.etree.ElementTree as ET

import numpy as np
from astropy.table import Table, vstack, unique

ENDPOINT = 'https://irsa.ipac.caltech.edu/TAP/sync'
COLUMNS = ['object_id', 'ra', 'dec', 'flux_detection_total', 'fluxerr_detection_total',
           'vis_det', 'spurious_flag', 'det_quality_flag']
for n in range(1, 5):
    COLUMNS += [f'flux_vis_{n}fwhm_aper', f'fluxerr_vis_{n}fwhm_aper']


def fetch(footprint, output, stripes=8):
    out = Path(output)
    if out.exists():
        raise FileExistsError(f'{out} already exists; reuse it or choose another output')
    src = Table.read(footprint)
    ra, dec = np.asarray(src['ra']), np.asarray(src['dec'])
    if not len(ra) or not np.isfinite(ra).all() or not np.isfinite(dec).all():
        raise ValueError('Footprint must contain finite sky positions')
    if np.ptp(ra) > 180 or stripes < 1:
        raise ValueError('Use a compact footprint not crossing RA=0 and positive stripes')
    edges = np.linspace(float(ra.min()) - .001, float(ra.max()) + .001, stripes + 1)
    d0, d1 = float(dec.min()) - .001, float(dec.max()) + .001
    tables, queries = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        query = (f"SELECT {','.join(COLUMNS)} FROM euclid_q1_mer_catalogue "
                 f"WHERE ra BETWEEN {lo:.9f} AND {hi:.9f} "
                 f"AND dec BETWEEN {d0:.9f} AND {d1:.9f}")
        params = urlencode(dict(QUERY=query, LANG='ADQL', REQUEST='doQuery',
                                FORMAT='votable', MAXREC=1000000))
        with urlopen(ENDPOINT + '?' + params, timeout=120) as response:
            payload = response.read()
        root = ET.fromstring(payload)
        for item in root.iter():
            if item.tag.rsplit('}', 1)[-1] == 'INFO' and item.get('name') == 'QUERY_STATUS':
                if item.get('value') in ('ERROR', 'OVERFLOW'):
                    raise RuntimeError(f"TAP {item.get('value')}: {item.text}; use more stripes if overflow")
        table = Table.read(BytesIO(payload), format='votable', use_names_over_ids=True)
        for name in list(table.colnames):
            if name != name.lower():
                table.rename_column(name, name.lower())
        tables.append(table); queries.append(query)
        print(f'[fetch] stripe {len(tables)}/{stripes}: {len(table)} rows', flush=True)
    table = unique(vstack(tables), keys='object_id')
    flux = np.asarray(table['flux_detection_total'], dtype=float)
    with np.errstate(invalid='ignore', divide='ignore'):
        table['mag_detection_total'] = np.where(flux > 0, 23.9 - 2.5 * np.log10(flux), np.nan)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.write(out)
    out.with_suffix('.query.json').write_text(json.dumps(dict(
        endpoint=ENDPOINT, table='euclid_q1_mer_catalogue',
        fetched_utc=datetime.now(timezone.utc).isoformat(), footprint=str(Path(footprint).resolve()),
        queries=queries, derived_columns={'mag_detection_total': '23.9 - 2.5*log10(flux_detection_total [microJy])'}, n_rows=len(table)), indent=2))
    print(f'[fetch] wrote {len(table)} unique objects to {out}', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--footprint-catalog', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--stripes', type=int, default=8)
    a = ap.parse_args()
    fetch(a.footprint_catalog, a.output, a.stripes)


if __name__ == '__main__':
    main()
