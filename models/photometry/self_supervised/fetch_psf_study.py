"""Retrieve the upstream GRID-PSF products for the local heldout-field sensitivity test."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path('models/photometry/self_supervised/vendor/euclid_forced_photometry/src').resolve()))
from euclid_phot.cutouts import discover_mer_mosaics
from euclid_phot.psf import extract_grid_psf

def main():
    out=Path('models/photometry/self_supervised/runs/psf_study')
    out.mkdir(parents=True,exist_ok=True)
    p=out/'products.json'
    if p.exists():products=json.loads(p.read_text())
    else:
     products=discover_mer_mosaics(53.25555525,-28.06391707,.03)
     p.write_text(json.dumps(products,indent=2))
    for b in ['VIS','Y','J','H']:
     d=extract_grid_psf(b,53.25555525,-28.06391707,products=products,tile_id='102044185',radius_arcsec=30,data_dir=out/'psfs')
     print(b,d['stamps'].shape,d['fwhm'].min(),d['fwhm'].max(),d['s3_path'],flush=True)

if __name__=='__main__':main()
