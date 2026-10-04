"""Fetch archive MAGZERO values and verify their applicability to the study pixel units."""

def main():
    import json
    from pathlib import Path
    from astropy.io import fits
    p=Path('models/photometry/self_supervised/runs/psf_study')
    products=json.loads((p/'products.json').read_text());result={}
    for band in ['VIS','Y','J','H']:
     product=products[band]['science']
     if product['tile_id']!='102044185':product=next(x for x in product['tiles'] if x['tile_id']=='102044185')
     with fits.open('s3://'+product['s3'],use_fsspec=True,fsspec_kwargs={'anon':True}) as h:
      headers=[dict(x.header) for x in h]
      values=[(i,d['MAGZERO']) for i,d in enumerate(headers) if 'MAGZERO' in d]
      print(band,values,flush=True)
      result['euclid_'+band]=dict(zero_points=values,s3=product['s3'],headers=headers)
    (p/'image_calibration.json').write_text(json.dumps(result,indent=2,default=str))
    
    import json
    from pathlib import Path
    import numpy as np
    from astropy.io import fits
    from astropy.wcs import WCS
    p=Path('models/photometry/self_supervised/runs/psf_study');meta=json.loads((p/'image_calibration.json').read_text())
    tile=Path('data/euclid_tiles_all_q1/tile_x00256_y02304_tract5063_patch_14_euclid.npz')
    with np.load(tile) as d:
     for band in ['VIS','Y','J','H']:
      a=d['img_'+band];y,x=np.array(a.shape)//2
      w=WCS(fits.Header.fromstring(str(d['wcs_'+band]),sep=''))
      ra,dec=w.pixel_to_world_values(x,y)
      m=meta['euclid_'+band]
      with fits.open('s3://'+m['s3'],use_fsspec=True,fsspec_kwargs={'anon':True}) as h:
       xx,yy=WCS(h[0].header).world_to_pixel_values(ra,dec);xx=int(round(float(xx)));yy=int(round(float(yy)))
       original=np.array(h[0].section[yy-4:yy+5,xx-4:xx+5]);cached=a[y-4:y+5,x-4:x+5]
       np.testing.assert_allclose(cached,original,rtol=1e-6,atol=1e-8,equal_nan=True)
       m['native_pixel_verification']=dict(tile=str(tile),pixels=81,rtol=1e-6,atol=1e-8)
       print(band,'native pixel units match archive',flush=True)
    (p/'image_calibration.json').write_text(json.dumps(meta,indent=2,default=str))

if __name__=='__main__':main()
