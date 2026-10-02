"""PSF diagnostics and paired, full-population PSF sensitivity report."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from .report import running_quantiles
from .pixel_psf import gaussian_kernel
from .psf_study import load_stamps,choose_kernels,CONDITIONS

BANDS=tuple('euclid_'+b for b in ('VIS','Y','J','H'))
MODES=('tractor_vis','image','foundation')
COLORS=dict(gaussian='0.45',core_gaussian='#d48318',grid='#1878b5')
LABELS=dict(gaussian='Original Gaussian',core_gaussian='Local-FWHM Gaussian',grid='Archive GRID-PSF')
MODEL_LABELS=dict(tractor_vis='Tractor + VIS profiles',image='Mixture + image prior',foundation='Mixture + foundation prior')


def psf_figure(folder):
    data=load_stamps(folder);kernels,cores,_=choose_kernels(data,20261201)
    with np.load(folder/'gaussian/tractor_inputs/scene_0000.npz') as d:
        sigmas={b:float(d[b+'__psf_sigma']) for b in BANDS}
    fig,axes=plt.subplots(2,4,figsize=(15,7))
    for i,b in enumerate(BANDS):
        stamp=kernels[b][0];core=cores[b][0];old=gaussian_kernel(stamp.shape,sigmas[b])
        extent=np.array([-stamp.shape[1]/2,stamp.shape[1]/2,-stamp.shape[0]/2,stamp.shape[0]/2])*.1
        axes[0,i].imshow(stamp,origin='lower',extent=extent,norm=LogNorm(vmin=max(stamp.max()*1e-4,1e-8),vmax=stamp.max()),cmap='magma')
        axes[0,i].set_title(b+' archive PSF');axes[0,i].set_xlabel('Arcsec')
        yy,xx=np.indices(stamp.shape);radius=np.hypot(xx-stamp.shape[1]//2,yy-stamp.shape[0]//2)*.1
        radii=np.linspace(0,max(stamp.shape)*.07,100)
        for name,k in [('gaussian',old),('core_gaussian',core),('grid',stamp)]:
            axes[1,i].plot(radii,[k[radius<=r].sum() for r in radii],color=COLORS[name],label=LABELS[name])
        axes[1,i].set_xlabel('Radius (arcsec)');axes[1,i].set_ylim(0,1.03);axes[1,i].grid(alpha=.2)
    axes[1,0].set_ylabel('Enclosed fraction of normalized finite stamp')
    handles,labels=axes[1,0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=3,bbox_to_anchor=(.5,.95),frameon=False)
    fig.suptitle('Euclid Q1 GRID-PSFs: example local samples and Gaussian approximations')
    fig.tight_layout(rect=(0,0,1,.89));fig.savefig(folder/'psf_profiles.png',dpi=160)
    return fig


def make_report(folder):
    folder=Path(folder);protocol=json.loads((folder/'protocol.json').read_text())
    calibration=json.loads((folder/'image_calibration.json').read_text())
    zero_points={}
    for band in BANDS:
        item=calibration[band]
        if 'native_pixel_verification' not in item:raise ValueError('Verify native pixel units before AB conversion')
        values={float(v) for _,v in item['zero_points']}
        if len(values)!=1:raise ValueError('Missing or ambiguous AB zero point')
        zero_points[band]=values.pop()
    assert not json.loads((folder/'failures.json').read_text())
    mixture=pd.read_csv(folder/'mixture_fluxes.csv');parts=[mixture]
    for c in CONDITIONS:
        target=folder/c/'tractor_comparison'
        assert not json.loads((target/'failures.json').read_text())
        d=pd.read_csv(target/'tractor_fluxes.csv');d['condition']=c;parts.append(d)
    frame=pd.concat(parts,ignore_index=True);keys=['scene','source','band']
    # Require every condition/model to cover exactly the same full population.
    expected=mixture[(mixture.condition=='grid')&(mixture.model=='foundation')].set_index(keys).sort_index()
    assert len(expected)==protocol['count']*2*10
    for (c,m),d in frame.groupby(['condition','model']):
        d=d.set_index(keys).sort_index()
        if not d.index.equals(expected.index):raise ValueError(f'Incomplete sample: {c}, {m}')
        np.testing.assert_allclose(d.truth_flux,expected.truth_flux,rtol=1e-10)
        assert np.isfinite(d.flux).all()
    assert len(frame)==protocol['count']*2*10*len(CONDITIONS)*len(MODES)
    frame.to_csv(folder/'comparison_fluxes.csv',index=False)
    stats=[]
    for (c,m,b),d in frame.groupby(['condition','model','band']):
        e=d.fractional_error.to_numpy()
        stats.append(dict(condition=c,model=m,band=b,n=len(e),bias_percent=100*np.median(e),
            median_absolute_error_percent=100*np.median(abs(e)),nonpositive=int((d.flux<=0).sum())))
    stats=pd.DataFrame(stats);stats.to_csv(folder/'summary.csv',index=False)
    pivot=frame.pivot(index=keys,columns=['condition','model'],values='fractional_error')
    scenes=sorted(frame.scene.unique());rng=np.random.default_rng(39082)
    sample=rng.integers(0,len(scenes),(2000,len(scenes)))
    arrays={(c,m):np.array([pivot.loc[s][c,m].unstack('band').reindex(columns=BANDS).to_numpy() for s in scenes]) for c in CONDITIONS for m in MODES}
    med={key:100*np.median(abs(a),axis=(0,1)) for key,a in arrays.items()}
    boot={key:100*np.median(abs(a[sample]).reshape(2000,-1,4),axis=1) for key,a in arrays.items()}
    rows=[]
    for m in MODES:
        for baseline in ('gaussian','core_gaussian'):
            for b,ix in [(b,[i]) for i,b in enumerate(BANDS)]+[('NISP_YJH',[1,2,3])]:
                delta=(boot['grid',m]-boot[baseline,m])[:,ix].mean(axis=1)
                rows.append(dict(model=m,band=b,baseline=baseline,difference_pp=float((med['grid',m]-med[baseline,m])[ix].mean()),
                    ci_low_pp=float(np.percentile(delta,2.5)),ci_high_pp=float(np.percentile(delta,97.5))))
    paired=pd.DataFrame(rows);paired.to_csv(folder/'paired_psf_effect.csv',index=False)
    figures=[psf_figure(folder)];counts=[]
    for fractional in (False,):
        fig,axes=plt.subplots(3,4,figsize=(16,11),sharey=True)
        for col,b in enumerate(BANDS):
            d=frame[frame.band.eq(b)]
            f=d.pivot(index=['scene','source'],columns=['condition','model'],values='flux')
            true=d.groupby(['scene','source']).truth_flux.first().reindex(f.index)
            abmag=zero_points[b]-2.5*np.log10(true.to_numpy())
            use=(f>0).all(axis=1).to_numpy() & (abmag>=20)
            if not fractional:counts.append(dict(band=b,total=len(f),common_positive=int(use.sum())))
            for row,m in enumerate(MODES):
                ax=axes[row,col]
                for c in CONDITIONS:
                    ratio=f[c,m].to_numpy()[use]/true.to_numpy()[use]
                    y=100*(ratio-1) if fractional else -2.5*np.log10(ratio)
                    x,q=running_quantiles(abmag[use],y,window=51)
                    ax.plot(x,q[1],color=COLORS[c],label=LABELS[c],lw=1.8)
                    ax.fill_between(x,q[0],q[2],color=COLORS[c],alpha=.14)
                ax.axhline(0,color='k',lw=.7,ls=':');ax.grid(alpha=.15)
                ax.set_title(f'{b} · N={use.sum()}/{len(f)}',fontsize=10)
                if row==2:ax.set_xlabel('True AB magnitude')
                if col==0:ax.set_ylabel(MODEL_LABELS[m]+'\n'+('Flux error / true flux (%)' if fractional else 'ΔAB magnitude (measured − true)'))
        handles,labels=axes[0,0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='upper center',ncol=3,bbox_to_anchor=(.5,.965),frameon=False)
        fig.suptitle('Identical archive-PSF simulated images; change only the fitting PSF',y=.995)
        fig.text(.5,.015,'Running median and central 68% distribution. '+('All signed fluxes.' if fractional else 'Common positive subset across all nine method/PSF combinations.'),ha='center')
        fig.tight_layout(rect=(0,.04,1,.93))
        stem='psf_fractional_offset' if fractional else 'psf_magnitude_offset'
        for ext in ('png','pdf'):fig.savefig(folder/f'{stem}.{ext}',dpi=160)
        figures.append(fig)
    pd.DataFrame(counts).to_csv(folder/'magnitude_selection.csv',index=False)
    return stats,paired,figures
