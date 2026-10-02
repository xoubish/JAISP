"""Matched Tractor/VIS and foundation photometry plots and paired statistics."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .report import running_quantiles
from .core import BANDS

MODES=('tractor_vis','image','foundation')
LABELS={'tractor_vis':'Tractor + VIS profiles','image':'Mixture + VIS prior','foundation':'Mixture + foundation prior'}
COLORS={'tractor_vis':'#8c4ab3','image':'#e88718','foundation':'#1878b5'}


def make_report(run):
    run=Path(run);out=run/'tractor_comparison'
    if json.loads((out/'failures.json').read_text()):raise ValueError('Audit failed scenes before reporting')
    tractor=pd.read_csv(out/'tractor_fluxes.csv')
    reference=pd.read_csv(run/'injections.csv');reference=reference[reference.model.isin(MODES)]
    # Enforce an identical full test population, not just the successful intersection.
    keys=['scene','source','band']
    expected=reference[reference.model.eq('foundation')].set_index(keys).sort_index()
    actual=tractor.set_index(keys).sort_index()
    if not expected.index.equals(actual.index):raise ValueError('Tractor sample does not match the full saved benchmark')
    np.testing.assert_allclose(actual.truth_flux,expected.truth_flux,rtol=1e-10)
    frame=pd.concat([reference,tractor],ignore_index=True)
    if not np.isfinite(frame[['flux','truth_flux','fractional_error']]).all().all():raise ValueError('Nonfinite measurements')
    frame.to_csv(out/'comparison_fluxes.csv',index=False)
    stats=[]
    for (band,mode),g in frame.groupby(['band','model']):
        e=g.fractional_error.to_numpy();stats.append(dict(band=band,model=mode,n=len(e),
            bias_percent=100*np.median(e),median_absolute_error_percent=100*np.median(abs(e)),
            nonpositive=int((g.flux<=0).sum())))
    stats=pd.DataFrame(stats);stats.to_csv(out/'summary.csv',index=False)
    pivot=frame.pivot(index=keys,columns='model',values='fractional_error')
    scenes=sorted(frame.scene.unique())
    arrays={m:np.array([pivot.loc[s][m].unstack('band').reindex(columns=BANDS).to_numpy() for s in scenes]) for m in MODES}
    rng=np.random.default_rng(57293);sampling=rng.integers(0,len(scenes),(2000,len(scenes)))
    metrics={m:100*np.median(abs(a),axis=(0,1)) for m,a in arrays.items()}
    boot={m:100*np.median(abs(a[sampling]).reshape(2000,-1,10),axis=1) for m,a in arrays.items()}
    comparisons=[]
    for alternative in ('foundation','image'):
        for label,ix in [(b,[i]) for i,b in enumerate(BANDS)]+[('NISP_YJH',[7,8,9]),('all_bands',list(range(10)))]:
            delta=(boot[alternative]-boot['tractor_vis'])[:,ix].mean(axis=1)
            comparisons.append(dict(model=alternative,band=label,
                difference_pp=float((metrics[alternative]-metrics['tractor_vis'])[ix].mean()),
                ci_low_pp=float(np.percentile(delta,2.5)),ci_high_pp=float(np.percentile(delta,97.5))))
    comparison=pd.DataFrame(comparisons);comparison.to_csv(out/'paired_bootstrap.csv',index=False)
    figures=[];counts=[]
    for bands,shape,stem in [(BANDS,(2,5),'all_bands'),(BANDS[6:],(1,4),'euclid')]:
        for fractional in (False,True):
            fig,axes=plt.subplots(*shape,figsize=(18,8) if len(bands)==10 else (16,4.8),sharey=True,squeeze=False)
            for band,ax in zip(bands,axes.flat):
                d=frame[frame.band.eq(band)]
                f=d.pivot(index=['scene','source'],columns='model',values='flux').reindex(columns=MODES)
                true=d.groupby(['scene','source']).truth_flux.first().reindex(f.index)
                use=np.ones(len(f),bool) if fractional else (f>0).all(axis=1).to_numpy()
                if len(bands)==10 and not fractional:counts.append(dict(band=band,total=len(f),common_positive=int(use.sum())))
                mag=-2.5*np.log10(true.to_numpy()[use])
                for mode in MODES:
                    ratio=f[mode].to_numpy()[use]/true.to_numpy()[use]
                    y=100*(ratio-1) if fractional else -2.5*np.log10(ratio)
                    x,q=running_quantiles(mag,y,window=51)
                    ax.plot(x,q[1],color=COLORS[mode],label=LABELS[mode],lw=2)
                    ax.fill_between(x,q[0],q[2],color=COLORS[mode],alpha=.14)
                ax.axhline(0,color='k',lw=.8,ls=':');ax.grid(alpha=.15)
                ax.set_title(f'{band} · N={use.sum()}/{len(f)}',fontsize=11)
                ax.set_xlabel('True instrumental magnitude',fontsize=10)
            for ax in axes[:,0]:ax.set_ylabel('Flux error / true flux (%)' if fractional else 'Measured − true magnitude (mag)')
            handles,labels=axes.flat[0].get_legend_handles_labels()
            fig.legend(handles,labels,loc='upper center',ncol=3,bbox_to_anchor=(.5,.93),frameon=False)
            fig.suptitle('Tractor VIS profiles versus mixture priors — controlled simulated blends',y=.995)
            fig.text(.5,.02,'Running median and central 68% distribution; '+('all signed measurements' if fractional else 'common positive-flux subset')+'. No AB zero points assumed.',ha='center',fontsize=10)
            fig.tight_layout(rect=(0,.06,1,.85))
            name=f'{stem}_{"fractional" if fractional else "magnitude"}_offset'
            for ext in ('png','pdf'):fig.savefig(out/f'{name}.{ext}',dpi=160)
            figures.append(fig)
    pd.DataFrame(counts).to_csv(out/'magnitude_selection.csv',index=False)
    return stats,comparison,figures
