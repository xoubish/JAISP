"""Paired, blend-cluster uncertainty and all-band known-flux plots."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, LogFormatterSciNotation, NullFormatter
from .core import BANDS
from .report import running_quantiles

MODELS=('v1_global','population','image','foundation')
LABELS=dict(v1_global='Old Gaussian',population='Mixture + population prior',
            image='Mixture + VIS prior',foundation='Mixture + foundation prior')
COLORS=dict(v1_global='0.5',population='C2',image='C1',foundation='C0')


def clustered_comparison(frame,draws=1000):
    """Bootstrap entire two-source, ten-band blends, paired between methods."""
    rng=np.random.default_rng(44891);output=[]
    for label,subset in [('all',frame),('weak_VIS',frame[frame.weak_vis]),
                         ('normal_VIS',frame[~frame.weak_vis]),
                         ('white_noise',frame[~frame.correlated_noise]),
                         ('correlated_noise',frame[frame.correlated_noise])]:
        if subset.empty:continue
        pivot=subset.pivot(index=['scene','source','band'],columns='model',values='fractional_error')
        if pivot.isna().any().any():raise ValueError('Missing predictions; audit failures before paired comparisons')
        scenes=sorted(subset.scene.unique())
        arrays={m:np.array([pivot.loc[s][m].unstack('band').reindex(columns=BANDS).to_numpy() for s in scenes]) for m in MODELS}
        sampling=rng.integers(0,len(scenes),(draws,len(scenes)))
        metrics={m:np.median(np.abs(a),axis=(0,1)) for m,a in arrays.items()}
        boot={m:np.median(np.abs(a[sampling]).reshape(draws,-1,len(BANDS)),axis=1) for m,a in arrays.items()}
        for reference in ('v1_global','population','image'):
            delta=boot['foundation']-boot[reference]
            for i,b in enumerate(BANDS):
                output.append(dict(subset=label,band=b,reference=reference,n_blends=len(scenes),
                    foundation_median_abs_error=metrics['foundation'][i],reference_median_abs_error=metrics[reference][i],
                    difference=metrics['foundation'][i]-metrics[reference][i],
                    ci_low=np.percentile(delta[:,i],2.5),ci_high=np.percentile(delta[:,i],97.5)))
            output.append(dict(subset=label,band='equal_band_average',reference=reference,n_blends=len(scenes),
                foundation_median_abs_error=np.mean(metrics['foundation']),reference_median_abs_error=np.mean(metrics[reference]),
                difference=np.mean(metrics['foundation']-metrics[reference]),
                ci_low=np.percentile(delta.mean(1),2.5),ci_high=np.percentile(delta.mean(1),97.5)))
    return pd.DataFrame(output)


def make_report(folder):
    folder=Path(folder);frame=pd.read_csv(folder/'injections.csv')
    counts=frame.groupby(['scene','source','model']).band.nunique()
    assert counts.eq(10).all(),'Every source must have every band'
    comparison=clustered_comparison(frame)
    comparison.to_csv(folder/'paired_bootstrap.csv',index=False)
    figures=[]
    for axis,label,filename in [('truth_flux','True flux (native image units)','known_flux_vs_flux.png'),
                                ('true_snr','True isolated-source S/N','known_flux_vs_snr.png'),
                                ('separation_arcsec','Separation (arcsec)','known_flux_vs_separation.png')]:
        fig,axes=plt.subplots(2,5,figsize=(17,8),sharey=True)
        for b,ax in zip(BANDS,axes.flat):
            for mode in MODELS:
                d=frame[(frame.band==b)&(frame.model==mode)]
                xx,q=running_quantiles(d[axis].to_numpy(),100*d.fractional_error.to_numpy(),window=51)
                ax.plot(xx,q[1],color=COLORS[mode],label=LABELS[mode],lw=1.5,ls='--' if mode=='population' else '-')
                if mode!='population':ax.fill_between(xx,q[0],q[2],color=COLORS[mode],alpha=.10)
            ax.axhline(0,color='k',lw=.7);ax.set_title(b);ax.set_xlabel(label)
            if axis in ('true_snr','truth_flux'):
                ax.set_xscale('log')
                ax.xaxis.set_major_locator(LogLocator(base=10,subs=(1.,2.,5.),numticks=4))
                ax.xaxis.set_major_formatter(LogFormatterSciNotation(labelOnlyBase=False))
                ax.xaxis.set_minor_formatter(NullFormatter())
        for ax in axes[:,0]:ax.set_ylabel('(Measured − true flux) / true flux (%)')
        handles,labels=axes.flat[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='upper center',ncol=4,bbox_to_anchor=(.5,.95))
        fig.suptitle('Known-flux blends: running median and central 68% distribution',y=.995)
        fig.tight_layout(rect=(0,0,1,.90));fig.savefig(folder/filename,dpi=150)
        figures.append(fig)
    stats=pd.read_csv(folder/'injection_summary.csv')
    fig,ax=plt.subplots(figsize=(12,4))
    x=np.arange(10)
    for i,mode in enumerate(MODELS):
        values=stats[stats.model==mode].set_index('band').reindex(BANDS)
        ax.bar(x+(i-1.5)*.2,100*values.median_absolute_error,width=.2,label=LABELS[mode],color=COLORS[mode])
    ax.set_xticks(x);ax.set_xticklabels([b.split('_')[1] for b in BANDS]);ax.set_ylabel('Median absolute fractional flux error (%)')
    ax.legend(fontsize=8);ax.set_title('Known flux recovery in all ten bands');fig.tight_layout()
    fig.savefig(folder/'known_flux_accuracy.png',dpi=150);figures.insert(0,fig)
    return comparison,figures


def magnitude_report(folder, zero_points=None, min_magnitude=None, window=51):
    """Paired magnitude residuals; default x is instrumental, never assumed AB.

    zero_points, if provided, must contain independently verified native-flux
    AB zero points for every band. Negative fluxes remain in the companion
    fractional-flux plot, while all magnitude curves share a positive subset.
    """
    folder=Path(folder)
    frame=pd.read_csv(folder/'injections.csv')
    modes=('v1_global','image','foundation')
    if zero_points is not None and set(zero_points)!=set(BANDS):
        raise ValueError('Supply verified native-flux AB zero points for all ten bands')
    if min_magnitude is not None and zero_points is None:
        raise ValueError('An AB magnitude cut requires verified zero points')
    figures=[];counts=[];curves=[]
    for fractional in (False,True):
        fig,axes=plt.subplots(2,5,figsize=(18,8),sharey=True)
        for band,ax in zip(BANDS,axes.flat):
            d=frame[frame.band.eq(band)&frame.model.isin(modes)]
            flux=d.pivot(index=['scene','source'],columns='model',values='flux').reindex(columns=modes)
            truth=d.groupby(['scene','source']).truth_flux
            if not truth.nunique().eq(1).all():raise ValueError('Inconsistent truth across methods')
            truth=truth.first().reindex(flux.index)
            valid=np.isfinite(truth)&(truth>0)&np.isfinite(flux).all(axis=1)
            if not valid.all():raise ValueError('Missing or nonfinite fluxes; audit before plotting')
            magnitude=(0. if zero_points is None else zero_points[band])-2.5*np.log10(truth)
            selected=valid if min_magnitude is None else valid&(magnitude>=min_magnitude)
            positive=(flux>0).all(axis=1)
            use=selected if fractional else selected&positive
            if not fractional:
                counts.append(dict(band=band,total=int(selected.sum()),common_positive=int((selected&positive).sum()),
                                   excluded_nonpositive=int((selected&~positive).sum())))
            for mode in modes:
                ratio=flux.loc[use,mode]/truth[use]
                residual=100*(ratio-1) if fractional else -2.5*np.log10(ratio)
                xx,q=running_quantiles(magnitude[use].to_numpy(),residual.to_numpy(),window=window)
                ax.plot(xx,q[1],color=COLORS[mode],lw=2,label=LABELS[mode])
                ax.fill_between(xx,q[0],q[2],color=COLORS[mode],alpha=.15)
                for x,lo,med,hi in zip(xx,*q):
                    curves.append(dict(band=band,model=mode,metric='fractional_percent' if fractional else 'magnitude',
                                       magnitude=x,p16=lo,median=med,p84=hi))
            ax.axhline(0,color='k',lw=.8,ls=':');ax.grid(alpha=.15)
            ax.set_title(f'{band}  ·  N={int(use.sum())}/{int(selected.sum())}',fontsize=11)
            ax.set_xlabel('True AB magnitude' if zero_points is not None else r'True instrumental magnitude ($-2.5\log_{10}F$)',fontsize=9)
        for ax in axes[:,0]:
            ax.set_ylabel('100 × (measured − true flux) / true flux' if fractional else r'$m_{\rm measured}-m_{\rm true}$ (mag)')
        handles,labels=axes.flat[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.945),ncol=3,frameon=False)
        fig.suptitle('Controlled simulated blends — running median and central 68% distribution',fontsize=15,y=.99)
        fig.text(.5,.025,'All signed flux measurements retained.' if fractional else
                 'Same positive-flux sources for all three methods. Positive offset = measured too faint.',ha='center',fontsize=10)
        fig.tight_layout(rect=(0,.055,1,.89))
        stem='known_fractional_flux_vs_magnitude' if fractional else 'known_magnitude_offset'
        for suffix in ('png','pdf'):fig.savefig(folder/f'{stem}.{suffix}',dpi=170)
        figures.append(fig)
    counts=pd.DataFrame(counts);counts.to_csv(folder/'magnitude_selection.csv',index=False)
    pd.DataFrame(curves).to_csv(folder/'magnitude_running_quantiles.csv',index=False)
    (folder/'magnitude_plot_protocol.json').write_text(json.dumps(dict(
        axis='instrumental magnitude' if zero_points is None else 'AB magnitude',zero_points=zero_points,
        min_magnitude=min_magnitude,requested_window=window,models=list(modes),
        shading='16th–84th percentile distribution, not uncertainty on the median',
        sample='Controlled simulations; not a survey catalog comparison'),indent=2))
    return counts,figures
