"""All-band diagnostics, deliberately separating fit quality from flux accuracy."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from .core import BANDS, MorphologyHead, ConstantMorphologyHead, fit_scene


def load_run(folder):
    folder=Path(folder)
    ckpt=torch.load(folder/'head.pt',weights_only=False,map_location='cpu')
    mode=ckpt['args'].get('feature_mode','foundation')
    if mode=='constant':head=ConstantMorphologyHead(ckpt['head_state']['mean_features'])
    else:head=MorphologyHead()
    head.load_state_dict(ckpt['head_state']);head.eval()
    cache=ckpt['metadata'].get('scene_cache') or str(folder/'scenes.pt')
    return head,torch.load(cache,weights_only=False,map_location='cpu')['splits'],ckpt


def measure(folder):
    """Re-evaluate complete catalog, including failed/partial per-band coverage."""
    head,splits,ckpt=load_run(folder)
    mode=ckpt['args'].get('feature_mode','foundation')
    rows=[]
    with torch.no_grad():
        for sid,scene in enumerate(splits['test']):
            for name,model in [('image_moments',None),(mode,head)]:
                cov=scene['covariance'] if model is None else model(scene['features'],scene['covariance'])
                fits=fit_scene(scene,cov)
                for b,fit in fits.items():
                    available={idx:j for j,idx in enumerate(fit['source_indices'].tolist())}
                    for idx,(ra,dec) in enumerate(scene['sky']):
                        j=available.get(idx)
                        frac=float(fit['footprint_fraction'][j]) if j is not None else 0.
                        rows.append(dict(scene=sid,source=idx,ra=ra,dec=dec,band=b,model=name,
                                         central=idx==scene['central'],n_neighbors=len(scene['sky'])-1,
                                         flux=float(fit['flux'][j]) if j is not None else np.nan,
                                         error=float(fit['error'][j]) if j is not None else np.nan,
                                         footprint_fraction=frac,flag=('ILL_CONDITIONED' if float(fit['condition'])>1e5 else 'OK_CONDITIONAL') if frac>.95 else 'PARTIAL_OR_NO_COVERAGE',
                                         condition=float(fit['condition']),
                                         reduced_chi2=float(fit['chi2']/fit['dof'])))
    table=pd.DataFrame(rows)
    table.to_csv(Path(folder)/'test_fluxes_flagged.csv',index=False)
    return table


def running_quantiles(x,y,window=31):
    ok=np.isfinite(x)&np.isfinite(y)
    x,y=np.asarray(x)[ok],np.asarray(y)[ok]
    order=np.argsort(x);x,y=x[order],y[order]
    if len(x)<10:return np.array([]),np.empty((3,0))
    window=min(window,max(7,len(x)//3))
    indices=np.unique(np.linspace(0,len(x)-window,min(40,len(x)-window+1)).astype(int))
    return np.array([np.median(x[i:i+window]) for i in indices]),np.array([np.percentile(y[i:i+window],[16,50,84]) for i in indices]).T


def figures(folder,control_folder):
    folder,control_folder=Path(folder),Path(control_folder)
    summary=json.loads((folder/'summary.json').read_text())['test']
    control=json.loads((control_folder/'summary.json').read_text())['test']
    names=['Image moments','Global correction','Foundation correction']
    values=[summary['before'],control['after'],summary['after']]
    fig,ax=plt.subplots(figsize=(12,4))
    x=np.arange(10)
    for k,(label,v) in enumerate(zip(names,values)):
        ax.bar(x+(k-1)*.25,[v[b] for b in BANDS],width=.25,label=label)
    ax.set_xticks(x);ax.set_xticklabels([b.split('_')[1] for b in BANDS]);ax.set_yscale('log')
    ax.set_ylabel('Held-out scene pixel χ² / degrees of freedom')
    ax.set_title('All bands: image reconstruction (not a flux-accuracy metric)');ax.legend()
    fig.tight_layout();fig.savefig(folder/'all_band_fit_quality.png',dpi=150)
    tables=[measure(folder),measure(control_folder)]
    fig2,axes=plt.subplots(2,5,figsize=(17,7),sharey=True)
    for b,ax in zip(BANDS,axes.flat):
        for tab,mode,color,label in zip(tables,['foundation','constant'],['C0','C1'],['Foundation','Global correction']):
            subset=tab[(tab.band==b)&(tab.flag=='OK_CONDITIONAL')]
            baseline=subset[subset.model=='image_moments'].set_index(['scene','source'])
            model=subset[subset.model==mode].set_index(['scene','source'])
            both=baseline.join(model,lsuffix='_before',rsuffix='_after',how='inner')
            snr=both.flux_before/both.error_before
            selected=snr>5
            delta=(both.flux_after-both.flux_before)/both.flux_before*100
            xx,q=running_quantiles(snr[selected].to_numpy(),delta[selected].to_numpy())
            if len(xx):
                ax.plot(xx,q[1],color=color,label=f'{label} (N={int(selected.sum())})')
                ax.fill_between(xx,q[0],q[2],color=color,alpha=.18)
        ax.axhline(0,color='gray',lw=.8);ax.set_xscale('log');ax.set_title(b);ax.set_xlabel('Image-moment flux / conditional error')
        if ax.get_legend_handles_labels()[0]:ax.legend(fontsize=7)
        else:ax.text(.5,.5,'Too few sources for a running curve',ha='center',va='center',transform=ax.transAxes,fontsize=8)
    for ax in axes[:,0]:ax.set_ylabel('Flux change from image moments (%)')
    fig2.suptitle('Median and central 68%: changes in measured flux, not errors against truth')
    fig2.tight_layout();fig2.savefig(folder/'all_band_flux_changes.png',dpi=150)
    head,splits,_=load_run(folder)
    # Prefer a genuinely crowded scene, never assume the first is crowded.
    scene=max(splits['test'],key=lambda s:len(s['sky']))
    with torch.no_grad():
        before=fit_scene(scene,scene['covariance'])
        after=fit_scene(scene,head(scene['features'],scene['covariance']))
    fig3,axes=plt.subplots(10,3,figsize=(8,24))
    for row,b in enumerate(BANDS):
        d=scene['bands'][b];im=d['image'].numpy();valid=d['mask'].numpy()
        lo,hi=np.percentile(im[valid],[5,99.5])
        axes[row,0].imshow(np.where(valid,im,np.nan),origin='lower',vmin=lo,vmax=hi,cmap='gray')
        axes[row,0].set_ylabel(b)
        for col,fit in enumerate([before[b],after[b]],1):
            residual=(im-fit['model'].numpy())/np.sqrt(d['variance'].numpy())
            axes[row,col].imshow(np.where(valid,residual,np.nan),origin='lower',vmin=-5,vmax=5,cmap='RdBu_r')
        for ax in axes[row]:ax.set_xticks([]);ax.set_yticks([])
    for ax,title in zip(axes[0],['Observed','Before residual / σ','Foundation residual / σ']):ax.set_title(title)
    fig3.suptitle(f"Crowded held-out scene: {len(scene['sky'])} modeled detections; native pixels",y=.999)
    fig3.tight_layout();fig3.savefig(folder/'all_band_scene.png',dpi=120)
    return fig,fig2,fig3
