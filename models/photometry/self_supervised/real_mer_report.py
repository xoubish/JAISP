"""Compare actual Euclid Q1 forced fluxes with matched MER catalog fluxes."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.colors import TwoSlopeNorm
from astropy.visualization import AsinhStretch, ImageNormalize
from .report import running_quantiles

OUT=Path('models/photometry/self_supervised/runs/real_mer')
EUCLID=('euclid_VIS','euclid_Y','euclid_J','euclid_H')
METHODS=('tractor_upstream','image','foundation')
LABELS={'tractor_upstream':'Tractor VIS profile → forced flux',
        'image':'JAISP image prior','foundation':'JAISP foundation prior'}
COLORS={'tractor_upstream':'#7c51a1','image':'#e18420','foundation':'#147ba6'}
REF={'euclid_VIS':('flux_vis_sersic','fluxerr_vis_sersic'),
     'euclid_Y':('flux_y_templfit','fluxerr_y_templfit'),
     'euclid_J':('flux_j_templfit','fluxerr_j_templfit'),
     'euclid_H':('flux_h_templfit','fluxerr_h_templfit')}


def collect():
    rows=[];failures=[];seen=set()
    for folder in sorted(OUT.glob('region_*/references.csv')):
        tractor_path=folder.parent/'tractor_fluxes.csv';mixture_path=folder.parent/'mixture_fluxes.csv'
        if not tractor_path.exists() or not mixture_path.exists():continue
        refs=pd.read_csv(folder)
        tractor=pd.read_csv(tractor_path);mixture=pd.read_csv(mixture_path)
        for tab in (tractor,mixture):
            if tab.duplicated(['region','source','band','model']).any():raise ValueError('Duplicate photometry rows')
        for band in EUCLID:
            fluxcol,errcol=REF[band]
            for i,ref in refs.iterrows():
                oid=int(ref.object_id)
                key=(oid,band)
                if key in seen:raise ValueError(f'Duplicate MER source/band {key}')
                seen.add(key)
                common=dict(region=int(ref.region),source=i,object_id=oid,band=band,
                    reference_flux_ujy=float(ref[fluxcol]),reference_error_ujy=float(ref[errcol]),
                    reference_mag=23.9-2.5*np.log10(ref[fluxcol]) if ref[fluxcol]>0 else np.nan,
                    clean_geometry=bool(ref.clean_geometry),nearest_neighbor_arcsec=float(ref.neighbor_arcsec),
                    is_star=bool(ref.is_star),mer_quality=int(ref.det_quality_flag))
                t=tractor[(tractor.source==i)&(tractor.band==band)]
                if set(t.model)!={'tractor_upstream','tractor_jointsky'}:raise ValueError('Missing Tractor methods')
                m=mixture[(mixture.source==i)&(mixture.band==band)]
                if set(m.model)!={'image','foundation'}:raise ValueError('Missing mixture methods')
                for method,value in zip(METHODS,[t[t.model=='tractor_upstream'].iloc[0],m[m.model=='image'].iloc[0],m[m.model=='foundation'].iloc[0]]):
                    rows.append(dict(**common,model=method,flux_ujy=float(value.flux_ujy),
                        error_ujy=float(value.error_ujy),footprint=float(value.footprint) if 'footprint' in value else 1.))
                joint=t[t.model=='tractor_jointsky'].iloc[0]
                rows.append(dict(**common,model='tractor_jointsky',flux_ujy=float(joint.flux_ujy),
                    error_ujy=float(joint.error_ujy),footprint=1.))
    if not rows:raise ValueError('No matched real photometry yet; run real_mer_tractor and real_mer_mixture')
    d=pd.DataFrame(rows)
    d['delta_ab']=np.where((d.flux_ujy>0)&(d.reference_flux_ujy>0),-2.5*np.log10(d.flux_ujy/d.reference_flux_ujy),np.nan)
    d['measured_snr']=d.flux_ujy/d.error_ujy
    d.to_csv(OUT/'matched_real_photometry.csv',index=False)
    return d


def summarize(d):
    methods=(*METHODS,'tractor_jointsky');rows=[]
    for b in EUCLID:
        for subset,keep in [('all',np.ones(len(d),bool)),('clean',d.clean_geometry.to_numpy()),
            ('isolated',d.nearest_neighbor_arcsec.to_numpy()>=3),('crowded',d.nearest_neighbor_arcsec.to_numpy()<1.5)]:
            for method in methods:
                group=d[(d.band==b)&(d.model==method)&keep]
                x=group[np.isfinite(group.delta_ab)]
                e=x.delta_ab.to_numpy()
                med=np.median(e) if len(e) else np.nan
                rows.append(dict(band=b,subset=subset,model=method,n_reference=group.object_id.nunique(),
                    n=len(e),median_delta_ab=med,
                    nmad_ab=1.4826*np.median(np.abs(e-med)) if len(e) else np.nan,
                    nonpositive_fraction=float((group.flux_ujy<=0).mean()) if len(group) else np.nan,
                    median_reported_snr=float(group.measured_snr.median()) if len(group) else np.nan))
    out=pd.DataFrame(rows);out.to_csv(OUT/'real_flux_summary.csv',index=False);return out


def make_plots(d):
    figures=[]
    methods=METHODS
    fig,axes=plt.subplots(2,2,figsize=(13,9),sharey=True)
    ylimit=1.5
    for band,ax in zip(EUCLID,axes.flat):
        counts=[];clipped=[]
        for method in methods:
            x=d[(d.band==band)&(d.model==method)&np.isfinite(d.delta_ab)].sort_values('reference_mag')
            counts.append(len(x))
            over=x.delta_ab.to_numpy()>ylimit;under=x.delta_ab.to_numpy() < -ylimit
            clipped.append(int(over.sum()+under.sum()))
            inside=~(over|under)
            ax.scatter(x.reference_mag.to_numpy()[inside],x.delta_ab.to_numpy()[inside],
                       color=COLORS[method],s=9,alpha=.15,linewidths=0,rasterized=True)
            if over.any():
                ax.scatter(x.reference_mag.to_numpy()[over],np.full(over.sum(),ylimit*.97),
                           color=COLORS[method],s=22,alpha=.75,marker='^',linewidths=0,rasterized=True)
            if under.any():
                ax.scatter(x.reference_mag.to_numpy()[under],np.full(under.sum(),-ylimit*.97),
                           color=COLORS[method],s=22,alpha=.75,marker='v',linewidths=0,rasterized=True)
            xx,q=running_quantiles(x.reference_mag.to_numpy(),x.delta_ab.to_numpy(),window=21)
            label=LABELS[method]
            if len(xx):
                ax.plot(xx,q[1],color=COLORS[method],lw=2,label=label)
                ax.fill_between(xx,np.clip(q[0],-ylimit,ylimit),np.clip(q[2],-ylimit,ylimit),
                                color=COLORS[method],alpha=.14)
            else:ax.plot([],[],color=COLORS[method],label=label)
        ax.text(.02,.97,
                f'MER refs: {d[d.band==band].object_id.nunique()}  |  positive AB offsets T/I/F: '
                f'{counts[0]}/{counts[1]}/{counts[2]}\n'
                f'edge triangles: >±{ylimit:g} mag ({clipped[0]}/{clipped[1]}/{clipped[2]})',
                transform=ax.transAxes,ha='left',va='top',fontsize=8,
                bbox=dict(facecolor='white',alpha=.78,edgecolor='none',pad=2))
        ax.axhline(0,color='k',lw=.8,ls=':');ax.grid(alpha=.2);ax.set_title(band)
        ax.set_ylim(-ylimit,ylimit)
        ax.set_xlabel('MER catalog AB magnitude (μJy reference)')
    for ax in axes[:,0]:ax.set_ylabel('Measured − MER ΔAB magnitude')
    handles,labels=axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=3,bbox_to_anchor=(.5,.015),fontsize=9)
    fig.suptitle('REAL Euclid Q1 images vs position-matched MER fluxes — all matched sources',y=.995)
    fig.text(.5,.955,'All 629 sources enter the comparison; AB offsets require positive MER and measured fluxes. '
             'Triangle markers show points beyond ±1.5 mag at the plot edges; quantiles use their full values.',
             ha='center',fontsize=9)
    fig.subplots_adjust(left=.08,right=.99,bottom=.09,top=.91,hspace=.28,wspace=.08)
    for ext in ('png','pdf'):fig.savefig(OUT/f'real_mer_delta_ab.png' if ext=='png' else OUT/'real_mer_delta_ab.pdf',dpi=170)
    figures.append(fig)
    # Offset distributions against neighbor distance, retaining the same sources per method.
    fig,axes=plt.subplots(1,4,figsize=(18,4.8),sharey=True)
    for band,ax in zip(EUCLID,axes):
        for method in methods:
            x=d[(d.band==band)&(d.model==method)&np.isfinite(d.delta_ab)]
            xx,q=running_quantiles(x.nearest_neighbor_arcsec.to_numpy(),x.delta_ab.to_numpy(),window=15)
            if len(xx):ax.plot(xx,q[1],color=COLORS[method],lw=1.7,label=method)
        ax.axhline(0,color='k',lw=.8,ls=':');ax.set_title(band);ax.set_xlabel('Nearest MER neighbor (arcsec)');ax.grid(alpha=.15)
    axes[0].set_ylabel('Measured − MER ΔAB magnitude');axes[0].legend(fontsize=7)
    fig.suptitle('Real-image flux offset versus crowding — all matched sources with positive fits')
    fig.tight_layout();fig.savefig(OUT/'real_mer_crowding.png',dpi=170);figures.append(fig)
    figures.append(diagnostic_tile(d))
    figures.append(diagnostic_residuals(d))
    return figures


def diagnostic_tile(d, region=None):
    """Show real image pixels and MER positions for a high-offset VIS region."""
    if region is None:
        vis=d[(d.band=='euclid_VIS')&(d.model=='foundation')&np.isfinite(d.delta_ab)]
        region=int(vis.groupby('region').delta_ab.apply(lambda x:np.median(np.abs(x))).idxmax())
    folder=OUT/f'region_{region:03d}'
    with np.load(folder/'scene.npz') as z:
        scene={k:z[k] for k in z.files}
    refs=pd.read_csv(folder/'references.csv').sort_values('region')
    fig,axes=plt.subplots(2,2,figsize=(13,10))
    cmap=plt.get_cmap('coolwarm');norm=TwoSlopeNorm(vmin=-1.5,vcenter=0,vmax=1.5)
    for band,ax in zip(EUCLID,axes.flat):
        image=scene[band+'__image'].astype(float)
        valid=scene[band+'__mask'].astype(bool)&np.isfinite(image)
        values=image[valid]
        lo,hi=np.percentile(values,[2,99.7]) if len(values) else (0.,1.)
        if not np.isfinite(lo+hi) or hi<=lo:lo,hi=float(np.nanmin(image)),float(np.nanmax(image)+1e-6)
        image_norm=ImageNormalize(vmin=lo,vmax=hi,stretch=AsinhStretch(a=.1),clip=True)
        ax.imshow(np.ma.masked_where(~valid,image),origin='lower',cmap='gray_r',norm=image_norm,
                  interpolation='nearest')
        positions=scene[band+'__positions']
        subset=d[(d.region==region)&(d.band==band)&(d.model=='foundation')].set_index('source')
        for i,ref in refs.iterrows():
            x,y=positions[int(ref.name)]
            value=subset.loc[int(ref.name),'delta_ab'] if int(ref.name) in subset.index else np.nan
            marker='o' if bool(ref.clean_geometry) else 's'
            if np.isfinite(value):
                color=cmap(norm(np.clip(value,-1.5,1.5)))
                ax.scatter([x],[y],s=80,marker=marker,c=[color],edgecolors='black',linewidths=.75,zorder=4)
            else:
                ax.scatter([x],[y],s=75,marker='x',c='#555555',linewidths=1.6,zorder=4)
            ax.annotate(str(int(ref.name)),(x,y),xytext=(3,3),textcoords='offset points',
                        fontsize=8,color='white',zorder=5,
                        path_effects=[path_effects.withStroke(linewidth=1.8,foreground='black')])
        ax.set_title(f'{band}  ({image.shape[1]} × {image.shape[0]} pixels)')
        ax.set_xlim(-.5,image.shape[1]-.5);ax.set_ylim(-.5,image.shape[0]-.5)
        ax.set_xlabel('Native image x (pixels)');ax.set_ylabel('Native image y (pixels)')
    scalar=plt.cm.ScalarMappable(norm=norm,cmap=cmap);scalar.set_array([])
    cbar=fig.colorbar(scalar,cax=fig.add_axes([.925,.20,.018,.53]))
    cbar.set_label('Foundation − MER ΔAB (color clipped at ±1.5 mag)')
    handles=[plt.Line2D([],[],marker='o',linestyle='none',markerfacecolor='white',markeredgecolor='black',label='Clean geometry'),
             plt.Line2D([],[],marker='s',linestyle='none',markerfacecolor='white',markeredgecolor='black',label='Edge / flagged / masked'),
             plt.Line2D([],[],marker='x',linestyle='none',color='#555555',label='No positive AB offset')]
    fig.legend(handles=handles,loc='lower center',ncol=3,bbox_to_anchor=(.48,.015),fontsize=9)
    count=int((refs.neighbor_arcsec<1).sum());bad=int((~refs.clean_geometry).sum())
    fig.suptitle(f'Real Euclid cutout region {region:03d}: MER objects overlaid on all four bands\n'
                 f'Worst median |VIS foundation ΔAB| region; {count} sources have a neighbor within 1″, '
                 f'{bad}/{len(refs)} fail the clean-geometry screen',y=.99)
    fig.subplots_adjust(left=.07,right=.90,bottom=.10,top=.82,wspace=.16,hspace=.24)
    fig.savefig(OUT/'real_mer_diagnostic_tile.png',dpi=180)
    return fig


def diagnostic_residuals(d, region=None):
    """Compare observed VIS pixels with both fitted models in noise units."""
    if region is None:
        vis=d[(d.band=='euclid_VIS')&(d.model=='foundation')&np.isfinite(d.delta_ab)]
        region=int(vis.groupby('region').delta_ab.apply(lambda x:np.median(np.abs(x))).idxmax())
    folder=OUT/f'region_{region:03d}'
    with np.load(folder/'scene.npz') as z:
        image=z['euclid_VIS__image'].astype(float)
        variance=z['euclid_VIS__variance'].astype(float)
        valid=z['euclid_VIS__mask'].astype(bool)&np.isfinite(image)&np.isfinite(variance)&(variance>0)
        positions=z['euclid_VIS__positions']
    with np.load(folder/'tractor_models.npz') as z:tractor=z['euclid_VIS__tractor_upstream'].astype(float)
    with np.load(folder/'mixture_models.npz') as z:foundation=z['euclid_VIS__foundation'].astype(float)
    refs=pd.read_csv(folder/'references.csv')
    models=[('Observed VIS',None),('Tractor residual',tractor),('Foundation residual',foundation)]
    fig,axes=plt.subplots(1,3,figsize=(15,5.3))
    vals=image[valid];lo,hi=np.percentile(vals,[2,99.7])
    im=axes[0].imshow(np.ma.masked_where(~valid,image),origin='lower',cmap='gray_r',
                      norm=ImageNormalize(vmin=lo,vmax=hi,stretch=AsinhStretch(a=.1),clip=True),
                      interpolation='nearest')
    axes[0].set_title('Observed VIS')
    resid_maps=[]
    for ax,(label,model) in zip(axes[1:],models[1:]):
        residual=np.zeros_like(image)
        residual[valid]=(image[valid]-model[valid])/np.sqrt(variance[valid])
        resid_maps.append(residual)
        clipped=np.ma.masked_where(~valid,np.clip(residual,-5,5))
        ax.imshow(clipped,origin='lower',cmap='coolwarm',vmin=-5,vmax=5,interpolation='nearest')
        ax.set_title(f'{label}  (RMS residual / σ = {np.sqrt(np.mean(residual[valid]**2)):.2f})')
    for ax in axes:
        for i,row in refs.iterrows():
            x,y=positions[i]
            ax.scatter([x],[y],s=64,marker='o' if bool(row.clean_geometry) else 's',
                       facecolors='none',edgecolors='black',linewidths=.75,zorder=4)
            ax.annotate(str(i),(x,y),xytext=(3,3),textcoords='offset points',fontsize=8,color='white',
                        path_effects=[path_effects.withStroke(linewidth=1.8,foreground='black')],zorder=5)
        ax.set_xlabel('Native image x (pixels)');ax.set_ylabel('Native image y (pixels)')
        ax.set_xlim(-.5,image.shape[1]-.5);ax.set_ylim(-.5,image.shape[0]-.5)
    cbar=fig.colorbar(plt.cm.ScalarMappable(norm=TwoSlopeNorm(vmin=-5,vcenter=0,vmax=5),cmap='coolwarm'),
                      cax=fig.add_axes([.925,.20,.018,.60]))
    cbar.set_label('(observed − model) / pixel σ, clipped at ±5')
    fig.suptitle(f'Region {region:03d}: actual VIS data and model residuals; labels match the four-band tile',y=.99)
    fig.subplots_adjust(left=.055,right=.90,bottom=.10,top=.84,wspace=.14)
    fig.savefig(OUT/'real_mer_diagnostic_residuals.png',dpi=180)
    return fig


def main():
    d=collect();summary=summarize(d);figures=make_plots(d)
    print('Matched sources:',d.object_id.nunique(),' | matched source-band-method rows:',len(d))
    print(summary[summary.subset=='clean'].pivot(index='band',columns='model',values='n').to_string())
    print(summary[summary.subset=='clean'].pivot(index='band',columns='model',values='median_delta_ab').round(3).to_string())

if __name__=='__main__':main()
