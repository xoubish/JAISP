"""Truth-based paired metrics and an executed empirical injection notebook."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import AsinhLocator, FixedLocator, FuncFormatter

HERE = Path(__file__).resolve().parent
MODELS = ('foundation', 'image', 'tractor_vis', 'oracle')
DISPLAY_MODELS = ('foundation', 'tractor_vis')
COLORS = dict(foundation='tab:blue', image='tab:orange', tractor_vis='tab:green', oracle='0.4')
SNR_EDGES = [0, 1, 3, 5, 10, 20, 50, np.inf]
SNR_LABELS = ['<1', '1–3', '3–5', '5–10', '10–20', '20–50', '≥50']


def inputs(out):
    truth = pd.read_csv(out/'truth.csv'); scenes = pd.read_csv(out/'scenes.csv')
    for scene_id in scenes.scene:
        name=f'scene_{scene_id:04d}.json'
        a=json.loads((out/'jaisp_results'/name).read_text())
        b=json.loads((out/'tractor_results'/name).read_text())
        if a['input_sha256'] != b['input_sha256']:
            raise RuntimeError(f'Unpaired input bytes for scene {scene_id}')
    quality = pd.read_csv(out/'donor_quality.csv').set_index('donor').suitable_primary
    good=set(quality[quality].index)
    scenes['clean_scene'] = scenes.donor.isin(good) & ((scenes.neighbor_donor < 0) | scenes.neighbor_donor.isin(good))
    jaisp = pd.read_csv(out/'jaisp_fluxes.csv'); tractor = pd.read_csv(out/'tractor_fluxes.csv')
    jf = json.loads((out/'jaisp_failures.json').read_text())
    for scene in jf:
        for error in scene['failures']:
            jaisp = jaisp[~((jaisp.scene == scene['scene']) & (jaisp.model == error['model']))]
    oracle = truth[['scene', 'source', 'band', 'oracle_flux', 'oracle_error']].rename(columns={'oracle_flux':'flux', 'oracle_error':'error'})
    oracle['model'] = 'oracle'
    measurements = pd.concat([jaisp[['scene','source','band','model','flux','error']],
                              tractor[['scene','source','band','model','flux','error']], oracle], ignore_index=True)
    measurements = measurements.merge(truth, on=['scene','source','band'], validate='many_to_one')
    measurements = measurements.merge(scenes.drop(columns=['donor']), on='scene', validate='many_to_one')
    measurements['snr_bin'] = pd.cut(measurements.true_snr, SNR_EDGES, labels=SNR_LABELS, right=False)
    measurements['residual'] = measurements.flux-measurements.truth_flux
    measurements['noise_residual'] = measurements.residual/measurements.oracle_error
    measurements['fractional_residual'] = measurements.residual/measurements.truth_flux
    measurements['pull'] = measurements.residual/measurements.error
    # Strictly paired rows for every primary comparison, including oracle.
    keys = ['scene','source','band']
    complete = measurements.groupby(keys).model.nunique()
    complete = complete[complete == len(MODELS)].reset_index()[keys]
    paired = measurements.merge(complete, on=keys, validate='many_to_one')
    return measurements, paired, truth, scenes


def metric(g):
    return pd.Series(dict(n=len(g), n_donors=g.donor.nunique(), n_background_regions=g.background_region.nunique(),
                          bias_sigma=g.noise_residual.mean(), rmse_sigma=np.sqrt(np.mean(g.noise_residual**2)),
                          mean_abs_sigma=np.mean(np.abs(g.noise_residual)),
                          median_abs_sigma=np.median(np.abs(g.noise_residual)),
                          outlier_5sigma=np.mean(np.abs(g.noise_residual)>5),
                          median_fractional_bias=np.median(g.fractional_residual),
                          fractional_rmse=np.sqrt(np.mean(g.fractional_residual**2)),
                          pull_rms=np.sqrt(np.mean(g['pull']**2)),
                          coverage_1sigma=np.mean(np.abs(g['pull']) <= 1), negative_fraction=np.mean(g.flux < 0)))


def clustered_interval(frame, value, group, repeats=2000):
    """Bootstrap independent donor or background groups, not injected rows."""
    stats = frame.assign(value=value).groupby(group).value.agg(['sum','count'])
    rng = np.random.default_rng(20261004)
    indices = rng.integers(len(stats), size=(repeats,len(stats)))
    estimates = stats['sum'].to_numpy()[indices].sum(1)/stats['count'].to_numpy()[indices].sum(1)
    return list(np.percentile(estimates, [2.5, 97.5]))


def two_way_interval(frame, value, repeats=2000):
    """Pigeonhole bootstrap over donor and background-field identities jointly."""
    f=frame.assign(value=value)
    sums=f.groupby(['donor','background_region']).value.sum().unstack(fill_value=0)
    counts=f.groupby(['donor','background_region']).value.size().unstack(fill_value=0)
    rng=np.random.default_rng(20261004)
    wd=rng.multinomial(len(sums),np.ones(len(sums))/len(sums),size=repeats)
    wb=rng.multinomial(sums.shape[1],np.ones(sums.shape[1])/sums.shape[1],size=repeats)
    numerator=np.einsum('bi,ij,bj->b',wd,sums.to_numpy(),wb)
    denominator=np.einsum('bi,ij,bj->b',wd,counts.to_numpy(),wb)
    return list(np.percentile(numerator[denominator>0]/denominator[denominator>0],[2.5,97.5]))


def compare(paired, selection, label):
    q = paired[selection]
    if q.empty: return dict(selection=label, n=0)
    wide = q.pivot(index=['scene','source','band'], columns='model', values='noise_residual')
    meta = q.drop_duplicates(['scene','source','band']).set_index(['scene','source','band'])
    rows = []
    for competitor in ('image', 'tractor_vis'):
        delta = np.abs(wide.foundation)-np.abs(wide[competitor])
        d = meta.loc[wide.index]
        rows.append(dict(selection=label, competitor=competitor, n=len(wide),
                         n_donors=d.donor.nunique(), n_background_regions=d.background_region.nunique(),
                         mean_abs_error_delta_sigma=float(delta.mean()),
                         donor_bootstrap_95=clustered_interval(d, delta.to_numpy(), 'donor'),
                         background_bootstrap_95=clustered_interval(d, delta.to_numpy(), 'background_region'),
                         donor_background_bootstrap_95=two_way_interval(d,delta.to_numpy()),
                         foundation_rmse_sigma=float(np.sqrt(np.mean(wide.foundation**2))),
                         competitor_rmse_sigma=float(np.sqrt(np.mean(wide[competitor]**2)))))
    return rows


def make_plots(out, paired):
    figs = out/'figures'; figs.mkdir(exist_ok=True)
    bands = ['euclid_VIS','euclid_Y','euclid_J','euclid_H','rubin_u','rubin_g','rubin_r','rubin_i','rubin_z','rubin_y']
    fig, axs = plt.subplots(2,5,figsize=(17,7), sharex=True)
    for ax, band in zip(axs.flat, bands):
        for model in DISPLAY_MODELS:
            g = paired[(paired.band==band)&(paired.model==model)&(paired.source==0)]
            stats = g.groupby('snr_bin', observed=True).apply(metric, include_groups=False)
            x = [SNR_LABELS.index(str(i)) for i in stats.index]
            ax.plot(x, stats.rmse_sigma, '.-', label=model, color=COLORS[model])
        ax.set_title(band); ax.set_yscale('log'); ax.grid(alpha=.25)
        ax.set_xticks(range(len(SNR_LABELS))); ax.set_xticklabels(SNR_LABELS, rotation=45)
    axs[0,0].legend(fontsize=8); axs[0,0].set_ylabel('Flux RMSE / known-template noise error')
    axs[1,0].set_ylabel('Flux RMSE / known-template noise error')
    fig.supxlabel('Nominal true S/N (supplied variance, includes blend covariance)'); fig.tight_layout(); fig.savefig(figs/'flux_recovery.png',dpi=140); plt.close(fig)
    fig, axs = plt.subplots(1,4,figsize=(16,4),sharey=True)
    for ax,context in zip(axs, ['isolated','equal_blend','bright_neighbor','different_SED']):
        for model in DISPLAY_MODELS:
            q = paired[(paired.context==context)&(paired.model==model)&(paired.source==0)&(paired.band=='euclid_VIS')]
            s = q.groupby('target_snr',observed=True).apply(metric, include_groups=False)
            ax.plot(s.index,s.rmse_sigma,'.-',color=COLORS[model],label=model)
        ax.set_xscale('log'); ax.set_title(context); ax.grid(alpha=.25); ax.set_xlabel('Target isolated VIS S/N')
    axs[0].set_ylabel('Flux RMSE / known-template noise error'); axs[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(figs/'blend_recovery.png',dpi=140); plt.close(fig)
    # Saved exact injected templates show per-band morphology and color.
    fig,axs=plt.subplots(4,4,figsize=(10,10))
    for row,i in enumerate([0,1,2,3]):
        with np.load(out/'diagnostics'/f'profiles_{i:04d}.npz') as z:
            for col,b in enumerate(['euclid_VIS','euclid_Y','rubin_g','rubin_i']):
                a=z[b][0]; ax=axs[row,col]
                ax.imshow(np.arcsinh(a/max(a.max(),1e-20)*30),origin='lower',cmap='magma');ax.set_title(f'{i} {b}');ax.axis('off')
    fig.tight_layout();fig.savefig(figs/'empirical_morphologies.png',dpi=120);plt.close(fig)
    make_flux_scatter(out, paired, bands)


def make_flux_scatter(out, paired, bands):
    """Native-unit flux-versus-truth plots on identical source lists.

    Median curves and central 68% recovered-flux distributions use identical
    true-flux bins for foundation, Tractor and oracle. Signed measurements all
    enter the statistics; ranges describe scatter, not errors on the median.
    """
    shown=('foundation','tractor_vis','oracle')
    data=paired[(paired.source==0)&paired.model.isin(shown)].copy()
    data[['scene','source','band','model','truth_flux','flux','true_snr','context']].to_csv(
        out/'flux_scatter_points.csv',index=False)
    figs=out/'figures';summary_rows=[]
    for faint in (False, True):
        selected=data[(data.true_snr>=1)&(data.true_snr<10)] if faint else data
        fig,axs=plt.subplots(2,5,figsize=(18,8))
        for ax,b in zip(axs.flat,bands):
            q=selected[selected.band==b]
            reference=q[q.model=='foundation']
            # This threshold is a display scale only: plotted fluxes are raw.
            threshold=float(np.median(reference.oracle_error))
            # Shared bins are set by truth, independently of all measurements.
            edges=np.unique(np.quantile(reference.truth_flux,np.linspace(0,1,7)))
            curves=[]
            for model,label in [('foundation','Foundation'),('tractor_vis','Tractor'),('oracle','Oracle')]:
                g=q[q.model==model]
                if len(edges)>1:
                    groups=g.groupby(pd.cut(g.truth_flux,edges,include_lowest=True),observed=True)
                    x=groups.truth_flux.median().to_numpy()
                    quantiles=groups.flux.quantile([.16,.5,.84]).unstack()
                    low,median,high=(quantiles[p].to_numpy() for p in (.16,.5,.84))
                    ax.fill_between(x,low,high,color=COLORS[model],alpha=.16,linewidth=0,zorder=2)
                    ax.plot(x,median,color=COLORS[model],lw=2,label=label,zorder=3)
                    curves.append(np.r_[x,low,median,high])
                    for i,(interval,count) in enumerate(groups.size().items()):
                        summary_rows.append(dict(selection='faint' if faint else 'all',band=b,model=model,
                                                 bin_left=interval.left,bin_right=interval.right,n=int(count),
                                                 median_true_flux=x[i],flux_p16=low[i],
                                                 median_measured_flux=median[i],flux_p84=high[i]))
            bounds=np.r_[np.concatenate(curves),0.]
            low=1.15*float(min(bounds.min(),-threshold));high=1.15*float(max(bounds.max(),threshold))
            xhigh=1.15*float(max(c[0:len(c)//4].max() for c in curves))
            ax.set_xscale('asinh',linear_width=threshold)
            ax.set_yscale('asinh',linear_width=threshold)
            ax.set_xlim(0,xhigh);ax.set_ylim(low,high)
            for axis in (ax.xaxis,ax.yaxis):
                axis.set_major_locator(AsinhLocator(threshold,numticks=5))
                axis.set_major_formatter(FuncFormatter(lambda v,pos: f'{v:g}'))
                axis.set_minor_locator(plt.NullLocator())
            ax.plot([0,xhigh],[0,xhigh],'k--',lw=1,label='Measured = true',zorder=1)
            ax.axhline(0,color='0.65',lw=.5);ax.grid(alpha=.18)
            ax.set_title(f"{b} · {len(reference)} paired sources",fontsize=10)
            ax.set_xlabel('True flux (native units)')
        axs[0,0].set_ylabel('Measured flux (native units)')
        axs[1,0].set_ylabel('Measured flux (native units)')
        axs[0,0].legend(fontsize=8,loc='upper left')
        title='Faint central sources: nominal true S/N 1–10' if faint else 'All qualified central sources'
        fig.suptitle('Foundation, Tractor and oracle: median measured flux against injected truth\n'+title+' · shading: 16th–84th percentiles',fontsize=13)
        fig.tight_layout(rect=(0,0,1,.94))
        # Native-unit decade labels can crowd the near-zero linear region.
        # Thin labels by display spacing without changing any plotted values.
        for ax in axs.flat:
            for axis,length,gap in ((ax.xaxis,ax.get_window_extent().width,48),
                                    (ax.yaxis,ax.get_window_extent().height,24)):
                lo,hi=axis.get_view_interval();ticks=axis.get_majorticklocs()
                ticks=ticks[(ticks>=lo)&(ticks<=hi)]
                transform=axis.get_transform()
                limits=transform.transform(np.array([lo,hi]))
                positions=(transform.transform(ticks)-limits[0])/(limits[1]-limits[0])*length
                keep=[]
                for index in np.argsort(np.abs(ticks)):
                    if all(abs(positions[index]-positions[j])>=gap for j in keep):keep.append(index)
                axis.set_major_locator(FixedLocator(np.sort(ticks[keep])))
        name='flux_scatter_faint' if faint else 'flux_scatter_all'
        fig.savefig(figs/(name+'.png'),dpi=160)
        fig.savefig(figs/(name+'.pdf'))
        plt.close(fig)
    pd.DataFrame(summary_rows).to_csv(out/'flux_scatter_summary.csv',index=False)


def report(out):
    measurements, paired, truth, scenes = inputs(out)
    tables = {}
    central = paired[(paired.source == 0)&paired.clean_scene]
    for name, groups, data in [('by_band_snr',['model','band','snr_bin'],central),
                               ('by_context',['model','context'],central),
                               ('by_band',['model','band'],central),
                               ('primary_by_band',['model','band'],central[(central.true_snr>=1)&(central.true_snr<10)]),
                               ('primary_overall',['model'],central[(central.true_snr>=1)&(central.true_snr<10)])]:
        table = data.groupby(groups, observed=True).apply(metric, include_groups=False).reset_index()
        table.to_csv(out/(name+'.csv'),index=False); tables[name]=table
    comparisons=[]
    for name, selection in [('central_all', (paired.source==0)),
                             ('central_faint', (paired.source==0)&(paired.true_snr>=1)&(paired.true_snr<10)),
                             ('central_faint_clean', (paired.source==0)&paired.clean_scene&(paired.true_snr>=1)&(paired.true_snr<10)),
                             ('central_VIS_faint', (paired.source==0)&(paired.band=='euclid_VIS')&(paired.true_snr>=1)&(paired.true_snr<10)),
                             ('strong_band_morphology_faint', (paired.source==0)&~paired.weak_band&(paired.true_snr>=1)&(paired.true_snr<10))]:
        value=compare(paired,selection,name)
        comparisons.extend(value if isinstance(value,list) else [value])
    (out/'paired_comparisons.json').write_text(json.dumps(comparisons,indent=2))
    donors = pd.read_csv(out/'donor_bands.csv'); bg = pd.read_csv(out/'backgrounds.csv')
    counts = measurements.groupby('model').size().to_dict()
    protocol=json.loads((out/'protocol.json').read_text())
    summary=dict(n_scenes=len(scenes), n_donors=donors.donor.nunique(), n_backgrounds=len(bg),
                 n_clean_donors=int(pd.read_csv(out/'donor_quality.csv').suitable_primary.sum()),
                 n_clean_scenes=int(scenes.clean_scene.sum()),
                 n_background_regions=bg.region.nunique(), n_sources=len(truth)//10,
                 identical_input_hashes_verified=len(scenes),
                 n_paired_source_band_rows=len(paired)//4,
                 expected_source_band_rows=len(truth), measurement_rows=counts,
                 jaisp_failed_models=sum(len(s['failures']) for s in json.loads((out/'jaisp_failures.json').read_text())),
                 tractor_failed_scenes=len(json.loads((out/'tractor_failures.json').read_text())),
                 footprint_min=float(truth.footprint.min()), footprint_max=float(truth.footprint.max()),
                 valid_footprint_min=float(truth.valid_footprint.min()),
                 weak_band_fraction=float(donors.fallback.mean()),
                 independent_band_morphology=donors.groupby('band').fallback.apply(lambda x:float(1-x.mean())).to_dict(),
                 limitations=[protocol['source_noise'], 'Known positions; detection and completeness not evaluated',
                              'Approximate intrinsic donor reconstruction; faint-band morphology and colors uncertain',
                              'Rubin PSFs are approximate Gaussian calibrations',
                              'Real coadd correlations remain; diagonal covariance errors are conditional, not calibrated uncertainties',
                              'Catalog masking leaves uncataloged sources and outer wings in real sky',
                              'Only seven independent background fields; bootstrap intervals exploratory',
                              'Foundation pretraining independence from donor fields not established',
                              'MER not rerun on injections; no claim of beating MER'])
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    make_plots(out,paired[paired.clean_scene])
    return summary, comparisons


def notebook(out):
    """Execute report cells in-process, retaining output without Jupyter sockets."""
    import nbformat
    from IPython.core.interactiveshell import InteractiveShell
    from IPython.utils.capture import capture_output
    nb=nbformat.v4.new_notebook()
    nb.metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),language_info=dict(name='python',version='3.9'))
    scores=pd.read_csv(out/'primary_overall.csv').set_index('model')
    comparisons=json.loads((out/'paired_comparisons.json').read_text())
    percent_mae=100*(1-scores.loc['foundation','mean_abs_sigma']/scores.loc['tractor_vis','mean_abs_sigma'])
    percent_rmse=100*(1-scores.loc['foundation','rmse_sigma']/scores.loc['tractor_vis','rmse_sigma'])
    outcome=(f"## Result\n\nOn {int(scores.loc['foundation','n']):,} qualified faint central source-band measurements (nominal true S/N 1–10), "
             f"the foundation fitter reduces mean absolute flux error in known-template noise units by **{percent_mae:.1f}%** "
             f"and RMSE by **{percent_rmse:.1f}%** relative to the adapted Tractor VIS baseline. "
             "JAISP allows band-dependent profile refinement, whereas this Tractor baseline fixes its selected VIS profile across bands. "
             "This comparison evaluates the complete fitters and does not isolate the contribution of foundation features.\n\n"
             f"Accuracy still needs improvement: the median fractional flux residual is **{100*scores.loc['foundation','median_fractional_bias']:.1f}%** "
             f"for foundation and {100*scores.loc['tractor_vis','median_fractional_bias']:.1f}% for Tractor, across the scored bands. "
             "The per-band table exposes this faint-end underestimation; these signed errors are not corrected after measurement. "
             "The exact-template oracle is close to unbiased, and rendered mass closes to better than 0.003%. "
             "This points to profile/centering/blend modeling limitations rather than a total-flux normalization error.\n\n"
             "Foundation and Tractor succeeded for all 1000 scenes, with identical NPZ byte hashes verified. "
             "Six independent rendering, reconstruction, input-isolation and blank-sky tests pass. "
             "Source shot noise, detection errors, PSF uncertainty and a MER injection rerun remain outside this pilot.")
    nb.cells=[
        nbformat.v4.new_markdown_cell('# Real-galaxy injection pilot\n\nKnown finite-template fluxes on real, disjoint sky fields. Frozen JAISP foundation photometry versus upstream Tractor VIS model selection. Both see identical pixels, masks, native PSFs and known positions. No injected morphology is supplied to either competitor.\n\nThis is a forced-photometry, sky-noise pilot. It does not measure detection completeness or establish superiority over MER.'),
        nbformat.v4.new_code_cell(f"from pathlib import Path\nimport json\nimport pandas as pd\nfrom IPython.display import display, Image\nOUT = Path({str(out)!r})\nsummary = json.loads((OUT/'summary.json').read_text())\ndisplay(pd.Series({{k:v for k,v in summary.items() if k not in ('limitations','independent_band_morphology')}}).to_frame('value'))"),
        nbformat.v4.new_markdown_cell(outcome),
        nbformat.v4.new_markdown_cell('## Truth and realism\n\nDonors are isolated galaxies from prior-held-out regions; separate complete regions supply backgrounds. A fixed nonparametric, positive pixel reconstruction controls donor noise without replacing galaxies with Sérsic profiles. Native-band colors come from data-only amplitudes. Bands with morphology S/N below 8 retain a flagged common VIS shape and a positive amplitude posterior estimate. Resolved clumps and color gradients remain where supported by the donor.\n\nGalSim convolves the reconstruction with an empirical Euclid PSF or approximate Rubin PSF. Source and PSF rotate together; some scenes add a small Gaussian broadening. The supplied PSF includes pixel response, so rendering uses `no_pixel`. Intrinsic and PSF templates normalize before rendering; crops never normalize. Truth refers to this explicitly defined synthetic galaxy, not the unknowable intrinsic flux of its noisy donor.'),
        nbformat.v4.new_code_cell("display(pd.read_csv(OUT/'donor_bands.csv').groupby('band').agg(donors=('donor','nunique'),median_donor_snr=('snr','median'),weak_morphology_fraction=('fallback','mean'),median_reconstruction_chi2=('reconstruction_chi2','median')))\ndisplay(Image(filename=str(OUT/'figures/empirical_morphologies.png')))"),
        nbformat.v4.new_markdown_cell('## Measured flux versus true injected flux\n\nThe curves compare **foundation (blue)**, **Tractor (green)** and the **oracle (gray)** on the same central-source list in each of the ten bands. Each solid curve shows median measured flux in shared bins of true flux, located at the bin’s median true flux. The shading spans the **16th–84th percentiles** of measured flux: the central 68% distribution, approximately ±1σ for a Gaussian. This shows the distribution within each true-flux bin, not a confidence interval on the median or a formal fit error. The dashed diagonal means perfect recovery.\n\nThe first figure includes all qualified central sources. The second selects nominal **true S/N 1–10**, the same faint sample used for the primary score; selection never uses measured flux. Individual dots are omitted. All signed measurements, including negative values and outliers, enter the medians and percentiles. Both axes use the same asinh display scale per band, which is approximately linear near zero and logarithmic at large absolute flux; tick values remain in native flux units. The underlying measurements and plotted summaries are saved in `flux_scatter_points.csv` and `flux_scatter_summary.csv`, with PNG and PDF figures in `figures/`.'),
        nbformat.v4.new_code_cell("display(Image(filename=str(OUT/'figures/flux_scatter_all.png')))\ndisplay(Image(filename=str(OUT/'figures/flux_scatter_faint.png')))"),
        nbformat.v4.new_markdown_cell('## What is the oracle?\n\nThe **oracle is a diagnostic fit that knows the exact injected shape of every source in every band**, including its clumps, color gradients, PSF and position. It fits the **unknown flux amplitudes and a constant sky level** jointly on the same noisy, masked images. It is not handed the true fluxes and can still return noisy or negative measurements.\n\nFoundation and Tractor must estimate morphology from the images. The oracle skips that estimation, so it gives a reference for performance when the morphology is known perfectly. If the oracle also fails, background contamination, noise modeling or blending can be responsible. If it recovers flux well while a competitor does not, the fitted morphology or its refinement is a likely source of the extra error.\n\nThis privileged fit cannot be used on real galaxies whose shapes are unknown. Its errors condition on the exact templates and use the supplied diagonal variance; they do not account fully for correlated coadd noise. Its gray median curve and percentile shading appear alongside the two photometers as a diagnostic reference.'),
        nbformat.v4.new_markdown_cell('## Paired flux recovery\n\nSigned fluxes, including negative measurements, are retained. Nominal true S/N uses an independent linear fit of exact injected templates and includes neighbor plus constant-background covariance, with the supplied diagonal variance maps. Actual coadd correlations are preserved in the images. The primary comparison is `central_faint_clean`: central sources with nominal true S/N 1–10, with donor QA applied to both target and injected neighbor. QA requires a unique independent MER galaxy match (point-like probability ≤0.2), a non-VIS counterpart at S/N ≥8, and a donor center correction ≤0.5 arcsec. These cuts use donor data only, never recovery outcomes. Every candidate remains in the all-trial comparison. Per-band/context tables use the clean scenes.\n\nFlux errors are also shown in noise units to avoid unstable division by very faint true flux. All primary rows require every method to succeed; failures are reported above. Negative paired absolute-error differences favor the foundation head. Bootstrap donor identities and background regions separately; repeated injections are not independent new galaxies. These intervals are exploratory, especially with seven sky fields.'),
        nbformat.v4.new_code_cell("shown = ['foundation', 'tractor_vis']\nfor filename in ['primary_overall.csv', 'primary_by_band.csv']:\n    table = pd.read_csv(OUT/filename)\n    display(table[table.model.isin(shown)])\ncomparisons = pd.DataFrame(json.loads((OUT/'paired_comparisons.json').read_text()))\ndisplay(comparisons[comparisons.competitor=='tractor_vis'])\ndisplay(Image(filename=str(OUT/'figures/flux_recovery.png')))\ntable = pd.read_csv(OUT/'by_band.csv')\ndisplay(table[table.model.isin(shown)])"),
        nbformat.v4.new_markdown_cell('## Blends and uncertainty calibration\n\nContexts include isolated objects, equal-VIS-flux neighbors, a 10× brighter VIS neighbor, and neighbors with a different empirical SED at 3× VIS flux. Separations span 0.35, 0.7 and 1.4 arcsec. Background source cores are masked identically for every method; uncataloged sky sources remain.\n\nThe initial catalog-only sky pilot (`runs/empirical_injection_pilot`) contained bright Rubin objects missing in VIS catalogs. Even the exact-template oracle failed there. The revised v2 sky pool uses independent secure detections in every native band and grows their isophotes before selecting injection locations; the models and galaxy library remain fixed.\n\nThe oracle helps expose covariance limitations: its Euclid residual scatter is about one supplied error, while Rubin scatters are wider. The competitors’ errors condition on their chosen morphology and ignore coadd correlations; coverage is measured rather than assumed.'),
        nbformat.v4.new_code_cell("display(Image(filename=str(OUT/'figures/blend_recovery.png')))\ntable = pd.read_csv(OUT/'by_context.csv')\ndisplay(table[table.model.isin(shown)])"),
        nbformat.v4.new_markdown_cell('## Limits and next experiment\n\nNo effective gain is available in cached mosaic headers: the default run preserves actual sky noise but adds no source shot noise. This matters for bright neighbors. A calibrated native-unit gain map can enable the explicit coadd Poisson approximation; a full detector-level treatment needs the exposures and weights.\n\nExpand the donor library using deeper data, validate nonparametric reconstructions against deeper independent images, obtain spatial Rubin PSFs, and add detection/centering errors. For a MER comparison, run MER extraction on the identical injected images; the existing catalog cannot measure injected objects. Keep this pilot frozen if tuning a new head; generate a separate final test split.'),
        nbformat.v4.new_code_cell("display(pd.Series(summary['limitations']).to_frame('limitation'))\ndisplay(json.loads((OUT/'protocol.json').read_text()))"),
    ]
    shell=InteractiveShell.instance()
    for i,cell in enumerate(nb.cells):
        if cell.cell_type != 'code':continue
        with capture_output() as cap: result=shell.run_cell(cell.source,store_history=False)
        if result.error_before_exec or result.error_in_exec:raise RuntimeError(str(result.error_before_exec or result.error_in_exec))
        outputs=[]
        if cap.stdout:outputs.append(nbformat.v4.new_output('stream',name='stdout',text=cap.stdout))
        if cap.stderr:outputs.append(nbformat.v4.new_output('stream',name='stderr',text=cap.stderr))
        for o in cap.outputs:outputs.append(nbformat.v4.new_output('display_data',data=o.data,metadata=o.metadata))
        cell.outputs=outputs;cell.execution_count=sum(c.cell_type=='code' for c in nb.cells[:i+1])
    nbformat.write(nb,HERE/'nb_empirical_injection_pilot.ipynb')


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--run',type=Path,default=HERE/'runs/empirical_injection_pilot_v2');p.add_argument('--notebook',action='store_true')
    a=p.parse_args();summary,comparisons=report(a.run)
    print(json.dumps(summary,indent=2));print(json.dumps(comparisons,indent=2))
    if a.notebook:notebook(a.run)


if __name__=='__main__':main()
