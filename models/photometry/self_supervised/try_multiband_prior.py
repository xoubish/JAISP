"""Prepare, train, and benchmark the expanded multiband morphology experiment.

Run stages in order. Existing checkpoints, simulations and notebook outputs are
never overwritten. Teacher targets are fitted native-pixel profiles, not catalogs.
"""
import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .core import BANDS
from .data import load_tile, make_scenes, wcs
from .mixture import dictionary, ellipse_from_pixels, positive_profile_fit, signed_measurement, fit_multiband
from .multiband_prior import (
    MODES, BandMorphologyPrior, extract_features, fit_band_priors,
    precision_from_errors, profile_error,
)
from .scene_features import SceneEncoder
from .injections import simulate_scene
from .run_mixture import predict_prior

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE = ROOT / 'models/photometry/self_supervised/runs/q1_all_bands'
DEFAULT_LEGACY = ROOT / 'models/photometry/self_supervised/runs/q1_mixture_calibrated'
DEFAULT_OUTPUT = ROOT / 'models/photometry/self_supervised/runs/q1_multiband_expanded'


def save_preparation(out, data, metadata, occupied, processed, failures):
    torch.save(dict(data=data, metadata=metadata, occupied=occupied,
                    processed=processed, failures=failures), out / 'prepared.pt')
    counts = {split: {band: sum(int(x['good'][band].sum()) for x in items)
                     for band in BANDS} for split, items in data.items()}
    report = dict(scenes={k: len(v) for k, v in data.items()}, teacher_counts=counts,
                  processed_tiles={k: len(v) for k, v in processed.items()}, failures=failures)
    (out / 'preparation.json').write_text(json.dumps(report, indent=2))


def teacher(scene, minimum_snr):
    targets, good = {}, {}
    ellipse = ellipse_from_pixels(scene)
    for band in BANDS:
        bank = dictionary(scene, band, ellipse)
        weights, _ = positive_profile_fit(scene['bands'][band], bank)
        measured = signed_measurement(scene['bands'][band], bank, weights)
        targets[band] = weights
        good[band] = (np.isfinite(measured['flux']) & np.isfinite(measured['error'])
                      & (measured['flux'] / measured['error'] >= minimum_snr)
                      & (measured['footprint'] > .98) & (measured['condition'] < 1e5))
    return dict(targets=targets, good=good)


def prepare(args):
    sys.path.insert(0, str(ROOT / 'models'))
    from foundation_utils import discover_tile_pairs
    from .astrometry import FrozenAstrometry

    original = torch.load(args.source / 'scenes.pt', weights_only=False, map_location='cpu')
    original_metadata = original['metadata']
    settings = dict(train_scenes=args.train_scenes, val_scenes=args.val_scenes,
                    scenes_per_tile=args.scenes_per_tile, minimum_snr=args.minimum_snr,
                    seed=args.seed, source=str(args.source.resolve()))
    metadata = dict(original=original_metadata, settings=settings,
        version='band_profile_prior_v1', bands=list(BANDS),
        feature_protocol='Fresh native scene; raw ten-band S/N+coverage 17x17; VIS stem 9x9; bottleneck 5x5',
        teacher='Joint image-only NNLS per band; conditional S/N and footprint cuts; no catalog labels',
        split='Original guarded RA partitions; nonoverlapping neighbor search areas; validation split by whole scene',
        psf='Fixed original training-calibrated Gaussian widths, shared by all methods',
        foundation_pretraining='May include downstream heldout sky; only prior-training sky is separated')
    cache = args.output / 'prepared.pt'
    if cache.exists():
        saved = torch.load(cache, weights_only=False, map_location='cpu')
        if saved['metadata'] != metadata:
            raise ValueError('Preparation settings changed: use another output directory')
        data, occupied, processed, failures = (saved[k] for k in ('data', 'occupied', 'processed', 'failures'))
    else:
        data = dict(train=[], val=[])
        occupied = []
        processed = dict(train=[], val=[])
        failures = []
    labels_payload = torch.load(ROOT / original_metadata['detection_cache'], weights_only=False, map_location='cpu')
    if labels_payload['config']['encoder_ckpt'] != original_metadata['foundation_checkpoint']:
        raise ValueError('Foundation and detection checkpoint provenance disagree')
    labels = labels_payload['labels']
    cache_dir = Path(original_metadata['feature_cache'])
    pairs = discover_tile_pairs(str(ROOT / 'data/rubin_tiles_all'), str(ROOT / 'data/euclid_tiles_all_q1'))
    np.random.default_rng(args.seed).shuffle(pairs)
    candidates = dict(train=[], val=[])
    b1, b2 = original_metadata['split']['ra_boundaries']
    guard = original_metadata['split']['guard_deg']
    for name, rp, ep in pairs:
        if name not in labels or not (cache_dir / (name + '_aug0.pt')).exists():
            continue
        with np.load(ep, allow_pickle=True) as e:
            wc = wcs(e['wcs_VIS'])
            # Some serialized WCS headers omit NAXIS; use the actual array then.
            if wc.pixel_shape is None:
                h, width = e['img_VIS'].shape
            else:
                h, width = int(wc.pixel_shape[1]), int(wc.pixel_shape[0])
            ra, _ = wc.pixel_to_world_values([0, width-1, 0, width-1], [0, 0, h-1, h-1])
        split = 'train' if max(ra) < b1-guard else ('val' if b1+guard < np.mean(ra) < b2-guard else None)
        if split:
            candidates[split].append((name, rp, ep))
    print('Guarded candidate tiles:', {k: len(v) for k, v in candidates.items()}, flush=True)
    astro = FrozenAstrometry(ROOT, original_metadata['astrometry'], original_metadata['foundation_checkpoint'])
    encoder = SceneEncoder(ROOT / original_metadata['foundation_checkpoint'])
    for split, desired in [('train', args.train_scenes), ('val', args.val_scenes)]:
        for pair in candidates[split]:
            if len(data[split]) >= desired:
                break
            if pair[0] in processed[split]:
                continue
            tile = load_tile(pair, labels)
            astro.apply(tile, cache_dir)
            bounds = (-np.inf, b1-guard) if split == 'train' else (b1+guard, b2-guard)
            scenes = make_scenes(tile, original_metadata['psf'], cache_dir,
                                 args.scenes_per_tile, occupied, ra_bounds=bounds)
            for scene in scenes:
                if len(data[split]) >= desired:
                    break
                try:
                    targets = teacher(scene, args.minimum_snr)
                    if not any(g.any() for g in targets['good'].values()):
                        continue
                    features = extract_features(scene, encoder)
                    data[split].append(dict(scene=scene, features=features, **targets))
                except (ValueError, RuntimeError) as exc:
                    failures.append(dict(split=split, tile=pair[0], error=str(exc)))
            processed[split].append(pair[0])
            print(f'{split}: {len(data[split])}/{desired} scenes, {len(processed[split])} tiles', flush=True)
            if len(processed[split]) % 10 == 0:
                save_preparation(args.output, data, metadata, occupied, processed, failures)
        save_preparation(args.output, data, metadata, occupied, processed, failures)
        if len(data[split]) < desired:
            print(f'Guarded {split} sky exhausted: {len(data[split])} usable scenes '
                  f'(requested cap {desired}). Teacher coverage is checked before training.', flush=True)
        if not data[split]:
            raise ValueError(f'No usable {split} scenes')


def combine(items):
    result = dict(features={k: np.concatenate([item['features'][k] for item in items])
                          for k in items[0]['features']},
                targets={b: np.concatenate([item['targets'][b] for item in items]) for b in BANDS},
                good={b: np.concatenate([item['good'][b] for item in items]) for b in BANDS})
    # Outer nuisance sources should not dominate the unsupervised projections.
    keep = np.stack(list(result['good'].values())).any(0)
    return {section: {key: values[keep] for key, values in entries.items()}
            for section, entries in result.items()}


def augment(items, encoder, rng):
    """Preserve bright image-fit targets; degrade all bands or VIS alone."""
    result = list(items)
    for i, item in enumerate(items):
        for scheme in ('all_2', 'vis_4'):
            scene = copy.deepcopy(item['scene'])
            for band, d in scene['bands'].items():
                factor = 2. if scheme == 'all_2' else (4. if band == 'euclid_VIS' else 1.)
                if factor == 1:
                    continue
                valid = d['mask'] & torch.isfinite(d['variance']) & (d['variance'] > 0)
                extra = torch.tensor(rng.normal(size=d['image'].shape), dtype=torch.float32)
                extra *= torch.where(valid, d['variance'], 0).sqrt() * np.sqrt(factor**2 - 1)
                d['image'] = d['image'] + extra
                d['variance'] = d['variance'] * factor**2
            result.append(dict(features=extract_features(scene, encoder), targets=item['targets'],
                               good=item['good'], augmentation=scheme))
        if (i+1) % 10 == 0 or i+1 == len(items):
            print('Feature augmentation', i+1, '/', len(items), flush=True)
    return result


def population_prediction(population, count):
    return {b: np.tile(population[b], (count, 1)) for b in BANDS}


def validation_partitions(items, seed):
    """Balance rare-band coverage while retaining disjoint whole scenes.

    This uses source eligibility counts only, never profile residuals or test
    truth, and is fixed before either prior fitting or uncertainty calibration.
    """
    counts = np.array([[int(item['good'][b].sum()) for b in BANDS] for item in items])
    totals = counts.sum(0).clip(1)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(len(items))
    order = sorted(shuffled, key=lambda i: -np.sum(counts[i]/totals))
    groups, assigned = [[], []], np.zeros((2, len(BANDS)))
    for index in order:
        choices = []
        for group in range(2):
            candidate = assigned.copy()
            candidate[group] += counts[index]
            sizes = [len(groups[j]) + (j == group) for j in range(2)]
            imbalance = np.sum(((candidate[0]-candidate[1])/totals)**2)
            imbalance += .1*((sizes[0]-sizes[1])/len(items))**2
            choices.append(imbalance)
        group = int(np.argmin(choices))
        assigned[group] += counts[index]
        groups[group].append(items[index])
    return groups


def train(args):
    saved = torch.load(args.output / 'prepared.pt', weights_only=False, map_location='cpu')
    data, metadata = saved['data'], saved['metadata']
    tuning, calibration = validation_partitions(data['val'], args.seed)
    independent = {split: {b: sum(int(x['good'][b].sum()) for x in items) for b in BANDS}
                   for split, items in [('train', data['train']), ('tuning', tuning), ('calibration', calibration)]}
    print('Independent teacher counts:', json.dumps(independent), flush=True)
    for split, counts in independent.items():
        minimum = 12 if split == 'train' else 8
        if any(n < minimum for n in counts.values()):
            raise ValueError(f'{split}: insufficient independent teachers; expand preparation before fitting')
    encoder = SceneEncoder(ROOT / metadata['original']['foundation_checkpoint'])
    augmented_file = args.output / 'augmented.pt'
    if augmented_file.exists():
        augmented = torch.load(augmented_file, weights_only=False, map_location='cpu')
    else:
        rng = np.random.default_rng(args.seed + 91)
        augmented = {split: augment(items, encoder, rng)
                     for split, items in [('train', data['train']), ('tuning', tuning), ('calibration', calibration)]}
        torch.save(augmented, augmented_file)
    training, validation, calibrating = (combine(augmented[s]) for s in ('train', 'tuning', 'calibration'))
    population = {b: training['targets'][b][training['good'][b]].mean(0) for b in BANDS}
    heads = {}
    report = {}
    for mode in MODES[1:]:
        head = BandMorphologyPrior().fit(training, validation, mode)
        heads[mode] = head
        prediction = head.predict(calibrating['features'])
        head.precisions = {b: precision_from_errors(prediction[b][calibrating['good'][b]],
                           calibrating['targets'][b][calibrating['good'][b]]) for b in BANDS}
        report[mode] = {b: {k: v for k, v in h.items() if k not in ('beta', 'population', 'trials')}
                        for b, h in head.heads.items()}
        for b in BANDS:
            good = calibrating['good'][b]
            report[mode][b]['calibration_error'] = profile_error(prediction[b][good], calibrating['targets'][b][good])
        print(mode, 'mean tuning profile error', np.mean([h['error'] for h in head.heads.values()]), flush=True)
    count = len(calibrating['features']['vis_image'])
    base = population_prediction(population, count)
    population_precision = {b: precision_from_errors(base[b][calibrating['good'][b]],
                            calibrating['targets'][b][calibrating['good'][b]]) for b in BANDS}
    metadata.update(independent_teacher_counts=independent, noise_training=['original', 'all bands RMS x2', 'VIS RMS x4'],
                    calibration='Independent validation scenes; CDF residual second moments, 20% shrinkage, 3% floor',
                    prior_strength=1., controls=list(MODES), scales_arcsec=[.025,.06,.12,.22,.38,.65,1.1])
    metadata['scene_counts'] = dict(train=len(data['train']), tuning=len(tuning), calibration=len(calibration))
    metadata['head_feature_dimensions'] = {mode: int(head.design(validation['features']).shape[1])
                                           for mode, head in heads.items()}
    if metadata['head_feature_dimensions']['all_images'] != metadata['head_feature_dimensions']['foundation']:
        raise ValueError('Raw ten-band and foundation controls must match latent head capacity')
    metadata['validation_partition'] = dict(method='Whole-scene eligibility-count balance', seed=args.seed,
        tuning_centers=[x['scene']['sky'][x['scene']['central']].tolist() for x in tuning],
        calibration_centers=[x['scene']['sky'][x['scene']['central']].tolist() for x in calibration])
    checkpoint = dict(heads=heads, population=population, population_precision=population_precision, metadata=metadata)
    torch.save(checkpoint, args.output / 'priors.pt')
    (args.output / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    (args.output / 'prior_validation.json').write_text(json.dumps(report, indent=2))


def benchmark(args):
    checkpoint = torch.load(args.output / 'priors.pt', weights_only=False, map_location='cpu')
    legacy = torch.load(args.legacy / 'priors.pt', weights_only=False, map_location='cpu')
    original = torch.load(args.source / 'scenes.pt', weights_only=False, map_location='cpu')
    encoder = SceneEncoder(ROOT / checkpoint['metadata']['original']['foundation_checkpoint'])
    rows, examples = [], []
    modes = (*MODES, 'legacy_foundation')
    templates = original['splits']['test']
    for i in range(args.count):
        seed = args.benchmark_seed + i
        scene, truth, config = simulate_scene(templates[i % len(templates)], seed, weak_vis=(seed // 2) % 2 == 0)
        features = extract_features(scene, encoder)
        priors = {mode: head.predict(features) for mode, head in checkpoint['heads'].items()}
        priors['population'] = population_prediction(checkpoint['population'], len(scene['sky']))
        fits, banks = {}, {}
        for mode in MODES:
            precision = checkpoint['population_precision'] if mode == 'population' else checkpoint['heads'][mode].precisions
            fits[mode] = fit_band_priors(scene, priors[mode], precision, banks=banks)
        from .scene_features import image_features
        # Reuse the same fresh encoder evaluation for the previous 3x3 readout.
        old_foundation = np.concatenate((features['bottleneck'][:, :, 1:4, 1:4],
                                        features['vis_stem'][:, :, 3:6, 3:6]), axis=1)
        old_features = dict(scene=scene, image=image_features(scene), foundation=old_foundation)
        old_prior = predict_prior(old_features, legacy['heads']['foundation'], legacy['population'], 'foundation')
        fits['legacy_foundation'] = fit_multiband(scene, old_prior, banks=banks,
            strength=legacy['metadata']['prior_strength'], band_strength=legacy['metadata']['band_strength'],
            prior_precision=legacy['heads']['foundation']['precision'])
        for mode, measured in fits.items():
            for band in BANDS:
                r = measured[band]
                if len(r['flux']) != 2 or not np.isfinite(r['flux']).all():
                    raise ValueError(f'Incomplete benchmark measurement: {i}/{mode}/{band}')
                for source in range(2):
                    actual = truth[band]['flux'][source]
                    rows.append(dict(scene=i, source=source, model=mode, band=band, truth_flux=actual,
                        flux=r['flux'][source], error=r['error'][source], fractional_error=r['flux'][source]/actual-1,
                        true_snr=truth[band]['snr'][source], **config))
        if i < 2:
            examples.append(dict(scene=scene, truth=truth, config=config, fits=fits))
        if (i+1) % 8 == 0:
            pd.DataFrame(rows).to_csv(args.output / 'injections_partial.csv', index=False)
        if (i+1) % 8 == 0 or i+1 == args.count:
            print('Fresh benchmark', i+1, '/', args.count, flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output / 'injections.csv', index=False)
    (args.output / 'injections_partial.csv').unlink(missing_ok=True)
    torch.save(examples, args.output / 'injection_examples.pt')
    protocol = dict(count=args.count, seed=args.benchmark_seed, models=list(modes),
        selection='All priors and precision matrices locked before any benchmark truth is evaluated',
        inputs='Same pixels, positions, PSFs and native solver; freshly encoded noisy scenes',
        noise='Alternating white and Gaussian-correlated noise; half weak VIS; all signed fluxes retained',
        scope='Independent exponential core/disk synthetic blends with known positions and approximate PSFs; not real survey truth')
    (args.output / 'injection_protocol.json').write_text(json.dumps(protocol, indent=2))
    report(args)


def report(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    frame = pd.read_csv(args.output / 'injections.csv')
    modes = (*MODES, 'legacy_foundation')
    expected = len(frame.scene.unique()) * 2 * len(BANDS) * len(modes)
    if len(frame) != expected or frame.duplicated(['scene', 'source', 'band', 'model']).any():
        raise ValueError('Missing or duplicate paired benchmark rows')
    summaries = []
    for (mode, band), group in frame.groupby(['model', 'band']):
        e = group.fractional_error.to_numpy()
        summaries.append(dict(model=mode, band=band, n=len(e), bias=np.median(e),
            median_absolute_error=np.median(np.abs(e)), nmad=1.4826*np.median(np.abs(e-np.median(e)))))
    stats = pd.DataFrame(summaries)
    stats.to_csv(args.output / 'injection_summary.csv', index=False)
    paired = []
    rng = np.random.default_rng(17291)
    for subset, keep in [('all', np.ones(len(frame), bool)), ('weak_VIS', frame.weak_vis),
                         ('normal_VIS', ~frame.weak_vis), ('white_noise', ~frame.correlated_noise),
                         ('correlated_noise', frame.correlated_noise)]:
        d = frame[keep]
        scenes = sorted(d.scene.unique())
        pivot = d.pivot(index=['scene', 'source', 'band'], columns='model', values='fractional_error')
        arrays = {m: np.array([pivot.loc[s][m].unstack('band').reindex(columns=BANDS).to_numpy()
                              for s in scenes]) for m in modes}
        sample = rng.integers(0, len(scenes), (2000, len(scenes)))
        medians = {m: np.median(np.abs(a), axis=(0,1)) for m, a in arrays.items()}
        boots = {m: np.median(np.abs(a[sample]).reshape(2000, -1, len(BANDS)), axis=1) for m, a in arrays.items()}
        for reference in ('vis_image', 'all_images', 'legacy_foundation', 'population'):
            difference = boots['foundation'] - boots[reference]
            point = medians['foundation'] - medians[reference]
            for j, band in enumerate((*BANDS, 'equal_band_average')):
                values = difference[:, j] if j < len(BANDS) else difference.mean(1)
                effect = point[j] if j < len(BANDS) else point.mean()
                paired.append(dict(subset=subset, reference=reference, band=band, n_blends=len(scenes),
                    difference=effect, ci_low=np.percentile(values,2.5), ci_high=np.percentile(values,97.5)))
    comparison = pd.DataFrame(paired)
    comparison.to_csv(args.output / 'paired_bootstrap.csv', index=False)
    accuracy = 100*stats.pivot(index='band', columns='model', values='median_absolute_error').reindex(BANDS)
    print(accuracy.round(3).to_string(), flush=True)
    print('Equal-band average (%)\n', accuracy.mean().round(3).to_string(), flush=True)
    fig, ax = plt.subplots(figsize=(13, 5))
    labels = dict(population='Population', vis_image='VIS pixels', all_images='Ten-band pixels',
                  foundation='Ten-band pixels + foundation', legacy_foundation='Previous foundation prior')
    x = np.arange(len(BANDS))
    for i, mode in enumerate(modes):
        ax.bar(x+(i-2)*.16, accuracy[mode], width=.16, label=labels[mode])
    ax.set_xticks(x, [b.split('_')[1] for b in BANDS])
    ax.set_ylabel('Median absolute fractional flux error (%)')
    ax.set_title('Fresh known-flux blends: expanded band-specific morphology priors')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output / 'known_flux_accuracy.png', dpi=160)
    fig.savefig(args.output / 'known_flux_accuracy.pdf')
    plt.close(fig)
    from .report import running_quantiles
    from matplotlib.ticker import LogLocator, FuncFormatter, NullFormatter
    colors = dict(population='C2', vis_image='C1', all_images='C4',
                  foundation='C0', legacy_foundation='0.5')
    fig, axes = plt.subplots(2, 5, figsize=(17, 8), sharey=True)
    for band, ax in zip(BANDS, axes.flat):
        for mode in modes:
            group = frame[(frame.band == band) & (frame.model == mode)]
            x, q = running_quantiles(group.true_snr.to_numpy(),
                                     100*group.fractional_error.to_numpy(), window=51)
            ax.plot(x, q[1], color=colors[mode], label=labels[mode],
                    linestyle='--' if mode == 'population' else '-')
            if mode in ('all_images', 'foundation'):
                ax.fill_between(x, q[0], q[2], color=colors[mode], alpha=.12)
        ax.axhline(0, color='k', linewidth=.7, linestyle=':')
        ax.set_xscale('log')
        ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:g}'))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_title(band)
        ax.set_xlabel('True isolated-source S/N')
        ax.grid(alpha=.15)
    for ax in axes[:, 0]:
        ax.set_ylabel('(Measured − true flux) / true flux (%)')
    handles, legend_labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc='upper center', ncol=5, bbox_to_anchor=(.5,.95), fontsize=9)
    fig.suptitle('Fresh known-flux blends: medians and central 68% source distributions', y=.995)
    fig.tight_layout(rect=(0,0,1,.90))
    fig.savefig(args.output / 'known_flux_vs_snr.png', dpi=160)
    fig.savefig(args.output / 'known_flux_vs_snr.pdf')
    plt.close(fig)
    results = dict(equal_band_error_percent=accuracy.mean().to_dict(), count=len(frame.scene.unique()),
        paired_all=comparison[(comparison.subset=='all') & (comparison.band=='equal_band_average')].to_dict('records'))
    (args.output / 'results.json').write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=['prepare', 'train', 'benchmark', 'report'])
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--legacy', type=Path, default=DEFAULT_LEGACY)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--train-scenes', type=int, default=300)
    parser.add_argument('--val-scenes', type=int, default=160)
    parser.add_argument('--scenes-per-tile', type=int, default=6)
    parser.add_argument('--minimum-snr', type=float, default=12.)
    parser.add_argument('--seed', type=int, default=20261002)
    parser.add_argument('--benchmark-seed', type=int, default=202610020)
    parser.add_argument('--count', type=int, default=256)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if args.stage == 'benchmark' and (args.count < 4 or args.count % 4):
        parser.error('--count must be a positive multiple of four to balance VIS/noise subsets')
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    globals()[args.stage](args)


if __name__ == '__main__':
    main()
