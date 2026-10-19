"""Test-time optimisation (TTO): refine the amortised shapes of one scene against its own pixels.

The head's prediction (after its learned refinement passes) initialises the free shape tensors
(profile/ellipse/mix vector, modulation map, centroid correction, per-band size/mix/modulation);
Adam then minimises the scene objective used in training's chi-square term, mean_b log(chi2_b/dof_b),
with the signed linear flux solve inside the loop. The parameterisation keeps every bound of the
head (monotone profiles, |log modulation| <= bound, |centroid| <= 0.2"), so this is scarlet-style
per-scene fitting from a learned starting point. Fluxes are still never predicted.
"""
import torch
from ..amortised_scarlet import scene_fit, scene_to
from .model import render, psf_windows, residual_windows, render_points, mix_templates


def amortised_state(model, scene):
    """Run the head exactly as model.run does and return the final free tensors plus the fixed inputs."""
    head = model.head; steps = head.steps
    scene = scene_to(scene, model.device)
    bn, stem, raw = model.encode(scene)
    params, feat = head(bn, stem, raw)
    psf = psf_windows(scene, model.device) if head.psf_input else None
    residual = torch.zeros_like(raw); band_out = None
    for k in range(steps + 1):
        band_out = head.band_branch(feat, raw, residual, psf) if head.per_band else None
        morph, norms = head.band_morphologies(params, band_out)
        templates = render(scene, morph, norms, model.device, convolve=not head.implicit_psf, offsets=head.offsets(params))
        if head.point: templates = mix_templates(templates, render_points(scene, model.device, head.offsets(params)), head.point_weight(params))
        results, total = scene_fit(scene, templates, model.device, fixed_background=True)
        if k == steps: break
        grad, = torch.autograd.grad(total, params['eps'], retain_graph=True)
        g = grad / grad.flatten(1).abs().amax(1).clamp_min(1e-30)[:, None, None, None]
        params = head.refine_step(params, g.detach(), feat, model.refine_scale)
        if head.per_band: residual = residual_windows(scene, results, model.device)
    return scene, params, band_out, results, total


def optimise(model, scene, steps=50, lr=.02, keep_best=True, free=('vector', 'eps', 'centroid', 'point', 'band_eps', 'band_vector'),
             snr_min=0., prior=0., band='euclid_VIS'):
    """Return (results, total, info) after `steps` Adam steps on the scene's shape tensors.

    snr_min: only sources whose amortised flux/error in `band` (or in any band) exceeds this are optimised; the
             others keep the network's shapes (their gradients are zeroed). Stops the free modulation from fitting
             noise around faint sources.
    prior:   weight of a quadratic tether (sum over free tensors of mean (x - x_amortised)^2) toward the start."""
    head = model.head
    scene, params, band_out, results0, total0 = amortised_state(model, scene)
    n = params['vector'].shape[0]; mask = torch.ones(n, dtype=torch.bool, device=params['vector'].device)
    if snr_min > 0:
        snr = torch.zeros(n, device=mask.device)
        for b, r in results0.items():
            snr[r['source_indices']] = torch.maximum(snr[r['source_indices']], (r['flux'] / r['error'].clamp_min(1e-30)).float().detach())
        mask = snr >= snr_min
    leaves = {}
    for k in ('vector', 'eps', 'centroid', 'point'):
        if k in params and k in free: leaves[k] = params[k].detach().clone().requires_grad_(True)
    if band_out is not None:
        for k in ('eps', 'vector'):
            if 'band_' + k in free: leaves['band_' + k] = band_out[k].detach().clone().requires_grad_(True)
    opt = torch.optim.Adam(leaves.values(), lr=lr)
    best = (float(total0), results0, {k: v.detach().clone() for k, v in leaves.items()}); history = [float(total0)]; last = {}

    def forward():
        p = {k: (leaves[k] if k in leaves else params[k].detach()) for k in params}
        b = None if band_out is None else {k: (leaves['band_' + k] if 'band_' + k in leaves else band_out[k].detach()) for k in band_out}
        morph, norms = head.band_morphologies(p, b)
        templates = render(scene, morph, norms, model.device, convolve=not head.implicit_psf, offsets=head.offsets(p))
        if head.point: templates = mix_templates(templates, render_points(scene, model.device, head.offsets(p)), head.point_weight(p))
        last['morph'], last['templates'] = morph, templates
        return scene_fit(scene, templates, model.device, fixed_background=True)

    init = {k: v.detach().clone() for k, v in leaves.items()}
    for i in range(steps):
        opt.zero_grad(); results, total = forward()
        loss = total + (prior * sum(((v - init[k]) ** 2).mean() for k, v in leaves.items()) if prior else 0.)
        loss.backward()
        if snr_min > 0:
            for v in leaves.values(): v.grad[~mask] = 0.
        opt.step(); history.append(float(total))
        if keep_best and float(total) < best[0]: best = (float(total), results, {k: v.detach().clone() for k, v in leaves.items()})
    if keep_best:
        for k, v in best[2].items(): leaves[k].data.copy_(v)
    with torch.no_grad(): pass
    results, total = forward()
    return results, float(total), dict(total0=float(total0), history=history, steps=steps, morph=last['morph'], templates=last['templates'], optimised=mask.detach().cpu().numpy())
