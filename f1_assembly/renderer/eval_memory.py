#!/usr/bin/env python3
"""Factor-isolated ablation of the conditioning channels. Paired, one factor
at a time, with clustered confidence intervals.

WHY THIS WAS REWRITTEN. The previous version compared full conditioning
against `cond[:, 3:] = 0`. On the nine-channel model that zeroes albedo AND
memory simultaneously, so its "memory" number was the effect of removing two
inputs at once and could not support any claim about memory. Every arm here
changes exactly one factor, and arms that cannot (the combination arms) are
labelled so they never get read as single-factor results.

The training loss cannot answer any of this. It samples a random timestep per
step and epsilon-MSE is dominated by which noise level was drawn, so the number
sits near 0.27 whether the model uses a channel or ignores it.

This removes every source of variance except the factor under test. For each
held-out frame it runs the SAME frame with the SAME noise at the SAME fixed
timesteps under each arm, and reports the difference inside the object mask.

    negative change vs full  ->  removing it HURT, so the model uses it
    change near zero         ->  the model ignores that input
    win rate near 50%        ->  no effect

Channel layout, from Interactions.__getitem__:  0:3 depth, 3:6 albedo,
6:9 memory. The reference photo is separate, via the IP-Adapter embedding.

  python eval_memory.py --cn memory_cn9/controlnet --data ../../f1_manual/traindata
  python eval_memory.py --dry-run          # print the arm matrix, no GPU

Writes eval_factors.json next to the checkpoint: every number in the table
with its interval and the settings that produced it, so a paper table cell can
be traced back to a run.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

# Arm definitions. `zero` lists conditioning channel slices to blank; `ref`
# False drops the reference embedding. `single` marks arms that differ from
# `full` in exactly one factor -- only those support a claim about that factor.
ARMS: dict[str, dict] = {
    'full':      {'zero': [],            'ref': True,  'single': None},
    'no_mem':    {'zero': [(6, 9)],      'ref': True,  'single': 'memory'},
    'no_alb':    {'zero': [(3, 6)],      'ref': True,  'single': 'albedo'},
    'no_depth':  {'zero': [(0, 3)],      'ref': True,  'single': 'depth'},
    'no_ref':    {'zero': [],            'ref': False, 'single': 'reference'},
    # The appearance-source axis of the ablation matrix is covered by the arms
    # above: reference+memory = full, reference only = no_mem, memory only =
    # no_ref. Only the both-off corner needs its own arm, and it moves two
    # factors at once by construction.
    'neither':   {'zero': [(6, 9)],      'ref': False, 'single': None},
}
DEFAULT_ARMS = 'full,no_mem,no_alb,no_ref'


def apply_arm(cond, ie, arm: dict, ch: int):
    """Return (conditioning, image_embeds) for one arm. Never mutates inputs."""
    c = cond.clone()
    for lo, hi in arm['zero']:
        if lo < ch:                       # a 6-channel checkpoint has no 6:9
            c[:, lo:min(hi, ch)] = 0
    e = ie if arm['ref'] else ie * 0
    return c, e


def cluster_bootstrap(diffs: list[float], keys: list[str], n: int = 2000,
                      seed: int = 0) -> tuple[float, float]:
    """95% CI on the mean paired difference, resampling CLUSTERS not frames.

    Neighbouring video frames are near-duplicates, so a naive per-frame
    bootstrap treats correlated samples as independent and reports an interval
    several times too narrow. Resampling whole sequences (and, once there is
    more than one object, whole objects) is what the draft asks for.
    """
    if not diffs:
        return (float('nan'), float('nan'))
    by: dict[str, list[float]] = {}
    for d, k in zip(diffs, keys):
        by.setdefault(k, []).append(d)
    groups = list(by.values())
    if len(groups) < 2:                   # one cluster: no honest interval
        return (float('nan'), float('nan'))
    rng = np.random.default_rng(seed)
    means = np.empty(n)
    idx = np.arange(len(groups))
    for i in range(n):
        pick = rng.choice(idx, size=len(groups), replace=True)
        vals = np.concatenate([groups[j] for j in pick])
        means[i] = vals.mean()
    return (float(np.percentile(means, 2.5)),
            float(np.percentile(means, 97.5)))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cn', help='trained controlnet dir')
    ap.add_argument('--data')
    ap.add_argument('--manifest', default=None)
    ap.add_argument('--sd', default='runwayml/stable-diffusion-v1-5')
    ap.add_argument('--n', type=int, default=48, help='held-out frames')
    ap.add_argument('--res', type=int, default=512)
    ap.add_argument('--timesteps', default='100,300,500,700,900')
    ap.add_argument('--arms', default=DEFAULT_ARMS,
                    help=f'comma-separated; available: {",".join(ARMS)}')
    ap.add_argument('--all-arms', action='store_true')
    ap.add_argument('--boot', type=int, default=2000)
    ap.add_argument('--out', default=None, help='JSON artifact path')
    ap.add_argument('--dry-run', action='store_true',
                    help='print the arm matrix and exit; needs no GPU')
    a = ap.parse_args()

    names = list(ARMS) if a.all_arms else \
        [s.strip() for s in a.arms.split(',') if s.strip()]
    unknown = [n for n in names if n not in ARMS]
    if unknown:
        raise SystemExit(f'unknown arms {unknown}; available: {list(ARMS)}')
    if 'full' not in names:
        names.insert(0, 'full')

    print('[eval] arm matrix')
    print(f'  {"arm":10s} {"zeroed channels":22s} {"ref":>4}  isolates')
    for n in names:
        arm = ARMS[n]
        z = ', '.join(f'{lo}:{hi}' for lo, hi in arm['zero']) or 'none'
        tag = arm['single'] or ('baseline' if n == 'full'
                                else 'COMBINATION - not single-factor')
        print(f'  {n:10s} {z:22s} {str(arm["ref"]):>4}  {tag}')
    if a.dry_run:
        return
    if not a.cn or not a.data:
        raise SystemExit('--cn and --data are required unless --dry-run')

    import torch
    import torch.nn.functional as F
    from diffusers import (ControlNetModel, DDPMScheduler,
                           StableDiffusionControlNetPipeline)
    from train_memory_cn import Interactions, step_prompts

    dev, wdt = 'cuda', torch.bfloat16
    cn = ControlNetModel.from_pretrained(a.cn, torch_dtype=wdt).to(dev).eval()
    ch = int(getattr(cn.config, 'conditioning_channels', 3) or 3)
    print(f'\n[eval] {a.cn}: {ch} conditioning channels')
    if ch < 9:
        print('[eval] WARNING: fewer than 9 channels; memory/albedo arms may '
              'be vacuous on this checkpoint.')
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        a.sd, controlnet=cn, torch_dtype=wdt, safety_checker=None)
    pipe.load_ip_adapter('h94/IP-Adapter', subfolder='models',
                         weight_name='ip-adapter_sd15.bin')
    tok, text = pipe.tokenizer, pipe.text_encoder.to(dev, wdt)
    vae, unet = pipe.vae.to(dev, wdt), pipe.unet.to(dev, wdt)
    img_enc = pipe.image_encoder.to(dev, wdt)
    sched = DDPMScheduler.from_pretrained(a.sd, subfolder='scheduler')

    man = a.manifest or os.path.join(a.data, '..', 'viewer_manifest.json')
    prompts = step_prompts(man)
    with torch.no_grad():
        ids = tok(prompts, padding='max_length', truncation=True,
                  max_length=tok.model_max_length,
                  return_tensors='pt').input_ids.to(dev)
        emb_step = text(ids)[0]

    va = Interactions(a.data, prompts, res=a.res, val=True, mem_drop=0.0,
                      with_pairs=False, mem_gap=1, mem_wrong=0.0)
    idx = [i for i, it in enumerate(va.items) if it['cand']]
    idx = idx[::max(1, len(idx) // a.n)][:a.n]
    ts = [int(x) for x in a.timesteps.split(',')]
    print(f'[eval] {len(idx)} frames x {len(ts)} timesteps x {len(names)} '
          f'arms, paired', flush=True)

    # vals[arm] and clus are aligned: one entry per (frame, timestep)
    vals: dict[str, list[float]] = {n: [] for n in names}
    per_t: dict[int, dict[str, list[float]]] = {
        t: {n: [] for n in names} for t in ts}
    clus: list[str] = []
    t0 = time.time()
    with torch.no_grad():
        for j, i in enumerate(idx):
            b = va[i]
            seq = va.items[i].get('seq', 'unknown')
            px = (b['pixel'][None]).to(dev, wdt)
            lat = (vae.encode(px).latent_dist.sample()
                   * vae.config.scaling_factor).float()
            ie = img_enc(b['ref_clip'][None].to(dev, wdt)).image_embeds
            emb = emb_step[int(b['step'])][None].to(wdt)
            cond = b['cond'][None].to(dev, wdt)
            m = F.max_pool2d(b['mask'][None].to(dev), 9, 1, 4)
            m = (F.interpolate(m, size=lat.shape[-2:],
                               mode='bilinear') > 0.2).float()
            if m.sum() < 1:
                continue
            for t_i in ts:
                # identical noise across arms: the ONLY difference is the arm
                g = torch.Generator(dev).manual_seed(1000 + t_i)
                noise = torch.randn(lat.shape, generator=g, device=dev)
                t = torch.tensor([t_i], device=dev).long()
                noisy = sched.add_noise(lat, noise, t)
                for name in names:
                    c, e = apply_arm(cond, ie, ARMS[name], ch)
                    down, mid = cn(noisy.to(wdt), t,
                                   encoder_hidden_states=emb,
                                   controlnet_cond=c, return_dict=False)
                    pred = unet(noisy.to(wdt), t, encoder_hidden_states=emb,
                                added_cond_kwargs={'image_embeds':
                                                   [e.unsqueeze(1)]},
                                down_block_additional_residuals=list(down),
                                mid_block_additional_residual=mid).sample
                    err = F.mse_loss(pred.float(), noise.float(),
                                     reduction='none')
                    v = float((err * m).sum() / (m.sum() * err.shape[1]))
                    vals[name].append(v)
                    per_t[t_i][name].append(v)
                clus.append(seq)
            if (j + 1) % 10 == 0:
                print(f'  {j+1}/{len(idx)} frames '
                      f'({time.time()-t0:.0f}s)', flush=True)

    n_obs = len(clus)
    base = np.array(vals['full'][:n_obs])
    print(f'\n  object-region epsilon MSE, {n_obs} paired observations, '
          f'{len(set(clus))} clusters')
    print(f'  {"arm":10s} {"MSE":>8} {"vs full":>9} {"95% CI":>18} '
          f'{"worse":>7}  isolates')
    report = {}
    for name in names:
        v = np.array(vals[name][:n_obs])
        d = v - base
        lo, hi = cluster_bootstrap((d / np.maximum(base, 1e-9) * 100).tolist(),
                                   clus, n=a.boot)
        pct = float(d.mean() / max(base.mean(), 1e-9) * 100)
        worse = float((d > 0).mean() * 100)
        ci = 'baseline' if name == 'full' else f'[{lo:+.1f}, {hi:+.1f}]%'
        tag = ARMS[name]['single'] or ('' if name == 'full' else 'COMBINATION')
        print(f'  {name:10s} {v.mean():>8.4f} {pct:>+8.1f}% {ci:>18} '
              f'{worse:>6.0f}%  {tag}')
        report[name] = {'mse': float(v.mean()), 'delta_pct': pct,
                        'ci95_pct': [lo, hi], 'worse_rate_pct': worse,
                        'isolates': ARMS[name]['single'],
                        'single_factor': ARMS[name]['single'] is not None}

    print('\n  Read: a POSITIVE change means removing that input made the '
          'prediction worse,')
    print('  so the model was using it. "worse" is the paired win rate; '
          '50% means no effect.')
    print('  Only rows with a factor named under "isolates" are '
          'single-factor results.')

    art = {'checkpoint': os.path.abspath(a.cn), 'data': os.path.abspath(a.data),
           'channels': ch, 'res': a.res, 'timesteps': ts,
           'n_frames': len(idx), 'n_obs': n_obs,
           'clusters': sorted(set(clus)), 'bootstrap': a.boot,
           'held_out': getattr(va, 'held', None),
           'arms': {k: {kk: vv for kk, vv in ARMS[k].items()} for k in names},
           'results': report,
           'per_timestep': {str(t): {n: float(np.mean(per_t[t][n]))
                                     for n in names} for t in ts},
           'wall_seconds': round(time.time() - t0, 1)}
    out = a.out or os.path.join(os.path.dirname(a.cn.rstrip('/')) or '.',
                                'eval_factors.json')
    with open(out, 'w') as f:
        json.dump(art, f, indent=2)
    print(f'\n  artifact -> {out}')


if __name__ == '__main__':
    main()
