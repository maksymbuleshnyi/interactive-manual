#!/usr/bin/env python3
"""Does the memory picture actually do anything? A paired A/B test.

The training loss cannot answer this. It samples a random timestep per step,
and epsilon-MSE is dominated by which noise level was drawn, so the number sits
around 0.27 whether the model learned to use the memory channels or ignored
them completely.

This removes every source of variance except the one under test. For each
held-out frame it runs the SAME frame twice, with the SAME noise and the SAME
fixed timesteps, once with the memory channels filled and once with them
zeroed, and reports the difference inside the object mask.

    with memory better  ->  the channels are being used
    identical           ->  the model is ignoring them (expected before
                            training: they are zero-initialised)

  python eval_memory.py --cn memory_cn/controlnet --data ../../f1_manual/traindata
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn.functional as F

from train_memory_cn import Interactions, step_prompts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cn', required=True, help='trained controlnet dir')
    ap.add_argument('--data', required=True)
    ap.add_argument('--manifest', default=None)
    ap.add_argument('--sd', default='runwayml/stable-diffusion-v1-5')
    ap.add_argument('--n', type=int, default=48, help='held-out frames')
    ap.add_argument('--res', type=int, default=512)
    ap.add_argument('--timesteps', default='100,300,500,700,900')
    a = ap.parse_args()

    import os
    from diffusers import (ControlNetModel, DDPMScheduler,
                           StableDiffusionControlNetPipeline)
    dev, wdt = 'cuda', torch.bfloat16
    cn = ControlNetModel.from_pretrained(a.cn, torch_dtype=wdt).to(dev).eval()
    ch = int(getattr(cn.config, 'conditioning_channels', 3) or 3)
    print(f'[eval] {a.cn}: {ch} conditioning channels')
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
    print(f'[eval] {len(idx)} frames x {len(ts)} timesteps, paired', flush=True)

    tot = {True: [], False: []}
    per_t = {t: {True: [], False: []} for t in ts}
    with torch.no_grad():
        for j, i in enumerate(idx):
            b = va[i]
            px = (b['pixel'][None]).to(dev, wdt)
            lat = (vae.encode(px).latent_dist.sample()
                   * vae.config.scaling_factor).float()
            ie = img_enc(b['ref_clip'][None].to(dev, wdt)).image_embeds
            emb = emb_step[int(b['step'])][None].to(wdt)
            cond = b['cond'][None].to(dev, wdt)
            blank = cond.clone()
            blank[:, 3:] = 0
            m = F.max_pool2d(b['mask'][None].to(dev), 9, 1, 4)
            m = (F.interpolate(m, size=lat.shape[-2:],
                               mode='bilinear') > 0.2).float()
            if m.sum() < 1:
                continue
            for t_i in ts:
                # identical noise for both arms: the ONLY difference is memory
                g = torch.Generator(dev).manual_seed(1000 + t_i)
                noise = torch.randn(lat.shape, generator=g, device=dev)
                t = torch.tensor([t_i], device=dev).long()
                noisy = sched.add_noise(lat, noise, t)
                for use_mem, c in ((True, cond), (False, blank)):
                    down, mid = cn(noisy.to(wdt), t,
                                   encoder_hidden_states=emb,
                                   controlnet_cond=c, return_dict=False)
                    pred = unet(noisy.to(wdt), t, encoder_hidden_states=emb,
                                added_cond_kwargs={'image_embeds':
                                                   [ie.unsqueeze(1)]},
                                down_block_additional_residuals=list(down),
                                mid_block_additional_residual=mid).sample
                    err = F.mse_loss(pred.float(), noise.float(),
                                     reduction='none')
                    v = float((err * m).sum() / (m.sum() * err.shape[1]))
                    tot[use_mem].append(v)
                    per_t[t_i][use_mem].append(v)
            if (j + 1) % 10 == 0:
                print(f'  {j+1}/{len(idx)} frames', flush=True)

    w, wo = np.mean(tot[True]), np.mean(tot[False])
    print('\n  object-region epsilon MSE on the held-out colourway')
    print(f'  {"timestep":>10} {"no memory":>11} {"with memory":>12} '
          f'{"change":>9}')
    for t_i in ts:
        x, y = np.mean(per_t[t_i][False]), np.mean(per_t[t_i][True])
        print(f'  {t_i:>10} {x:>11.4f} {y:>12.4f} '
              f'{(y-x)/max(x,1e-9)*100:>8.1f}%')
    print(f'  {"ALL":>10} {wo:>11.4f} {w:>12.4f} '
          f'{(w-wo)/max(wo,1e-9)*100:>8.1f}%')
    d = np.array(tot[True]) - np.array(tot[False])
    win = float((d < 0).mean()) * 100
    print(f'\n  memory wins on {win:.0f}% of the paired trials '
          f'(50% = no effect)')
    print('  a negative change means the memory picture is being used')


if __name__ == '__main__':
    main()
