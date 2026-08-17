#!/usr/bin/env python3
"""Teach the existing diffusion stack to use a memory picture.

The model is the one already being served: Stable Diffusion 1.5, frozen, with
a frozen IP-Adapter grounded on a photo of the parts lying loose on the bench.
The only thing trained is the depth ControlNet, whose first layer is widened
from three channels to six:

    channels 0-2   the blockout depth, exactly what the manual page renders
    channels 3-5   the memory picture: what this object looked like a moment
                   ago, drawn in the current view

The new channels start at zero, so before a single step of training the model
behaves exactly as it does today, and training can only add the ability to use
them.

This replaces chaining. Chaining starts the sampler from the previous frame's
latent, so every frame inherits the last one's defects and the look drifts
until a forced purge snaps it back. Here every frame is denoised from pure
noise and the previous frame is only CONDITIONING, so nothing compounds and
nothing has to be purged.

Trained on trajectories, because at serving time the memory is the model's own
output: imperfect, and taken from a slightly different viewpoint. So it is fed
a degraded neighbouring frame here, and at the start of an interaction or
right after a step change it is fed nothing, which is the case it must also
handle.

  python train_memory_cn.py --data ../../f1_manual/traindata --steps 6000
"""
from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

COND_CH = 9          # depth(3) + albedo(3) + memory(3)
BASE = 'a 3D printed plastic Formula One model on a wooden workbench, '
TAIL = 'photo, natural window light, shallow depth of field'
CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)


def step_prompts(manifest_path: str) -> list[str]:
    """The same caption the manual page sends at serving time: the base scene
    plus the role of each part introduced in this step. Built from the same
    manifest the page reads, so train and serve cannot drift apart."""
    m = json.load(open(manifest_path))
    parts, out = m['parts'], []
    for st in m['steps']:
        roles, seen = [], set()
        for n in st.get('new', []):
            r = (parts.get(n) or {}).get('role', '')
            r = r.split('.')[0].strip()
            if r and r not in seen:
                seen.add(r)
                roles.append(r)
        out.append(BASE + (('; '.join(roles) + ', ') if roles else '') + TAIL)
    return out


class Interactions(Dataset):
    """One example = one frame, plus the frame before it in the same
    interaction, which is what the memory channels carry."""

    def __init__(self, root: str, prompts: list[str], res: int = 512,
                 val: bool = False, mem_drop: float = 0.25,
                 with_pairs: bool = True, mem_gap: int = 8,
                 mem_wrong: float = 0.12):
        self.root, self.res, self.mem_drop = root, res, mem_drop
        self.mem_gap, self.mem_wrong = mem_gap, mem_wrong
        self.prompts = prompts
        meta = json.load(open(os.path.join(root, 'trajectories.json')))
        # hold out a whole colourway, never single frames: the question is
        # whether it works on an appearance it has not seen, not whether it
        # can interpolate between two frames of a clip it memorised
        self.held = sorted({s['palette'] for s in meta['seqs']})[0]
        self.items = []
        self.runs = []                  # contiguous frames within one step
        self.by_stem = {}
        for s in meta['seqs']:
            if (s['palette'] == self.held) != val:
                continue
            fr = s['frames']
            cur_run = []
            for f in fr:
                if f.get('cut') and cur_run:
                    self.runs.append(cur_run)
                    cur_run = []
                cur_run.append(f['stem'])
            if cur_run:
                self.runs.append(cur_run)
            run = 0                     # first frame since the last step change
            for i, f in enumerate(fr):
                if f.get('cut') or i == 0:
                    run = i
                # every earlier frame of this step is a candidate memory, not
                # just the one immediately before. At serving time the page
                # hands back whatever it rendered for this viewpoint bucket,
                # which can be many degrees and many seconds stale, and a
                # model trained only on the previous frame smears when the
                # view moves faster than that
                self.items.append({'cur': f['stem'], 'step': f['step'],
                                   'cand': [fr[j]['stem']
                                            for j in range(run, i)],
                                   'ref': s['ref'], 'seq': s['id'],
                                   'other': [fr[j]['stem'] for j in
                                             range(0, len(fr), 11)]})
                self.by_stem[f['stem']] = self.items[-1]
        n_traj = len(self.items)
        # the independent pairs cover far more viewpoints than the 16 clips;
        # they carry no memory, which is exactly the cold-start case
        n_pair = 0
        pf = os.path.join(root, 'manifest.json')
        if with_pairs and os.path.exists(pf):
            for e in json.load(open(pf)):
                if (e['palette'] == self.held) != val:
                    continue
                self.items.append({'cur': e['stem'], 'step': e['step'],
                                   'cand': [], 'other': [], 'ref': e['ref'],
                                   'seq': 'pairs'})
                n_pair += 1
        n_alb = sum(1 for it in self.items[:200] if os.path.exists(
            os.path.join(root, 'alb_' + it['cur'] + '.png')))
        print(f"[data:{'val' if val else 'train'}] {n_traj} trajectory frames "
              f"+ {n_pair} independent frames, held-out colourway "
              f"'{self.held}'", flush=True)
        if not val and n_alb == 0:
            print('[data] WARNING: no alb_*.png found. The albedo channels '
                  'will be zero and colour goes back to being guessed. This '
                  'is an old capture: recapture with the current page.',
                  flush=True)

    def __len__(self):
        return len(self.items)

    def _im(self, name, mode='RGB'):
        im = Image.open(os.path.join(self.root, name + '.png')).convert(mode)
        if im.size != (self.res, self.res):
            im = im.resize((self.res, self.res), Image.BILINEAR)
        return im

    def _arr(self, name, missing_ok=False):
        if missing_ok and not os.path.exists(
                os.path.join(self.root, name + '.png')):
            return np.zeros((self.res, self.res, 3), np.float32)
        return np.asarray(self._im(name), dtype=np.float32) / 255.0

    @staticmethod
    def _degrade(im, res):
        """At serving time the memory is the model's own output, so it is
        never clean, and the viewpoint it was drawn from is never exactly the
        current one. Blur it, add noise, shift it, and knock a hole in it, so
        the model learns to lean on it without trusting it pixel for pixel."""
        if random.random() < 0.6:
            k = random.choice([2, 3])
            im = im.resize((res // k, res // k), Image.BILINEAR) \
                   .resize((res, res), Image.BILINEAR)
        a = np.asarray(im).astype(np.float32) / 255.0
        a = a + np.random.normal(0, random.uniform(0.0, 0.05), a.shape)
        sx, sy = random.randint(-8, 8), random.randint(-8, 8)
        a = np.roll(np.roll(a, sx, axis=1), sy, axis=0)
        if random.random() < 0.25:          # a region it has never seen
            h = res // random.randint(3, 6)
            x, y = random.randint(0, res - h), random.randint(0, res - h)
            a[y:y + h, x:x + h] = 0.0
        return np.clip(a, 0, 1).astype(np.float32)

    def _t(self, arr):
        return torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1)

    def __getitem__(self, i):
        it = self.items[i]
        depth = self._arr('cond_' + it['cur'])
        target = self._arr('tgt_' + it['cur'])
        # flat per-part colour: colour is an input, never a decision
        alb = self._arr('alb_' + it['cur'], missing_ok=True)
        cand = it['cand']
        if not cand or random.random() < self.mem_drop:
            mem = np.zeros_like(depth)                    # cold start
        elif random.random() < self.mem_wrong:
            # memory from somewhere else entirely: the assembly is at a
            # different stage or the camera is nowhere near. The target is
            # still the correct frame, so this teaches the model to believe
            # the depth over the memory, which is what stops the trails
            mem = self._degrade(self._im('tgt_' + random.choice(it['other'])),
                                self.res)
        else:
            # anywhere earlier in this step, not just one frame back, so the
            # viewpoint gap spans what the viewer's memory actually hands back
            k = random.randint(1, min(len(cand), self.mem_gap))
            mem = self._degrade(self._im('tgt_' + cand[-k]), self.res)
        mp = os.path.join(self.root, 'mask_' + it['cur'] + '.png')
        if os.path.exists(mp):
            mask = np.asarray(self._im('mask_' + it['cur'], 'L'),
                              dtype=np.float32)[..., None] / 255.0
        else:                       # no cached mask: weight the frame evenly
            mask = np.ones(depth.shape[:2] + (1,), np.float32)
        ref = np.asarray(self._im(it['ref']).resize((224, 224), Image.BILINEAR)
                         ).astype(np.float32) / 255.0
        return {'cond': torch.cat([self._t(depth), self._t(alb),
                                   self._t(mem)], 0),
                'pixel': self._t(target) * 2 - 1,
                'mask': self._t(mask),
                'step': torch.tensor(int(it['step'])),
                'stem': it['cur'],
                'ref_clip': self._t((ref - CLIP_MEAN) / CLIP_STD)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--manifest', default=None,
                    help='viewer_manifest.json (default: <data>/../)')
    ap.add_argument('--out', default='memory_cn')
    ap.add_argument('--steps', type=int, default=6000)
    ap.add_argument('--bs', type=int, default=4)
    ap.add_argument('--res', type=int, default=512)
    ap.add_argument('--lr', type=float, default=1e-5)
    ap.add_argument('--sd', default='runwayml/stable-diffusion-v1-5')
    ap.add_argument('--cn', default='lllyasviel/control_v11f1p_sd15_depth')
    ap.add_argument('--mem-drop', type=float, default=0.25,
                    help='fraction of frames trained with no memory at all')
    ap.add_argument('--mem-gap', type=int, default=8,
                    help='how far back in the step the memory may come from. '
                         '1 reproduces the old behaviour, where memory was '
                         'always aligned with the answer and the model learned '
                         'to copy it, which is what smears when the view turns')
    ap.add_argument('--mem-wrong', type=float, default=0.12,
                    help='fraction of frames handed deliberately WRONG memory '
                         'while the target stays correct, which teaches it to '
                         'take position from the depth and only colour from '
                         'the memory')
    ap.add_argument('--bg-weight', type=float, default=0.08)
    ap.add_argument('--no-pairs', action='store_true')
    ap.add_argument('--rollout-every', type=int, default=200,
                    help='every N steps, replay part of an interaction with '
                         'the model feeding its OWN output forward as memory; '
                         '0 disables')
    ap.add_argument('--rollout-start', type=int, default=1000,
                    help='no rollouts before this step: early renders are '
                         'noise and teach nothing')
    ap.add_argument('--rollout-k', type=int, default=5,
                    help='frames per rollout')
    ap.add_argument('--rollout-steps', type=int, default=8,
                    help='sampling steps inside a rollout')
    ap.add_argument('--self-p', type=float, default=0.5,
                    help='chance of using self-generated memory when it exists')
    ap.add_argument('--self-cache', type=int, default=3000)
    ap.add_argument('--val-every', type=int, default=500)
    ap.add_argument('--clip-every', type=int, default=1000,
                    help='render a short held-out clip every N steps; 0 off')
    ap.add_argument('--clip-frames', type=int, default=36)
    ap.add_argument('--clip-fps', type=int, default=12)
    ap.add_argument('--clip-steps', type=int, default=8,
                    help='sampling steps per clip frame')
    ap.add_argument('--workers', type=int, default=6)
    a = ap.parse_args()

    from accelerate import Accelerator
    from diffusers import (ControlNetModel, DDPMScheduler,
                           StableDiffusionControlNetPipeline,
                           UniPCMultistepScheduler)
    acc = Accelerator(mixed_precision='bf16')
    dev, wdt = acc.device, torch.bfloat16

    cn = ControlNetModel.from_pretrained(a.cn)
    old = cn.controlnet_cond_embedding.conv_in
    if old.in_channels == 3:
        new = torch.nn.Conv2d(COND_CH, old.out_channels, old.kernel_size,
                              old.stride, old.padding)
        with torch.no_grad():
            new.weight.zero_()
            new.weight[:, :3] = old.weight      # depth behaves as before
            new.bias.copy_(old.bias)            # the rest starts ignored
        cn.controlnet_cond_embedding.conv_in = new
        cn.register_to_config(conditioning_channels=COND_CH)
        print(f'[model] conditioning 3 -> {COND_CH} channels '
              '(depth, albedo, memory), new channels zero-init', flush=True)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        a.sd, controlnet=cn, safety_checker=None)
    pipe.load_ip_adapter('h94/IP-Adapter', subfolder='models',
                         weight_name='ip-adapter_sd15.bin')
    pipe.set_ip_adapter_scale(0.7)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    tok, text = pipe.tokenizer, pipe.text_encoder.to(dev, wdt)
    vae, unet = pipe.vae.to(dev, wdt), pipe.unet.to(dev, wdt)
    img_enc = pipe.image_encoder.to(dev, wdt)
    cn = cn.to(dev)                     # fp32 master weights, autocast forward
    for m in (text, vae, unet, img_enc):
        m.requires_grad_(False)
    cn.requires_grad_(True)
    cn.train()

    sched = DDPMScheduler.from_pretrained(a.sd, subfolder='scheduler')
    T = sched.config.num_train_timesteps
    man = a.manifest or os.path.join(a.data, '..', 'viewer_manifest.json')
    prompts = step_prompts(man)
    print(f'[prompt] {len(prompts)} step captions, e.g. step 1: '
          f'{prompts[1][:110]}...', flush=True)
    with torch.no_grad():                        # one embedding per step
        ids = tok(prompts + [''], padding='max_length',
                  max_length=tok.model_max_length, truncation=True,
                  return_tensors='pt').input_ids.to(dev)
        emb_all = text(ids)[0]
        emb_step, emb_null = emb_all[:-1], emb_all[-1]

    ds = Interactions(a.data, prompts, res=a.res, mem_drop=a.mem_drop,
                      with_pairs=not a.no_pairs, mem_gap=a.mem_gap,
                      mem_wrong=a.mem_wrong)
    va = Interactions(a.data, prompts, res=a.res, val=True, mem_drop=0.0,
                      with_pairs=False, mem_gap=1, mem_wrong=0.0)
    dl = DataLoader(ds, batch_size=a.bs, shuffle=True, num_workers=a.workers,
                    pin_memory=True, drop_last=True,
                    persistent_workers=a.workers > 0)
    opt = torch.optim.AdamW(cn.parameters(), lr=a.lr, weight_decay=1e-2)
    cn, opt, dl = acc.prepare(cn, opt, dl)
    os.makedirs(a.out, exist_ok=True)

    def sample(step_i):
        """A held-out frame drawn twice: once with the memory channels blank,
        once with the real previous frame in them. If the memory is doing
        anything, the second one is closer to the truth."""
        net = acc.unwrap_model(cn)
        net.eval()
        pipe.controlnet = net
        idx = [i for i, it in enumerate(va.items) if it['cand']][:3]
        tiles = []
        with torch.no_grad(), torch.autocast('cuda', wdt):
            for i in idx:
                b = va[i]
                cond = b['cond'][None].to(dev, wdt)
                blank = cond.clone()
                blank[:, 6:] = 0
                pe = emb_step[int(b['step'])][None].to(wdt)
                ne = emb_null[None].to(wdt)
                ip = pipe.prepare_ip_adapter_image_embeds(
                    [Image.fromarray(
                        ((b['ref_clip'].permute(1, 2, 0).numpy() * CLIP_STD
                          + CLIP_MEAN) * 255).clip(0, 255).astype(np.uint8))],
                    None, torch.device(dev), 1, True)
                row = []
                for c in (blank, cond):
                    g = torch.Generator(dev).manual_seed(0)
                    row.append(pipe(
                        prompt_embeds=pe, negative_prompt_embeds=ne,
                        ip_adapter_image_embeds=ip, image=c,
                        num_inference_steps=18, guidance_scale=5.0,
                        generator=g, controlnet_conditioning_scale=1.15
                    ).images[0])
                mem = (b['cond'][6:].permute(1, 2, 0).numpy() * 255)
                tgt = ((b['pixel'].permute(1, 2, 0).numpy() + 1) * 127.5)
                tiles.append([Image.fromarray(mem.astype(np.uint8)),
                              row[0], row[1],
                              Image.fromarray(tgt.astype(np.uint8))])
        w = a.res
        strip = Image.new('RGB', (w * 4, w * len(tiles)))
        for r, row in enumerate(tiles):
            for c, im in enumerate(row):
                strip.paste(im.resize((w, w)), (c * w, r * w))
        strip.save(os.path.join(a.out, f'val_{step_i:06d}.jpg'), quality=90)
        net.train()
        print(f'[val] {a.out}/val_{step_i:06d}.jpg  columns: memory in, '
              f'output without memory, output with memory, truth', flush=True)

    # ---- self-generated memory -------------------------------------------
    # Training has been feeding the model CAPTURED frames as memory: clean,
    # correct, perfectly consistent with the target. At serving it gets its own
    # render, with its own mistakes, and trusts it the same way. That mismatch
    # is why memory made things worse rather than better. So every so often,
    # replay a stretch of an interaction with the model rendering each frame
    # and passing its OWN output forward, and train on those.
    self_mem = {}

    def rollout():
        net = acc.unwrap_model(cn)
        net.eval()
        pipe.controlnet = net
        run = random.choice(ds.runs)[:a.rollout_k]
        it0 = ds.by_stem[run[0]]
        ref = ds._im(it0['ref']).resize((224, 224), Image.BILINEAR)
        mem = None
        with torch.no_grad(), torch.autocast('cuda', wdt):
            for st in run:
                it = ds.by_stem[st]
                cond = torch.cat([
                    ds._t(ds._arr('cond_' + st)),
                    ds._t(ds._arr('alb_' + st, missing_ok=True)),
                    ds._t(mem) if mem is not None
                    else torch.zeros(3, a.res, a.res)], 0)[None].to(dev, wdt)
                img = pipe(prompts[int(it['step'])], image=cond,
                           ip_adapter_image=ref, num_inference_steps=
                           a.rollout_steps, guidance_scale=1.0,
                           controlnet_conditioning_scale=1.15).images[0]
                mem = np.asarray(img).astype(np.float32) / 255.0
                self_mem[st] = (mem * 255).astype(np.uint8)
        if len(self_mem) > a.self_cache:          # bound the memory footprint
            for k in list(self_mem)[:len(self_mem) - a.self_cache]:
                self_mem.pop(k, None)
        net.train()

    # ---- periodic clip ----------------------------------------------------
    # A still says nothing about flicker, trails or drift, which are the whole
    # problem. So every checkpoint renders a few seconds of a HELD-OUT
    # interaction the way it will actually be served: each frame's memory is
    # the model's own previous output. Same frames every time, so the clips are
    # comparable across checkpoints.
    clip_starts = {r[0] for r in va.runs}
    clip_items = va.items[:a.clip_frames]

    def clip(step_i):
        import subprocess
        net = acc.unwrap_model(cn)
        net.eval()
        pipe.controlnet = net
        d = os.path.join(a.out, f'clip_{step_i:06d}')
        os.makedirs(d, exist_ok=True)
        ref = va._im(clip_items[0]['ref']).resize((224, 224), Image.BILINEAR)
        mem, panels = None, []
        with torch.no_grad(), torch.autocast('cuda', wdt):
            for n, it in enumerate(clip_items):
                st = it['cur']
                if st in clip_starts:          # new part: forget, as when served
                    mem = None
                depth = va._im('cond_' + st)
                cond = torch.cat([
                    va._t(va._arr('cond_' + st)),
                    va._t(va._arr('alb_' + st, missing_ok=True)),
                    va._t(mem) if mem is not None
                    else torch.zeros(3, a.res, a.res)], 0)[None].to(dev, wdt)
                g = torch.Generator(dev).manual_seed(0)
                img = pipe(prompts[int(it['step'])], image=cond,
                           ip_adapter_image=ref, generator=g,
                           num_inference_steps=a.clip_steps,
                           guidance_scale=1.0,
                           controlnet_conditioning_scale=1.15).images[0]
                mem = np.asarray(img).astype(np.float32) / 255.0
                p = Image.new('RGB', (a.res * 3, a.res))
                p.paste(depth, (0, 0))
                p.paste(img.resize((a.res, a.res)), (a.res, 0))
                p.paste(va._im('tgt_' + st), (a.res * 2, 0))
                p.save(os.path.join(d, f'f{n:04d}.png'))
                panels.append(p)
        mp4 = os.path.join(a.out, f'clip_{step_i:06d}.mp4')
        try:
            subprocess.run(['ffmpeg', '-y', '-framerate', str(a.clip_fps),
                            '-i', os.path.join(d, 'f%04d.png'), '-c:v',
                            'libx264', '-pix_fmt', 'yuv420p', '-crf', '18',
                            mp4], check=True, capture_output=True)
            for f in os.listdir(d):
                os.remove(os.path.join(d, f))
            os.rmdir(d)
            out = mp4
        except Exception:                       # no ffmpeg: a GIF still plays
            out = os.path.join(a.out, f'clip_{step_i:06d}.gif')
            panels[0].save(out, save_all=True, append_images=panels[1:],
                           duration=int(1000 / a.clip_fps), loop=0)
        net.train()
        print(f'[clip] {out}  ({len(panels)} frames, held-out colourway, '
              'blockout | neural | truth)', flush=True)

    print(f'[train] {len(ds)} frames, {len(ds.runs)} step-runs, '
          f'bs {a.bs}, {a.steps} steps', flush=True)
    step, ema = 0, None
    while step < a.steps:
        for b in dl:
            if step >= a.steps:
                break
            px = b['pixel'].to(dev, wdt)
            cond = b['cond'].to(dev, wdt)
            mask = b['mask'].to(dev)
            # swap in the model's own renders where a rollout produced one
            if self_mem and a.self_p > 0:
                for bi, stm in enumerate(b['stem']):
                    got = self_mem.get(stm)
                    if got is not None and random.random() < a.self_p:
                        cond[bi, 6:] = torch.from_numpy(
                            got.astype(np.float32) / 255.0
                        ).permute(2, 0, 1).to(dev, wdt)
            with torch.no_grad():
                lat = (vae.encode(px).latent_dist.sample()
                       * vae.config.scaling_factor).float()
                ie = img_enc(b['ref_clip'].to(dev, wdt)).image_embeds
                drop = torch.rand(ie.shape[0], 1, device=dev) < 0.10
                ie = ie * (~drop).to(wdt)
                emb = emb_step[b['step'].to(dev)]
                pdrop = torch.rand(px.shape[0], device=dev) < 0.10
                emb = torch.where(pdrop[:, None, None], emb_null[None],
                                  emb).to(wdt)
            noise = torch.randn_like(lat)
            t = torch.randint(0, T, (lat.shape[0],), device=dev).long()
            noisy = sched.add_noise(lat, noise, t)
            # explicit autocast: the ControlNet keeps fp32 master weights
            with torch.autocast('cuda', wdt):
                down, mid = cn(noisy.to(wdt), t, encoder_hidden_states=emb,
                               controlnet_cond=cond, return_dict=False)
                pred = unet(noisy.to(wdt), t, encoder_hidden_states=emb,
                            added_cond_kwargs={'image_embeds':
                                               [ie.unsqueeze(1)]},
                            down_block_additional_residuals=list(down),
                            mid_block_additional_residual=mid).sample

            # object-weighted loss: the bench is cheap, and inventing
            # structure where there is none is expensive
            m_dil = F.max_pool2d(mask, 9, 1, 4)
            m_lat = (F.interpolate(m_dil, size=lat.shape[-2:],
                                   mode='bilinear') > 0.2).float()
            err = F.mse_loss(pred.float(), noise.float(), reduction='none')
            obj = (err * m_lat).sum() / (m_lat.sum() * err.shape[1] + 1e-6)
            bg_r = 1 - m_lat
            bg = (err * bg_r).sum() / (bg_r.sum() * err.shape[1] + 1e-6)
            bg_err = (err.mean(1, keepdim=True) * bg_r).flatten(1)
            k = max(1, int(bg_err.shape[1] * 0.10))
            loss = obj + a.bg_weight * bg \
                + 0.5 * torch.topk(bg_err, k, dim=1).values.mean()

            acc.backward(loss)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            v = loss.item()
            ema = v if ema is None else 0.98 * ema + 0.02 * v
            if step % 50 == 0:
                print(f'step {step:6d}  loss {v:.4f}  ema {ema:.4f}',
                      flush=True)
            if (a.rollout_every and step >= a.rollout_start
                    and step % a.rollout_every == 0):
                try:
                    rollout()
                    print(f'[rollout] step {step}: {len(self_mem)} frames of '
                          'self-generated memory cached', flush=True)
                except Exception as e:
                    print('[rollout] skipped:', e, flush=True)
            if step % a.val_every == 0 or step == a.steps:
                acc.unwrap_model(cn).save_pretrained(
                    os.path.join(a.out, 'controlnet'))
                print(f'[ckpt] step {step} -> {a.out}/controlnet', flush=True)
                try:
                    sample(step)
                except Exception as e:
                    print('[val] skipped:', e, flush=True)
            if a.clip_every and (step % a.clip_every == 0 or step == a.steps):
                try:
                    clip(step)
                except Exception as e:
                    print('[clip] skipped:', e, flush=True)
    print('done')


if __name__ == '__main__':
    main()
