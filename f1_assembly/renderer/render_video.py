#!/usr/bin/env python3
"""Play a captured interaction back through the trained renderer and write a
side-by-side video: the blockout on the left, what the model makes of it in
the middle, the real render on the right.

This is the model running exactly as it does when served, frame after frame,
with each frame's memory being the model's OWN previous output rather than the
ground truth. So the video shows the drift and the flicker honestly, not a
best case assembled from independent stills.

  python render_video.py --cn memory_cn384/controlnet \
      --data ../../f1_manual/traindata --seq papaya_v0 --out clip_papaya

Add --no-memory for the same clip with the memory channels blanked, which is
the A/B for what the memory is buying.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess

import numpy as np
import torch
from PIL import Image, ImageDraw

from train_memory_cn import step_prompts

NEG = ('cluttered background, text, watermark, oversharpened, cartoon, '
       'lowres, duplicated parts')


def label(im, text):
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, im.width, 22], fill=(0, 0, 0))
    d.text((8, 6), text, fill=(255, 255, 255))
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cn', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', default='clip')
    ap.add_argument('--seq', default=None, help='sequence id, default first')
    ap.add_argument('--manifest', default=None)
    ap.add_argument('--sd', default='runwayml/stable-diffusion-v1-5')
    ap.add_argument('--res', type=int, default=384)
    ap.add_argument('--steps', type=int, default=8)
    ap.add_argument('--cfg', type=float, default=1.0)
    ap.add_argument('--ip-scale', type=float, default=0.7)
    ap.add_argument('--cn-scale', type=float, default=1.15)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--fast', action='store_true', help='LCM few-step')
    ap.add_argument('--no-memory', action='store_true')
    ap.add_argument('--frames', type=int, default=0, help='0 = whole clip')
    ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--fps', type=int, default=12)
    ap.add_argument('--left', default='depth', choices=['depth', 'sch'],
                    help="left panel: the raw depth, or the line-art "
                         "schematic captured with ?sch=1")
    ap.add_argument('--panels', type=int, default=3, choices=[2, 3],
                    help='2 = schematic and render only, for showing; '
                         '3 = adds the target render, for judging')
    ap.add_argument('--upscale', type=int, default=0,
                    help='panel size in the video, e.g. 768')
    ap.add_argument('--no-labels', action='store_true')
    a = ap.parse_args()

    from diffusers import (ControlNetModel, LCMScheduler,
                           StableDiffusionControlNetPipeline,
                           UniPCMultistepScheduler)
    dev, wdt = 'cuda', torch.bfloat16
    cn = ControlNetModel.from_pretrained(a.cn, torch_dtype=wdt)
    ch = int(getattr(cn.config, 'conditioning_channels', 3) or 3)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        a.sd, controlnet=cn, torch_dtype=wdt, safety_checker=None)
    pipe.load_ip_adapter('h94/IP-Adapter', subfolder='models',
                         weight_name='ip-adapter_sd15.bin')
    pipe.set_ip_adapter_scale(a.ip_scale)
    if a.fast:
        pipe.load_lora_weights('latent-consistency/lcm-lora-sdv1-5')
        pipe.fuse_lora()
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    else:
        pipe.scheduler = UniPCMultistepScheduler.from_config(
            pipe.scheduler.config)
    pipe.to(dev)
    pipe.set_progress_bar_config(disable=True)
    print(f'[video] {a.cn}: {ch} conditioning channels, '
          f'memory {"OFF" if a.no_memory else "ON"}', flush=True)

    meta = json.load(open(os.path.join(a.data, 'trajectories.json')))
    seq = next((s for s in meta['seqs'] if s['id'] == a.seq), None) \
        if a.seq else meta['seqs'][0]
    if seq is None:
        raise SystemExit(f'no sequence {a.seq}; have '
                         f'{[s["id"] for s in meta["seqs"]][:8]} ...')
    frames = seq['frames'][::a.stride]
    if a.frames:
        frames = frames[:a.frames]
    man = a.manifest or os.path.join(a.data, '..', 'viewer_manifest.json')
    prompts = step_prompts(man)
    ref = Image.open(os.path.join(a.data, seq['ref'] + '.png')).convert('RGB')
    print(f'[video] {seq["id"]}: {len(frames)} frames', flush=True)

    def load(name):
        im = Image.open(os.path.join(a.data, name + '.png')).convert('RGB')
        return im.resize((a.res, a.res), Image.BILINEAR)

    def to_cond(depth, alb, mem):
        def chan(im):
            if im is None:
                return torch.zeros(3, a.res, a.res)
            return torch.from_numpy(
                np.asarray(im.resize((a.res, a.res), Image.BILINEAR))
                .astype(np.float32) / 255.0).permute(2, 0, 1)
        parts = [chan(depth)]
        if ch >= 9:
            parts.append(chan(alb))
        if ch >= 6:
            parts.append(chan(mem))
        return torch.cat(parts, 0)[None].to(dev, wdt)

    outdir = a.out + '_frames'
    os.makedirs(outdir, exist_ok=True)
    mem = None
    for i, f in enumerate(frames):
        depth = load('cond_' + f['stem'])
        truth = load('tgt_' + f['stem'])
        ap = os.path.join(a.data, 'alb_' + f['stem'] + '.png')
        alb = load('alb_' + f['stem']) if os.path.exists(ap) else None
        if f.get('cut'):            # the object changed: forget, as when served
            mem = None
        g = torch.Generator(dev).manual_seed(a.seed)
        img = pipe(prompts[int(f['step'])],
                   image=to_cond(depth, alb,
                                 None if a.no_memory else mem),
                   ip_adapter_image=ref,
                   negative_prompt=(NEG if a.cfg > 1.0 else None),
                   num_inference_steps=a.steps, guidance_scale=a.cfg,
                   controlnet_conditioning_scale=a.cn_scale,
                   generator=g).images[0]
        mem = img                   # the model's own output, as when served
        sp = os.path.join(a.data, 'sch_' + f['stem'] + '.png')
        left = (load('sch_' + f['stem']) if a.left == 'sch'
                and os.path.exists(sp) else depth)
        tiles = [(left, 'schematic the agent built' if a.left == 'sch'
                  else 'blockout depth'),
                 (img.resize((a.res, a.res)), 'neural render')]
        if a.panels == 3:
            tiles.append((truth, 'target render'))
        w = a.upscale or a.res
        panel = Image.new('RGB', (w * len(tiles), w))
        for c, (im, txt) in enumerate(tiles):
            im = im.resize((w, w), Image.LANCZOS)
            panel.paste(im if a.no_labels else label(im, txt), (c * w, 0))
        panel.save(os.path.join(outdir, f'f{i:05d}.png'))
        if (i + 1) % 10 == 0:
            print(f'  {i+1}/{len(frames)}', flush=True)

    mp4 = a.out + '.mp4'
    cmd = ['ffmpeg', '-y', '-framerate', str(a.fps), '-i',
           os.path.join(outdir, 'f%05d.png'), '-c:v', 'libx264',
           '-pix_fmt', 'yuv420p', '-crf', '18', mp4]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        print(f'[video] wrote {mp4}')
    except Exception as e:
        print(f'[video] frames are in {outdir}/ but ffmpeg failed ({e}). '
              f'Encode with:\n  {" ".join(cmd)}')


if __name__ == '__main__':
    main()
