#!/usr/bin/env python3
"""Neural render service: takes the conditioning the manual page produced from
its OWN blockout and returns a photographic frame.

The point of the split is that no CAD ever reaches this process. It receives
depth (and optionally a mask and an id map) rendered from the agent's boxes,
plus a caption and a reference photo, and returns an image. Whatever detail
appears in the output was added by the model.

  python serve_render.py --port 8900 [--fast] [--res 512]
Tunnel from the laptop:
  ssh -N -L 8900:watgpuNNN:8900 m2bulesh@watgpu.cs.uwaterloo.ca
"""
from __future__ import annotations

import argparse
import base64
import io
import math
import os
import threading
import time

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool
from PIL import Image
from pydantic import BaseModel

NEG = ("cluttered background, text, watermark, oversharpened, cartoon, "
       "lowres, duplicated parts")


class RenderReq(BaseModel):
    depth: str                      # data URL, the blockout's depth
    albedo: str | None = None       # data URL, flat per-part colour
    mask: str | None = None         # data URL, object vs bench
    ref: str | None = None          # data URL, the user's reference photo
    prompt: str = ""
    steps: int = 0                  # 0 = server default
    strength: float = 0.0           # >0 turns on img2img over `init`
    init: str | None = None         # data URL, e.g. the schematic itself
    cn_scale: float = 1.15
    seed: int = 0


def decode(data_url: str | None, size: int | None = None) -> Image.Image | None:
    if not data_url:
        return None
    b = base64.b64decode(data_url.split(',', 1)[-1])
    im = Image.open(io.BytesIO(b)).convert('RGB')
    return im.resize((size, size), Image.BILINEAR) if size else im


def naturalize(depth: Image.Image, mask: Image.Image | None) -> Image.Image:
    """Normalise the client's depth so the subject uses the full range, then
    fill anything empty with a plausible room gradient. The pretrained depth
    ControlNet saw full-frame MiDaS-style depth, never an object on black,
    and a client's near/far choice must not change what the model sees."""
    a = np.asarray(depth.convert('L')).astype(np.float32)
    valid = a > 3
    if mask is not None:
        m = np.asarray(mask.convert('L')).astype(np.float32) > 40
        if m.sum() > 50:
            valid = valid | m
    if valid.sum() > 50:
        lo, hi = np.percentile(a[valid], 1), np.percentile(a[valid], 99)
        if hi - lo > 1e-3:
            a = np.clip((a - lo) / (hi - lo), 0, 1) * 215 + 40
    grad = np.linspace(35, 110, a.shape[0], dtype=np.float32)[:, None]
    bg = np.repeat(grad, a.shape[1], axis=1)
    out = np.where(valid, a, bg)
    return Image.fromarray(out.astype(np.uint8)).convert('RGB')


def to_cond(depth: Image.Image, alb: Image.Image | None,
            mem: Image.Image | None, res: int, ch: int) -> torch.Tensor:
    """The conditioning a trained ControlNet expects, as a tensor because the
    pipeline's image preprocessing only understands three channels:

        3 channels   depth
        6 channels   depth, memory
        9 channels   depth, albedo (flat per-part colour), memory
    """
    def chan(im):
        if im is None:
            return torch.zeros(3, res, res)
        if im.size != (res, res):
            im = im.resize((res, res), Image.BILINEAR)
        return torch.from_numpy(
            np.asarray(im.convert('RGB')).astype(np.float32) / 255.0
        ).permute(2, 0, 1)
    parts = [chan(depth)] + ([chan(alb)] if ch >= 9 else []) + [chan(mem)]
    return torch.cat(parts, 0)[None].to('cuda', torch.bfloat16)


def build_app(a) -> FastAPI:
    from diffusers import (ControlNetModel, LCMScheduler,
                           StableDiffusionControlNetImg2ImgPipeline,
                           StableDiffusionControlNetPipeline,
                           UniPCMultistepScheduler)
    print('[render] loading ...', flush=True)
    cn = ControlNetModel.from_pretrained(a.memory_cn or a.cn,
                                         torch_dtype=torch.bfloat16)
    cond_ch = int(getattr(cn.config, 'conditioning_channels', 3) or 3)
    memory_mode = cond_ch >= 6
    if memory_mode:
        # a trained ControlNet saw the page's raw depth and learned this
        # object; the naturalising and the img2img chaining were both crutches
        # for a pretrained one, and both would now feed it the wrong thing
        print(f'[render] trained ControlNet: {cond_ch} conditioning channels '
              f'({"depth, albedo, memory" if cond_ch >= 9 else "depth, memory"}'
              '), chaining off, depth passed through raw', flush=True)
        a.chain = 0.0
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        a.sd, controlnet=cn, torch_dtype=torch.bfloat16, safety_checker=None)
    pipe.load_ip_adapter('h94/IP-Adapter', subfolder='models',
                         weight_name='ip-adapter_sd15.bin')
    pipe.set_ip_adapter_scale(a.ip_scale)
    # LoRAs need the peft backend; without it, fall back to plain sampling
    # rather than dying, since the service is still useful at full steps
    try:
        import peft                                    # noqa: F401
        have_peft = True
    except ImportError:
        have_peft = False
        if a.lora or a.fast:
            print('[render] peft not installed: ignoring --fast/--lora. '
                  'Install with:  python -m pip install peft', flush=True)
    use_lora = a.lora and have_peft
    use_fast = a.fast and have_peft
    if use_lora:
        pipe.load_lora_weights(a.lora, adapter_name='item')
        pipe.set_adapters(['item'], [a.lora_scale])
        print(f'[render] item LoRA {a.lora} @ {a.lora_scale}')
    if use_fast:
        pipe.load_lora_weights('latent-consistency/lcm-lora-sdv1-5',
                               adapter_name='lcm')
        names = (['item', 'lcm'] if use_lora else ['lcm'])
        weights = ([a.lora_scale, 1.0] if use_lora else [1.0])
        pipe.set_adapters(names, weights)
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        print('[render] LCM few-step mode')
    else:
        pipe.scheduler = UniPCMultistepScheduler.from_config(
            pipe.scheduler.config)
    if use_lora or use_fast:
        pipe.fuse_lora()
        pipe.unload_lora_weights()
    a.fast = use_fast
    if a.tiny_vae:
        try:
            from diffusers import AutoencoderTiny
            pipe.vae = AutoencoderTiny.from_pretrained(
                'madebyollin/taesd', torch_dtype=torch.bfloat16)
            print('[render] tiny VAE decoder')
        except Exception as e:
            print('[render] tiny VAE unavailable, keeping the full one:', e)
    if not torch.cuda.is_available():
        raise SystemExit('[render] no CUDA device visible. Are you on a GPU '
                         'node with an allocation? Check nvidia-smi.')
    dev = torch.cuda.get_device_name(0)
    pipe.to('cuda')
    pipe.set_progress_bar_config(disable=True)
    if a.compile:
        pipe.unet = torch.compile(pipe.unet, mode='reduce-overhead')
        pipe.controlnet = torch.compile(pipe.controlnet, mode='reduce-overhead')
        print('[render] torch.compile on, first frames will be slow', flush=True)
    # pass the VAE explicitly: from_pipe drops components whose class differs
    # from what it expects, which is what happens with the tiny decoder
    pipe_i2i = StableDiffusionControlNetImg2ImgPipeline.from_pipe(
        pipe, vae=pipe.vae)
    # from_pipe shares the scheduler object, and schedulers carry per-run state
    # (timesteps, step index), so give the second pipeline its own
    pipe_i2i.scheduler = type(pipe.scheduler).from_config(pipe.scheduler.config)
    # one inference at a time: FastAPI runs sync endpoints in a threadpool, and
    # two concurrent runs corrupt the scheduler's step counter
    gpu_lock = threading.Lock()
    # a fresh scheduler is built per frame (LCM keeps a step index across a
    # run), and PNDM's leftover key makes it warn every single time
    sched_cfg = {k: v for k, v in dict(pipe.scheduler.config).items()
                 if k != 'skip_prk_steps'}
    print(f'[render] ready on {dev} | res {a.res} | '
          f'{"4-step LCM" if a.fast else "18-step"} | '
          f'{"memory" if memory_mode else "chain " + str(a.chain)} | '
          f'torch {torch.__version__}', flush=True)

    app = FastAPI()
    app.add_middleware(CORSMiddleware, allow_origins=['*'],
                       allow_methods=['*'], allow_headers=['*'])
    state = {'last': None}

    # once an IP-Adapter is loaded the UNet always expects an image embed, so
    # a missing reference must mute the adapter rather than omit the argument
    blank_ref = Image.new('RGB', (224, 224), (127, 127, 127))

    # The prompt and the reference photo are the same on every frame, yet both
    # were being re-encoded each time, the reference through a CLIP image
    # tower that is larger than the UNet itself. Encode once, reuse.
    emb_cache = {}

    def cached_prompt(prompt, do_cfg):
        key = (prompt, do_cfg)
        if key not in emb_cache:
            emb_cache.clear()
            pe, ne = pipe.encode_prompt(
                prompt, 'cuda', 1, do_cfg,
                negative_prompt=(NEG if do_cfg else None))
            emb_cache[key] = (pe, ne)
        return emb_cache[key]

    ip_cache = {'key': None, 'embeds': None}

    def cached_ip(ref_im, do_cfg):
        key = (id(ref_im), do_cfg)
        if ip_cache['key'] != key:
            ip_cache['embeds'] = pipe.prepare_ip_adapter_image_embeds(
                [ref_im], None, torch.device('cuda'), 1, do_cfg)
            ip_cache['key'] = key
        return ip_cache['embeds']

    def generate(depth_im, prompt, cn_scale, seed, steps_req,
                 strength=0.0, init=None, mask=None, alb=None):
      with gpu_lock:
          # a fresh scheduler per call: LCM keeps a step index across the run
          # and a cancelled request can leave it half-set
          pipe.scheduler = type(pipe.scheduler).from_config(sched_cfg)
          pipe_i2i.scheduler = type(pipe.scheduler).from_config(sched_cfg)
          depth = depth_im if memory_mode else naturalize(depth_im, mask)

          # ---- chaining: grow this frame out of the previous one so the look
          # persists instead of being re-rolled every frame. Reset when the
          # view jumps or the step changes, and purge periodically, because
          # chained frames inherit each other's defects.
          chain = strength if strength > 0 else a.chain
          cur_d = np.asarray(depth.convert('L'), dtype=np.float32)
          prev_d = state.get('last_depth')
          moved = 1e9 if (prev_d is None or prev_d.shape != cur_d.shape) \
              else float(np.abs(cur_d - prev_d).mean())
          state['last_depth'] = cur_d
          if init is None and chain > 0 and state.get('last') is not None:
              fresh = (moved > a.chain_reset
                       or state.get('chained', 0) >= a.chain_purge
                       or prompt != state.get('last_prompt'))
              if not fresh:
                  init = state['last']
                  strength = chain
                  state['chained'] = state.get('chained', 0) + 1
              else:
                  state['chained'] = 0
          step_changed = prompt != state.get('last_prompt')
          state['last_prompt'] = prompt
          ref = state.get('ref')
          pipe.set_ip_adapter_scale(a.ip_scale if ref is not None else 0.0)
          pipe_i2i.set_ip_adapter_scale(a.ip_scale if ref is not None else 0.0)
          steps = steps_req or a.steps or (4 if a.fast else 18)
          # 1.0 disables the unconditional pass, halving the work per frame.
          # A ControlNet trained on this one object already pins the output,
          # so guidance buys much less than it does with a pretrained one.
          cfg = 1.0 if a.fast else a.cfg
          g = torch.Generator('cuda').manual_seed(seed)
          do_cfg = cfg > 1.0
          kw = dict(generator=g, guidance_scale=cfg,
                    controlnet_conditioning_scale=cn_scale)
          try:
              pe, ne = cached_prompt(prompt, do_cfg)
              kw['prompt_embeds'] = pe
              if do_cfg:
                  kw['negative_prompt_embeds'] = ne
              kw['ip_adapter_image_embeds'] = cached_ip(
                  ref if ref is not None else blank_ref, do_cfg)
              prompt = None
          except Exception as e:      # older diffusers: fall back to re-encoding
              print('[render] embed cache unavailable:', e, flush=True)
              kw['ip_adapter_image'] = ref if ref is not None else blank_ref
              if not a.fast:
                  kw['negative_prompt'] = NEG
          if memory_mode:
              # the memory picture, not a starting latent: the frame is still
              # denoised from noise, so nothing accumulates and nothing has to
              # be purged. Prefer what the page remembers for THIS viewpoint;
              # fall back to the frame just drawn. Forget on a step change,
              # since the object itself changed.
              # the memory is NOT reprojected into the current view, so it is
              # only usable while the view is close to the one that produced
              # it. The page's own memory is filed by viewpoint and is safe;
              # the server's last frame is only safe if barely anything moved,
              # otherwise the model paints the old silhouette into the new
              # frame and the object smears
              if step_changed:
                  mem = None
              elif init is not None:
                  mem = init
              elif moved <= a.mem_reset:
                  mem = state.get('mem')
              else:
                  mem = None
              state['path'] = ('memory' if init is not None else
                               'memory-last' if mem is not None else 'cold')
              if mem is not None and a.mem_scale < 1.0:
                  mem = Image.blend(Image.new('RGB', mem.size, (0, 0, 0)),
                                    mem.convert('RGB'), a.mem_scale)
              img = pipe(prompt,
                         image=to_cond(depth, alb, mem, a.res, cond_ch),
                         num_inference_steps=steps, **kw).images[0]
              state['mem'] = img
              state['last'] = img
              return img
          state['path'] = 'chain' if (strength > 0 and init is not None) \
              else 'fresh'
          if strength > 0 and init is not None:
              # img2img runs only strength x steps of the schedule, so ask for
              # more to keep the actual work per frame the same
              eff = max(2, int(math.ceil(steps / max(strength, 0.05))))
              img = pipe_i2i(prompt, image=init, control_image=depth,
                             strength=strength, num_inference_steps=eff,
                             **kw).images[0]
          else:
              img = pipe(prompt, image=depth, num_inference_steps=steps,
                         **kw).images[0]
          state['last'] = img
          if mask is not None and init is not None:
              al = (np.asarray(mask.convert('L')).astype(np.float32)/255.)[..., None]
              al = np.clip(al*1.15, 0, 1)
              comp = np.asarray(img).astype(np.float32)*al + \
                  np.asarray(init).astype(np.float32)*(1-al)
              img = Image.fromarray(comp.astype(np.uint8))
          return img

    if a.compile or a.warmup:
        t0 = time.time()
        blank = Image.new('RGB', (a.res, a.res), (128, 128, 128))
        for _ in range(3 if a.compile else 1):
            generate(blank, '', 1.15, 0, 0)
        print(f'[render] warmup {time.time()-t0:.1f}s', flush=True)

    def to_jpeg(img, q=88):
        buf = io.BytesIO()
        img.convert('RGB').save(buf, 'JPEG', quality=q)
        return buf.getvalue()

    # ---- streaming socket: binary in, binary out, no base64, no handshake
    # per frame. The client keeps exactly one frame in flight, which is the
    # simplest correct backpressure.
    import json as _json

    @app.websocket('/ws')
    async def ws_render(sock: WebSocket):
        await sock.accept()
        cfg = {'prompt': '', 'cn_scale': 1.15, 'seed': 0, 'steps': 0}
        n = 0
        print('[render] websocket open', flush=True)
        try:
            while True:
                msg = await sock.receive()
                if msg.get('text'):
                    d = _json.loads(msg['text'])
                    if d.get('ref'):
                        state['ref'] = decode(d['ref'], 224)
                    for k in ('prompt', 'cn_scale', 'seed', 'steps',
                              'strength'):
                        if k in d:
                            cfg[k] = d[k]
                    continue
                data = msg.get('bytes')
                if not data:
                    if msg.get('type') == 'websocket.disconnect':
                        break
                    continue
                t0 = time.time()
                init_im = alb_im = None

                def _jpg(b):
                    return Image.open(io.BytesIO(b)).convert('RGB').resize(
                        (a.res, a.res), Image.BILINEAR) if b else None

                if len(data) > 7 and data[:1] == b'\x01':
                    # [0x01][depth len][albedo len][depth][albedo][memory]
                    n_d = int.from_bytes(data[1:4], 'big')
                    n_a = int.from_bytes(data[4:7], 'big')
                    depth_b = data[7:7 + n_d]
                    alb_im = _jpg(data[7 + n_d:7 + n_d + n_a])
                    init_im = _jpg(data[7 + n_d + n_a:])
                elif len(data) > 4 and data[:1] == b'\x00':
                    n_depth = int.from_bytes(data[1:4], 'big')
                    depth_b = data[4:4 + n_depth]
                    init_im = _jpg(data[4 + n_depth:])
                else:
                    depth_b = data
                depth_im = Image.open(io.BytesIO(depth_b)).convert('RGB') \
                                .resize((a.res, a.res), Image.BILINEAR)
                img = await run_in_threadpool(
                    generate, depth_im, cfg['prompt'], cfg['cn_scale'],
                    cfg['seed'], cfg['steps'],
                    cfg.get('strength', 0.0), init_im, None, alb_im)
                await sock.send_bytes(to_jpeg(img))
                n += 1
                # say out loud whether the albedo arrived and carries
                # colour: a stale page sends the old payload, and zeroed
                # colour channels are exactly what makes colour drift again
                if alb_im is None:
                    alb_note = ' NO-ALBEDO'
                else:
                    ex = np.asarray(alb_im).reshape(-1, 3)
                    alb_note = (' +alb' if ex.max() > 12 else ' ALBEDO-BLACK')
                print(f'[render] ws {int((time.time()-t0)*1000)} ms  '
                      f'{state.get("path")}'
                      f'{" +mem" if init_im is not None else ""}'
                      f'{alb_note if cond_ch >= 9 else ""}', flush=True)
        except WebSocketDisconnect:
            print('[render] websocket closed', flush=True)
        except Exception as e:
            print('[render] websocket error:', e, flush=True)

    @app.get('/health')
    def health():
        return {'ok': True, 'device': dev, 'res': a.res, 'fast': a.fast,
                'steps': 4 if a.fast else 18, 'lora': bool(a.lora),
                'memory': memory_mode, 'cond_channels': cond_ch}

    @app.post('/render')
    def render(r: RenderReq):
        t0 = time.time()
        m = decode(r.mask, a.res)
        ref = decode(r.ref, 224)
        if ref is not None:
            state['ref'] = ref
        img = generate(decode(r.depth, a.res), r.prompt, r.cn_scale, r.seed,
                       r.steps, r.strength, decode(r.init, a.res), m,
                       decode(r.albedo, a.res))
        ms = int((time.time()-t0)*1000)
        print(f'[render] http {ms} ms  {state.get("path")}  res={a.res}',
              flush=True)
        return {'img': 'data:image/jpeg;base64,'
                       + base64.b64encode(to_jpeg(img, 90)).decode(),
                'ms': ms}

    return app


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--sd', default='runwayml/stable-diffusion-v1-5')
    ap.add_argument('--cn', default='lllyasviel/control_v11f1p_sd15_depth')
    ap.add_argument('--memory-cn', default=None,
                    help='trained 6-channel ControlNet (train_memory_cn.py '
                         'output, e.g. memory_cn/controlnet): the previous '
                         'frame becomes conditioning instead of a start latent')
    ap.add_argument('--lora', default=None,
                    help='item LoRA dir (train_item_lora.py output)')
    ap.add_argument('--lora-scale', type=float, default=0.8)
    ap.add_argument('--ip-scale', type=float, default=0.7)
    ap.add_argument('--res', type=int, default=512)
    ap.add_argument('--fast', action='store_true', help='LCM few-step')
    ap.add_argument('--steps', type=int, default=0,
                    help='override the step count (2 is usable with --fast)')
    ap.add_argument('--mem-reset', type=float, default=4.0,
                    help='mean depth change (0-255) above which the previous '
                         'frame is too stale to use as memory; lower this if '
                         'the object smears while you turn')
    ap.add_argument('--mem-scale', type=float, default=1.0,
                    help='attenuate the memory picture (0 disables it)')
    ap.add_argument('--cfg', type=float, default=5.0,
                    help='guidance scale; 1.0 skips the unconditional pass '
                         'and halves the cost per frame')
    ap.add_argument('--tiny-vae', action='store_true',
                    help='TAESD decoder: much cheaper than the full VAE')
    ap.add_argument('--compile', action='store_true',
                    help='torch.compile the UNet and ControlNet (~1.4x)')
    ap.add_argument('--warmup', action='store_true',
                    help='run a throwaway frame at startup')
    ap.add_argument('--chain', type=float, default=0.0,
                    help='0.4-0.6 grows each frame from the previous one, '
                         'which keeps the look steady between frames')
    ap.add_argument('--chain-reset', type=float, default=10.0,
                    help='mean depth change (0-255) that counts as a jump '
                         'and starts a fresh frame')
    ap.add_argument('--chain-purge', type=int, default=8,
                    help='force a fresh frame every N chained frames')
    ap.add_argument('--port', type=int, default=8900)
    ap.add_argument('--ws', default='auto',
                    help="uvicorn websocket implementation: auto, "
                         "websockets, websockets-sansio, wsproto, none")
    a = ap.parse_args()
    import uvicorn
    app = build_app(a)
    # uvicorn falls back to HTTP-only, and refuses upgrades with 403, when it
    # cannot load a websocket implementation. Say so out loud rather than
    # leaving it to be discovered from the client side.
    cfg = uvicorn.Config(app, host='0.0.0.0', port=a.port, ws=a.ws)
    try:
        cfg.load()          # the protocol classes only exist after loading
    except Exception as e:
        print('[render] uvicorn config load failed:', e, flush=True)
    impl = getattr(cfg, 'ws_protocol_class', 'unknown')
    print(f'[render] websocket implementation: {impl}', flush=True)
    if impl in (None, 'unknown'):
        print('[render] NO websocket support: upgrades will be refused with '
              '403. Try --ws websockets-sansio, or --ws wsproto after '
              'installing wsproto.', flush=True)
    uvicorn.Server(cfg).run()
