#!/usr/bin/env python3
"""Render realistic frames from geometry maps with the fine-tuned ControlNet,
grounded on a reference photo via pretrained IP-Adapter.

This is the style-generalization test the whole experiment exists for:
  --depth   any depth map (held-out IKEA pose, OR exported from the agent's
            custom 3D manual — the contract is engine-agnostic)
  --ref     any reference photo (the user's own object)
  --real    optional ground-truth frame -> adds it to the triptych

Examples:
  # held-out IKEA pose, its own video as reference
  python infer_renderer.py --cn renderer_ckpt/controlnet \
      --depth data/maps/stig_YqWk45Sawt4/frame_084450_depth.png \
      --ref   data/pairs/stig_YqWk45Sawt4/frame_016113.jpg \
      --real  data/pairs/stig_YqWk45Sawt4/frame_084450.jpg

  # the agent's custom manual + the user's photo (the pitch shot)
  python infer_renderer.py --cn renderer_ckpt/controlnet \
      --depth agent_scene_depth.png --ref my_photo.jpg
"""
from __future__ import annotations

import argparse
import os

import torch
from PIL import Image

PROMPT = "a photo of furniture being assembled indoors, realistic"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cn", required=True, help="trained controlnet dir")
    ap.add_argument("--depth", required=True)
    ap.add_argument("--ref", required=True, help="reference photo (appearance)")
    ap.add_argument("--real", default=None, help="optional GT for the triptych")
    ap.add_argument("--sd", default="runwayml/stable-diffusion-v1-5")
    ap.add_argument("--ip-scale", type=float, default=0.7)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="out.jpg")
    ap.add_argument("--mask", default=None,
                    help="object silhouette (e.g. the _pid.png map): model "
                         "generates INSIDE it; outside, latents are pinned to "
                         "--bg every denoising step (blended latent diffusion)")
    ap.add_argument("--bg", default=None,
                    help="background image for the pinned region "
                         "(default: the --ref photo)")
    ap.add_argument("--edge", type=int, default=8,
                    help="soft blend band at the mask boundary, px at 512")
    ap.add_argument("--history", default=None,
                    help="previous generated/real frame (sequence rollout). "
                         "Omit for the first frame of a session.")
    ap.add_argument("--normal", default=None,
                    help="normal map (default: derived from --depth path)")
    ap.add_argument("--pid", default=None,
                    help="part-ID map (default: derived from --depth path)")
    ap.add_argument("--prompt", default=None,
                    help="text prompt (e.g. the per-item training prompt)")
    a = ap.parse_args()

    from diffusers import (ControlNetModel, StableDiffusionControlNetPipeline,
                           UniPCMultistepScheduler)

    cn = ControlNetModel.from_pretrained(a.cn, torch_dtype=torch.bfloat16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        a.sd, controlnet=cn, torch_dtype=torch.bfloat16, safety_checker=None)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models",
                         weight_name="ip-adapter_sd15.bin")
    pipe.set_ip_adapter_scale(a.ip_scale)
    pipe.to("cuda")

    depth = Image.open(a.depth).convert("RGB")
    ref = Image.open(a.ref).convert("RGB")
    w = 512 * depth.width // depth.height // 8 * 8
    depth_in = depth.resize((w, 512), Image.BILINEAR)

    # multi-channel condition if the checkpoint expects it; built manually as
    # a tensor to bypass 3ch image preprocessing
    import numpy as np
    cond_ch = cn.config.conditioning_channels

    def load_t(path):
        im = Image.open(path).convert("RGB").resize((w, 512), Image.BILINEAR)
        return torch.from_numpy(np.asarray(im).astype("float32") / 255.0
                                ).permute(2, 0, 1)

    if cond_ch > 3:
        d_t = torch.from_numpy(np.asarray(depth_in).astype("float32") / 255.0
                               ).permute(2, 0, 1)
        h_t = load_t(a.history) if a.history else torch.zeros_like(d_t)
        chans = [d_t]
        if cond_ch == 12:               # depth + normal + pid + history
            normal_p = a.normal or a.depth.replace("_depth.png", "_normal.png")
            pid_p = a.pid or a.depth.replace("_depth.png", "_pid.png")
            chans += [load_t(normal_p), load_t(pid_p)]
        chans.append(h_t)
        depth_in = torch.cat(chans, 0)[None].to("cuda", torch.bfloat16)

    g = torch.Generator("cuda").manual_seed(a.seed)

    # --- masked latent blending: generate object, pin background to a photo
    callback = None
    if a.mask:
        import numpy as np
        from PIL import ImageFilter
        h_lat, w_lat = 512 // 8, w // 8
        m = Image.open(a.mask).convert("L").resize((w, 512), Image.NEAREST)
        m = m.point(lambda p: 255 if p > 8 else 0)
        m = m.filter(ImageFilter.GaussianBlur(a.edge))           # soft edge
        m_lat = torch.from_numpy(
            np.asarray(m.resize((w_lat, h_lat), Image.BILINEAR))
            .astype("float32") / 255.0)[None, None].to("cuda", torch.bfloat16)

        bg = Image.open(a.bg or a.ref).convert("RGB").resize((w, 512), Image.BILINEAR)
        bg_t = torch.from_numpy(np.asarray(bg).astype("float32") / 127.5 - 1.0
                                ).permute(2, 0, 1)[None].to("cuda", torch.bfloat16)
        with torch.no_grad():
            bg_lat = (pipe.vae.encode(bg_t).latent_dist.sample()
                      * pipe.vae.config.scaling_factor)
        bg_noise = torch.randn(bg_lat.shape, generator=g, device="cuda",
                               dtype=torch.bfloat16)

        def callback(p, i, t, kw):
            lat = kw["latents"]
            last = i >= a.steps - 1
            bg_now = bg_lat if last else p.scheduler.add_noise(
                bg_lat, bg_noise, t.reshape(1))
            kw["latents"] = m_lat * lat + (1 - m_lat) * bg_now.to(lat.dtype)
            return kw

    img = pipe(a.prompt or PROMPT, image=depth_in, ip_adapter_image=ref,
               num_inference_steps=a.steps, generator=g,
               guidance_scale=5.0,
               callback_on_step_end=callback,
               callback_on_step_end_tensor_inputs=["latents"]).images[0]

    depth_disp = depth.resize((w, 512), Image.BILINEAR)
    panels = [(depth_disp, "depth (condition)"),
              (ref.resize((w, 512)), "reference (appearance)"),
              (img, "GENERATED")]
    if a.real:
        panels.append((Image.open(a.real).convert("RGB").resize((w, 512)),
                       "real ground truth"))
    from PIL import ImageDraw
    sheet = Image.new("RGB", (w * len(panels), 512 + 30), (12, 12, 15))
    d = ImageDraw.Draw(sheet)
    for i, (p, lab) in enumerate(panels):
        sheet.paste(p, (i * w, 30))
        d.text((i * w + 8, 8), lab, fill=(225, 225, 235))
    sheet.save(a.out, quality=92)
    print("wrote", os.path.abspath(a.out))


if __name__ == "__main__":
    main()
