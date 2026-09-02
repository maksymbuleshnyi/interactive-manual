#!/usr/bin/env python3
"""Fine-tune a depth ControlNet to render realistic furniture from geometry
maps, GROUNDED on a reference photo (IP-Adapter active DURING training), with
MASKED loss (only furniture pixels are graded).

The grounding is the point: every training example is conditioned on a random
OTHER frame of the same video via the pretrained IP-Adapter (frozen), so the
ControlNet learns to produce the target WORKING WITH reference-conditioned
denoising — structure from the depth map, appearance from the reference.
Reference dropout (10%) keeps it robust to a missing/weak reference.

  - SD 1.5 + IP-Adapter attn processors: frozen
  - ControlNet (init: control_v11f1p_sd15_depth): trained
  - condition = depth map; reference = same-video random frame

Run (single H200):
  accelerate launch --mixed_precision bf16 train_renderer.py \
      --data /path/to/real_grounding --steps 8000 --bs 8
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

PROMPT = "a photo of furniture being assembled indoors, realistic"
# per-item identity hook for the 5-item overfit; unseen items fall back to
# the generic prompt (identity must then come from reference + maps)
PROMPTS = {
    "stig": "a black metal bar stool with backrest, photo, indoors",
    "ronninge": "a black wooden chair being assembled, photo, indoors",
    "teodores": "a red plastic chair being assembled, photo, indoors",
    "gladom": "a green round metal tray side table, photo, indoors",
    "laiva": "a black wooden bookshelf being assembled, photo, indoors",
}
RES = 512


MAX_HIST_GAP = 40      # video frames (~1.3s): larger gaps break a sequence


class Sequences(Dataset):
    """One example = one SEQUENCE:

        reference   1 grounding image, FIXED for the whole sequence
                    (a frame outside the window, same video)
        cond        K depth maps  (+ chained history channels)
        pixel       K corresponding real frames (targets)

    History chaining inside the window: frame k's condition carries frame
    k-1's real image (corrupted); k=0 gets zeros = the session-start mode.
    """

    def __init__(self, root: str, split: str, k: int = 4,
                 index: str | None = None):
        self.root = root
        self.k = k
        idx = json.load(open(index or os.path.join(root, "index.json")))
        if split not in idx and split == "val":
            split = "val_seen"          # scaling-tranche index layout
        self.items = idx[split]

        def fid(ex):  # ".../frame_012345_depth.png" -> 12345
            return int(ex["depth"].rsplit("frame_", 1)[1].split("_")[0])
        def vdir(ex):
            return ex["depth"].rsplit("/", 1)[0]

        by_vid: dict[str, list[int]] = {}
        for i, ex in enumerate(self.items):
            by_vid.setdefault(vdir(ex), []).append(i)
        # contiguous runs -> windows at several temporal STRIDES (1,2,4): the
        # same K frames spanning 1x/2x/4x the time, so motion-rich transitions
        # (a part flying in, a big camera move) are in-distribution, not only
        # near-static working moments
        self.windows: list[tuple[list[int], list[int]]] = []  # (window, vid pool)
        self.weights: list[float] = []     # motion score per window
        for vid, idxs in by_vid.items():
            idxs.sort(key=lambda i: fid(self.items[i]))
            run = [idxs[0]]
            runs = []
            for a_, b_ in zip(idxs, idxs[1:]):
                if fid(self.items[b_]) - fid(self.items[a_]) <= MAX_HIST_GAP:
                    run.append(b_)
                else:
                    runs.append(run)
                    run = [b_]
            runs.append(run)
            for r in runs:
                for stride in (1, 2, 4):
                    span = k * stride
                    if len(r) < span:
                        continue
                    for s in range(0, len(r) - span + 1, max(1, span // 2)):
                        win = r[s: s + span: stride]
                        if len(win) != k:
                            continue
                        # motion = everything that happened across the span
                        mot = sum(self.items[j].get("mot", 0.0)
                                  for j in r[s + 1: s + span])
                        # framing factor, per frame, window scored by its
                        # WORST frame: close-ups (cov~1) and structureless
                        # depth (edge~0) both null the conditioning value
                        def f_of(j):
                            ex_j = self.items[j]
                            cov = ex_j.get("cov", 0.3)
                            edge = ex_j.get("edge", 0.02)
                            f_cov = min(1.0, max(0.02, (0.92 - cov) / 0.25)) \
                                * min(1.0, cov / 0.05)
                            f_edge = min(1.0, edge / 0.015)
                            return f_cov * f_edge
                        frame_f = min(f_of(j) for j in win)
                        self.windows.append((win, idxs))
                        self.weights.append(0.002 + mot * frame_f)
        n_hi = sum(w > 0.05 for w in self.weights)
        print(f"[data:{split}] {len(self.items)} frames -> "
              f"{len(self.windows)} sequences of {k} (strides 1/2/4), "
              f"{n_hi} motion-rich; motion-weighted sampling ON")

    def __len__(self):
        return len(self.windows)

    def _load(self, rel: str, mode: str) -> Image.Image:
        return Image.open(os.path.join(self.root, rel)).convert(mode)

    @staticmethod
    def _corrupt(im: Image.Image) -> torch.Tensor:
        """History degradation: at rollout the model sees its OWN imperfect
        output here, so train on imperfect history (anti exposure-bias)."""
        if random.random() < 0.5:
            k = random.choice([2, 4])
            im = im.resize((RES // k, RES // k), Image.BILINEAR) \
                   .resize((RES, RES), Image.BILINEAR)
        ha = np.asarray(im).astype(np.float32) / 255.0
        ha += np.random.normal(0, random.uniform(0.0, 0.06), ha.shape)
        return torch.from_numpy(np.clip(ha, 0, 1).astype(np.float32)
                                ).permute(2, 0, 1)

    def __getitem__(self, i):
        win, pool = self.windows[i]
        exs = [self.items[j] for j in win]

        # ONE crop box for the WHOLE sequence: union mask BBOX, padded, square.
        # OBJECT-CENTRIC + ADAPTIVE: thin objects get magnified to fill the
        # frame (a 6px tube at native res is <1 latent pixel after the VAE's
        # 8x downsample -- the model cannot draw what it cannot see).
        first_real = self._load(exs[0]["real"], "RGB")
        w, h = first_real.size
        acc = np.zeros((h, w), bool)
        for ex in exs:
            acc |= np.asarray(self._load(ex["mask"], "L")) > 0
        ys, xs = np.nonzero(acc)
        if len(xs):
            bx0, bx1 = int(xs.min()), int(xs.max())
            by0, by1 = int(ys.min()), int(ys.max())
        else:
            bx0, by0, bx1, by1 = w // 4, h // 4, 3 * w // 4, 3 * h // 4
        side = max(bx1 - bx0, by1 - by0, 64)
        side = int(side * random.uniform(1.2, 1.6))     # pad around the object
        side = min(side, min(w, h))                     # cannot exceed frame
        cx = (bx0 + bx1) // 2 + random.randint(-side // 8, side // 8)
        cy = (by0 + by1) // 2 + random.randint(-side // 8, side // 8)
        # PART-LEVEL ZOOM mode (35%): for sprawling thin objects the union
        # bbox can exceed 512, which would zoom OUT and thin the tubes
        # further. Zooming into a random on-object region trains thick-tube
        # close-ups, so fine structure is actually representable post-VAE.
        if len(xs) and random.random() < 0.35:
            side = max(96, int(side * random.uniform(0.45, 0.7)))
            j = random.randrange(len(xs))
            cx, cy = int(xs[j]), int(ys[j])
        x0 = max(0, min(w - side, cx - side // 2))
        y0 = max(0, min(h - side, cy - side // 2))
        box = (x0, y0, x0 + side, y0 + side)

        def prep(im, norm):
            im = im.crop(box).resize((RES, RES), Image.BILINEAR)
            a = np.asarray(im).astype(np.float32) / 255.0
            if a.ndim == 2:
                a = a[..., None]
            if norm:
                a = a * 2.0 - 1.0
            return torch.from_numpy(a).permute(2, 0, 1)

        # ONE grounding image for the WHOLE sequence — chosen like a user
        # would: WELL-FRAMED (decent coverage + depth structure; no blurred
        # macro shots). Mostly far in time (forces "copy appearance, not
        # pose" — a near-identical reference leaks structure), sometimes
        # near (matches the demo, where the photo resembles session start).
        def good_ref(j):
            e = self.items[j]
            return (0.04 < e.get("cov", 0.3) < 0.75
                    and e.get("edge", 0.02) > 0.015)
        outside = [j for j in pool if j not in win]
        pool_good = [j for j in outside if good_ref(j)] or outside or list(win)
        if random.random() < 0.3:                      # near-in-time mode
            lo = pool.index(win[0])
            near = [j for j in pool_good
                    if abs(pool.index(j) - lo) <= 12]
            pool_good = near or pool_good
        ref = self._load(self.items[random.choice(pool_good)]["real"], "RGB") \
            .resize((224, 224), Image.BILINEAR)
        ref_a = np.asarray(ref).astype(np.float32) / 255.0
        clip_mean = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
        clip_std = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)
        ref_t = torch.from_numpy((ref_a - clip_mean) / clip_std).permute(2, 0, 1)

        # the sequence: K frames, history chained (k=0 -> zeros = session start)
        # condition = depth + normal + part-ID + history = 12 channels
        pixels, conds, masks = [], [], []
        for k, ex in enumerate(exs):
            real = self._load(ex["real"], "RGB")
            depth = self._load(ex["depth"], "RGB")
            normal = self._load(ex["normal"], "RGB")
            pid = self._load(ex["pid"], "RGB")
            mask = self._load(ex["mask"], "L")
            if k == 0:
                hist = torch.zeros(3, RES, RES)
            else:
                hist = self._corrupt(
                    self._load(exs[k - 1]["real"], "RGB").crop(box)
                    .resize((RES, RES), Image.BILINEAR))
            pixels.append(prep(real, True))
            conds.append(torch.cat([prep(depth, False), prep(normal, False),
                                    prep(pid, False), hist], 0))
            masks.append(prep(mask, False))

        return {"pixel": torch.stack(pixels),       # [K,3,H,W] targets
                "cond": torch.stack(conds),         # [K,12,H,W] d+n+pid+hist
                "mask": torch.stack(masks),         # [K,1,H,W]
                "ref_clip": ref_t,                  # [3,224,224] ONE per seq
                "item": exs[0]["item"]}             # for the per-item prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--index", default=None,
                    help="index file overriding <data>/index.json, e.g. a "
                         "scaling/index_nNN.json tranche with object-disjoint "
                         "splits (its val_seen serves as val)")
    ap.add_argument("--out", default="renderer_ckpt")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--bs", type=int, default=1,
                    help="SEQUENCES per batch (frames per step = bs * seq_len)")
    ap.add_argument("--seq-len", type=int, default=8,
                    help="frames per training sequence")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--sd", default="runwayml/stable-diffusion-v1-5")
    ap.add_argument("--cn", default="lllyasviel/control_v11f1p_sd15_depth")
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--bg-weight", type=float, default=0.08,
                    help="loss weight OUTSIDE the furniture mask (object=1.0). "
                         "Keep well under 0.5: dataset background includes the "
                         "person assembling — high weight teaches rendering them.")
    a = ap.parse_args()

    from accelerate import Accelerator
    from diffusers import (ControlNetModel, DDPMScheduler,
                           StableDiffusionControlNetPipeline)

    acc = Accelerator(mixed_precision="bf16")
    dev = acc.device
    wdt = torch.bfloat16

    # Build via the pipeline so load_ip_adapter (public API) wires the adapter
    # attn processors + image projection into the UNet, then take the parts.
    cn = ControlNetModel.from_pretrained(a.cn)
    # -> 12ch conditioning: depth(0-2) + normal(3-5) + part-ID(6-8) + hist(9-11).
    # From a 3ch pretrained depth CN: depth weights kept, rest zero-init.
    # From a 6ch v2 checkpoint (depth+history): depth kept at 0-2, its trained
    # HISTORY weights remapped to 9-11, normal/pid zero-init — warm start.
    old = cn.controlnet_cond_embedding.conv_in
    if old.in_channels in (3, 6):
        new = torch.nn.Conv2d(12, old.out_channels, old.kernel_size,
                              old.stride, old.padding)
        with torch.no_grad():
            new.weight.zero_()
            new.weight[:, :3] = old.weight[:, :3]          # depth
            if old.in_channels == 6:
                new.weight[:, 9:12] = old.weight[:, 3:6]   # history remap
            new.bias.copy_(old.bias)
        cn.controlnet_cond_embedding.conv_in = new
        cn.register_to_config(conditioning_channels=12)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        a.sd, controlnet=cn, safety_checker=None)
    pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models",
                         weight_name="ip-adapter_sd15.bin")
    tok, text = pipe.tokenizer, pipe.text_encoder.to(dev, wdt)
    vae, unet = pipe.vae.to(dev, wdt), pipe.unet.to(dev, wdt)
    image_encoder = pipe.image_encoder.to(dev, wdt)
    cn = cn.to(dev)
    for m in (text, vae, unet, image_encoder):
        m.requires_grad_(False)
    cn.train()

    noise_sched = DDPMScheduler.from_pretrained(a.sd, subfolder="scheduler")
    with torch.no_grad():
        # one text embedding per item (plus generic fallback), computed once
        names = list(PROMPTS) + ["_generic"]
        texts = [PROMPTS[n] for n in PROMPTS] + [PROMPT]
        ids = tok(texts, padding="max_length",
                  max_length=tok.model_max_length, truncation=True,
                  return_tensors="pt").input_ids.to(dev)
        embs = text(ids)[0]                          # [n_items+1,77,768]
        prompt_emb = {n: embs[i] for i, n in enumerate(names)}

    ds = Sequences(a.data, "train", k=a.seq_len, index=a.index)
    sampler = torch.utils.data.WeightedRandomSampler(
        ds.weights, num_samples=len(ds), replacement=True)
    dl = DataLoader(ds, batch_size=a.bs, sampler=sampler, num_workers=8,
                    pin_memory=True, drop_last=True, persistent_workers=True)
    opt = torch.optim.AdamW(cn.parameters(), lr=a.lr, weight_decay=1e-2)
    cn, opt, dl = acc.prepare(cn, opt, dl)
    print(f"[train] {len(ds)} sequences, bs {a.bs} x {a.seq_len} frames, "
          f"{a.steps} steps")

    os.makedirs(a.out, exist_ok=True)
    step, ema = 0, None
    while step < a.steps:
        for batch in dl:
            if step >= a.steps:
                break
            B, K = batch["pixel"].shape[:2]
            # flatten sequences into the batch: [B,K,...] -> [B*K,...]
            px = batch["pixel"].flatten(0, 1).to(dev, wdt)
            cond = batch["cond"].flatten(0, 1).to(dev, wdt)
            mask = batch["mask"].flatten(0, 1).to(dev, wdt)

            with torch.no_grad():
                lat = vae.encode(px).latent_dist.sample() * vae.config.scaling_factor
                # grounding: ONE reference per sequence, expanded to its K
                # frames; 10% dropout -> robust to weak/absent reference
                img_emb = image_encoder(batch["ref_clip"].to(dev, wdt)).image_embeds
                drop = (torch.rand(img_emb.shape[0], 1, device=dev) < 0.10)
                img_emb = (img_emb * (~drop).to(wdt)).repeat_interleave(K, 0)
            noise = torch.randn_like(lat)
            t = torch.randint(0, noise_sched.config.num_train_timesteps,
                              (lat.shape[0],), device=dev).long()
            noisy = noise_sched.add_noise(lat, noise, t)
            emb = torch.stack([prompt_emb.get(it, prompt_emb["_generic"])
                               for it in batch["item"]]).repeat_interleave(K, 0)

            down, mid = acc.unwrap_model(cn)(noisy, t, encoder_hidden_states=emb,
                                             controlnet_cond=cond, return_dict=False)
            pred = unet(noisy, t, encoder_hidden_states=emb,
                        added_cond_kwargs={"image_embeds": [img_emb.unsqueeze(1)]},
                        down_block_additional_residuals=[d.to(wdt) for d in down],
                        mid_block_additional_residual=mid.to(wdt)).sample

            # masked loss, per-REGION normalized:
            #   loss = mean(err over object) + bg_weight * mean(err over bg)
            # so the object gets a FIXED gradient share regardless of its
            # size -- a plain weighted mean lets big backgrounds drown small
            # objects. Mask is max-pool DILATED before downsampling so thin
            # tubes keep full weight at latent resolution (bilinear shrink
            # would fade a 6px tube to ~0.3 weight).
            m_dil = F.max_pool2d(mask, kernel_size=9, stride=1, padding=4)
            m_lat = F.interpolate(m_dil, size=lat.shape[-2:], mode="bilinear")
            m_lat = (m_lat > 0.2).float()
            err = F.mse_loss(pred.float(), noise.float(), reduction="none")
            obj = (err * m_lat).sum() / (m_lat.sum() * err.shape[1] + 1e-6)
            bg_region = 1 - m_lat
            bg = (err * bg_region).sum() / (bg_region.sum()
                                            * err.shape[1] + 1e-6)
            # false-positive penalty (hard-example mining): an INVENTED
            # structure in the background is a concentrated error hot spot;
            # penalize the worst 10% of bg pixels hard, so "draw 10 tubes,
            # 3 match the mask" stops scoring well, while plausible-but-
            # inexact backgrounds stay cheap under the small mean term.
            err_px = err.mean(1, keepdim=True)          # [B,1,h,w]
            bg_err = (err_px * bg_region).flatten(1)    # zeros inside object
            k = max(1, int(bg_err.shape[1] * 0.10))
            bg_top = torch.topk(bg_err, k, dim=1).values.mean()
            loss = obj + a.bg_weight * bg + 0.5 * bg_top

            acc.backward(loss)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            v = loss.item()
            ema = v if ema is None else 0.98 * ema + 0.02 * v
            if step % 50 == 0:
                print(f"step {step:6d}  loss {v:.4f}  ema {ema:.4f}", flush=True)
            if step % a.val_every == 0 or step == a.steps:
                acc.unwrap_model(cn).save_pretrained(os.path.join(a.out, "controlnet"))
                print(f"[ckpt] step {step} -> {a.out}/controlnet", flush=True)
    print("done")


if __name__ == "__main__":
    main()
