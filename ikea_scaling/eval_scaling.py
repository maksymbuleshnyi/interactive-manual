#!/usr/bin/env python3
"""Held-out-OBJECT evaluation of a renderer checkpoint. The number this
produces is the one the scaling curve is made of.

For each frame of the test_unseen split (objects the checkpoint never trained
on), it builds the structural conditioning, picks a reference photo of that
object from a distant part of the same video, generates, and scores against
the real frame INSIDE the object mask. Scene-level metrics on this data are
meaningless -- the background of a real assembly video is a person and a
living room the model was never asked to reproduce -- so every fidelity metric
here is object-region.

Cold-start protocol: the history channels are zeros, the way a session begins.
Rollout drift is a separate evaluation; mixing the two in one number would
hide which one failed.

Aggregation follows the paper's rule: means per object FAMILY, confidence
interval by bootstrap over families, never over correlated frames. With one
family the CI is refused rather than fabricated.

  python eval_scaling.py --cn scaling_runs/n04/controlnet \
      --index scaling/index_n04.json
  python eval_scaling.py --cn ... --index ... --split val_seen   # sanity arm

Writes <out>/eval_<split>.json (every number + settings, traceable to a run)
and <out>/sheet_<split>.jpg (depth | pid | generated | real, per family).
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
PROMPT = "a photo of furniture being assembled indoors, realistic"
NEG = "cartoon, lowres, text, watermark, duplicated parts"


def cluster_bootstrap(per_family: dict[str, list[float]], n: int = 2000,
                      seed: int = 0) -> tuple[float, float]:
    groups = [v for v in per_family.values() if v]
    if len(groups) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = np.arange(len(groups))
    means = np.empty(n)
    for i in range(n):
        pick = rng.choice(idx, size=len(groups), replace=True)
        means[i] = float(np.mean([np.mean(groups[j]) for j in pick]))
    return (float(np.percentile(means, 2.5)),
            float(np.percentile(means, 97.5)))


def pick_reference(rows: list[dict], i: int) -> dict:
    """A real photo of the SAME object from a distant moment of the video:
    far enough that the eval frame is not its near-duplicate, chosen for
    object coverage so the reference actually shows the thing."""
    far = [r for r in rows if abs(rows.index(r) - i) > max(3, len(rows) // 4)]
    pool = far or [r for j, r in enumerate(rows) if j != i] or rows
    return max(pool, key=lambda r: r.get("cov", 0))


def bbox_crop(arr: np.ndarray, mask: np.ndarray, pad: int = 16):
    ys, xs = np.where(mask > 0.5)
    if len(ys) == 0:
        return None
    y0, y1 = max(0, ys.min() - pad), min(arr.shape[0], ys.max() + pad)
    x0, x1 = max(0, xs.min() - pad), min(arr.shape[1], xs.max() + pad)
    return arr[y0:y1, x0:x1]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cn", required=True)
    ap.add_argument("--index", required=True,
                    help="a scaling/index_nNN.json tranche")
    ap.add_argument("--split", default="test_unseen",
                    choices=["test_unseen", "val_seen"])
    ap.add_argument("--root", default=HERE,
                    help="dir the index paths are relative to")
    ap.add_argument("--per-object", type=int, default=24)
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--cfg", type=float, default=4.5)
    ap.add_argument("--ip-scale", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sd", default="runwayml/stable-diffusion-v1-5")
    ap.add_argument("--out", default=None,
                    help="default: the checkpoint's parent dir")
    a = ap.parse_args()

    import torch
    import torch.nn.functional as F
    from diffusers import (ControlNetModel, StableDiffusionControlNetPipeline,
                           UniPCMultistepScheduler)

    dev, wdt = "cuda", torch.bfloat16
    cn = ControlNetModel.from_pretrained(a.cn, torch_dtype=wdt)
    ch = int(getattr(cn.config, "conditioning_channels", 3) or 3)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        a.sd, controlnet=cn, torch_dtype=wdt, safety_checker=None)
    pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models",
                         weight_name="ip-adapter_sd15.bin")
    pipe.set_ip_adapter_scale(a.ip_scale)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to(dev)
    pipe.set_progress_bar_config(disable=True)
    print(f"[eval] {a.cn}: {ch} conditioning channels", flush=True)

    # optional perceptual metrics; the artifact records which ones ran
    lpips_fn = dino = None
    try:
        import lpips as lpips_mod
        lpips_fn = lpips_mod.LPIPS(net="alex").to(dev).eval()
    except Exception as e:
        print(f"[eval] lpips unavailable ({e}); skipping")
    try:
        dino = torch.hub.load("facebookresearch/dinov2",
                              "dinov2_vits14").to(dev).eval()
    except Exception as e:
        print(f"[eval] DINOv2 unavailable ({e}); skipping")

    idx = json.load(open(a.index))
    rows = idx[a.split]
    if a.split == "test_unseen":
        trained_on = set(idx.get("objects", []))
        leak = {r["base"] for r in rows} & trained_on
        if leak:
            raise SystemExit(f"SPLIT LEAK: {sorted(leak)} are in both train "
                             f"and {a.split}; refusing to produce a number")
    by_seq: dict[str, list[dict]] = {}
    for r in rows:
        by_seq.setdefault(r["seq"], []).append(r)
    for v in by_seq.values():
        v.sort(key=lambda r: r["real"])

    # spread the per-object budget across that object's sequences
    by_fam: dict[str, list[tuple[dict, dict]]] = {}   # fam -> [(ex, ref)]
    for seq, srows in by_seq.items():
        fam = srows[0]["base"]
        take = srows[:: max(1, len(srows) // max(1, a.per_object))]
        for r in take:
            by_fam.setdefault(fam, []).append(
                (r, pick_reference(srows, srows.index(r))))
    for fam in by_fam:
        by_fam[fam] = by_fam[fam][: a.per_object]
    n_total = sum(len(v) for v in by_fam.values())
    print(f"[eval] {a.split}: {len(by_fam)} families, {n_total} frames",
          flush=True)

    def im(rel, mode="RGB"):
        return Image.open(os.path.join(a.root, rel)).convert(mode) \
                    .resize((a.res, a.res), Image.BILINEAR)

    def chan(img):
        t = torch.from_numpy(np.asarray(img, np.float32) / 255.0)
        return t[None].repeat(3, 1, 1) if t.ndim == 2 else t.permute(2, 0, 1)

    def build_cond(r):
        parts = [chan(im(r["depth"], "L"))]
        if ch >= 12:
            parts += [chan(im(r["normal"])), chan(im(r["pid"]))]
        if ch >= 6:
            parts.append(torch.zeros(3, a.res, a.res))     # cold-start history
        return torch.cat(parts, 0)[None].to(dev, wdt)

    metrics = ["l1_obj", "psnr_obj"] + (["lpips_obj"] if lpips_fn else []) \
        + (["dino_obj"] if dino else [])
    per_fam: dict[str, dict[str, list[float]]] = {
        f: {m: [] for m in metrics} for f in by_fam}
    frames_out: list[dict] = []
    sheet: list[Image.Image] = []
    t0 = time.time()

    with torch.no_grad():
        for fam, pairs in by_fam.items():
            for k, (r, ref_row) in enumerate(pairs):
                ref = Image.open(os.path.join(a.root, ref_row["real"])) \
                           .convert("RGB")
                g = torch.Generator(dev).manual_seed(a.seed)
                gen = pipe(PROMPT, image=build_cond(r), ip_adapter_image=ref,
                           negative_prompt=NEG if a.cfg > 1 else None,
                           num_inference_steps=a.steps, guidance_scale=a.cfg,
                           generator=g,
                           height=a.res, width=a.res).images[0]
                real = im(r["real"])
                mask = np.asarray(im(r["mask"], "L"), np.float32) / 255.0
                if mask.sum() < 32:
                    continue
                ga = np.asarray(gen.resize((a.res, a.res)), np.float32) / 255
                ra = np.asarray(real, np.float32) / 255
                m3 = mask[..., None]
                l1 = float((np.abs(ga - ra) * m3).sum() / (m3.sum() * 3))
                mse = float((((ga - ra) ** 2) * m3).sum() / (m3.sum() * 3))
                row = {"family": fam, "seq": r["seq"], "real": r["real"],
                       "ref": ref_row["real"],
                       "l1_obj": l1,
                       "psnr_obj": float(10 * np.log10(1.0 / max(mse, 1e-9)))}
                gc, rc = bbox_crop(ga, mask), bbox_crop(ra, mask)
                if gc is not None and lpips_fn is not None:
                    def tt(x):
                        t = torch.from_numpy(x).permute(2, 0, 1)[None]
                        return F.interpolate(t * 2 - 1, size=(224, 224),
                                             mode="bilinear").to(dev)
                    row["lpips_obj"] = float(lpips_fn(tt(gc), tt(rc)).item())
                if gc is not None and dino is not None:
                    def dfeat(x):
                        t = torch.from_numpy(x).permute(2, 0, 1)[None]
                        t = F.interpolate(t, size=(224, 224), mode="bilinear")
                        mean = torch.tensor([0.485, 0.456, 0.406])
                        std = torch.tensor([0.229, 0.224, 0.225])
                        t = (t - mean[None, :, None, None]) \
                            / std[None, :, None, None]
                        return dino(t.to(dev))
                    fg, fr = dfeat(gc), dfeat(rc)
                    row["dino_obj"] = float(F.cosine_similarity(fg, fr).item())
                for m in metrics:
                    if m in row:
                        per_fam[fam][m].append(row[m])
                frames_out.append(row)
                if k < 4:                       # contact sheet, first few
                    strip = Image.new("RGB", (a.res * 4, a.res))
                    for c, img in enumerate([im(r["depth"]), im(r["pid"]),
                                             gen.resize((a.res, a.res)),
                                             real]):
                        strip.paste(img, (c * a.res, 0))
                    sheet.append(strip)
            done = sum(len(v["l1_obj"]) for v in per_fam.values())
            print(f"  {fam}: {len(per_fam[fam]['l1_obj'])} frames "
                  f"({done}/{n_total}, {time.time()-t0:.0f}s)", flush=True)

    out_dir = a.out or os.path.dirname(a.cn.rstrip("/"))
    os.makedirs(out_dir, exist_ok=True)
    summary = {}
    print(f"\n  {'metric':10s} {'mean':>8} {'95% CI over families':>22}")
    for m in metrics:
        fam_means = {f: v[m] for f, v in per_fam.items() if v[m]}
        mean = float(np.mean([np.mean(v) for v in fam_means.values()]))
        lo, hi = cluster_bootstrap(fam_means)
        summary[m] = {"mean": mean, "ci95": [lo, hi],
                      "per_family": {f: float(np.mean(v))
                                     for f, v in fam_means.items()}}
        print(f"  {m:10s} {mean:>8.4f}      [{lo:.4f}, {hi:.4f}]")

    art = {"checkpoint": os.path.abspath(a.cn), "index": os.path.abspath(a.index),
           "split": a.split, "channels": ch, "res": a.res, "steps": a.steps,
           "cfg": a.cfg, "ip_scale": a.ip_scale, "seed": a.seed,
           "n_objects_trained": idx.get("n_objects"),
           "objects_trained": idx.get("objects"),
           "holdout": idx.get("holdout"),
           "metrics_available": metrics, "summary": summary,
           "frames": frames_out, "wall_seconds": round(time.time() - t0, 1)}
    jp = os.path.join(out_dir, f"eval_{a.split}.json")
    json.dump(art, open(jp, "w"), indent=2)
    print(f"\n  artifact -> {jp}")
    if sheet:
        H = min(len(sheet), 24)
        board = Image.new("RGB", (a.res * 4, a.res * H))
        for i, s in enumerate(sheet[:H]):
            board.paste(s, (0, i * a.res))
        sp = os.path.join(out_dir, f"sheet_{a.split}.jpg")
        board.resize((a.res * 2, a.res * H // 2)).save(sp, quality=88)
        print(f"  contact sheet -> {sp}  (depth | part-ID | generated | real)")


if __name__ == "__main__":
    main()
