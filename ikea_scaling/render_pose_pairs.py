#!/usr/bin/env python3
"""Render (schematic 3D, real photo) pairs at IDENTICAL poses.

For chosen annotated frames of a real assembly video, renders the item's part
meshes through the dataset's per-frame annotated camera (per-part 3x4
extrinsics + 3x3 intrinsics) and composes [render | real frame] side by side.
This is the visual proof of the sim-to-real pairing: same pose, two worlds.

Usage:
  python render_pose_pairs.py --pairs pairs/stig_YqWk45Sawt4 --n 3
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))

PALETTE = [(31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40),
           (148, 103, 189), (140, 86, 75), (227, 119, 194), (127, 127, 127),
           (188, 189, 34), (23, 190, 207), (174, 199, 232), (255, 187, 120)]


def load_obj(path: str):
    vs, tris = [], []
    for line in open(path):
        if line.startswith("v "):
            vs.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("f "):
            idx = [int(t.split("/")[0]) - 1 for t in line.split()[1:]]
            for k in range(1, len(idx) - 1):
                tris.append([idx[0], idx[k], idx[k + 1]])
    return np.asarray(vs, float), np.asarray(tris, int)


def render_frame(fr: dict, meshes: dict, w: int, h: int) -> Image.Image:
    """Rasterize all annotated parts through their per-part extrinsics."""
    im = Image.new("RGB", (w, h), (250, 249, 246))
    draw = ImageDraw.Draw(im)
    polys = []          # (depth, screen_tri, color)
    for i, pid_s in enumerate(fr["parts"]):
        try:
            K = np.asarray(fr["intrinsics"][i], float).reshape(3, 3)
            E = np.asarray(fr["extrinsics"][i], float)
            E = E.reshape(4, 4) if E.size == 16 else E.reshape(3, 4)
        except (ValueError, IndexError, TypeError):
            continue                       # malformed annotation entry: skip part
        R, t = E[:3, :3], E[:3, 3]
        for pid in (int(p) for p in str(pid_s).split(",") if p.strip().isdigit()):
            if pid not in meshes:
                continue
            v, tris = meshes[pid]
            vc = v @ R.T + t                          # to camera space
            z = vc[:, 2]
            px = K[0, 0] * vc[:, 0] / np.maximum(z, 1e-6) + K[0, 2]
            py = K[1, 1] * vc[:, 1] / np.maximum(z, 1e-6) + K[1, 2]
            col = PALETTE[pid % len(PALETTE)]
            for tri in tris:
                tz = z[tri]
                if (tz <= 1e-4).any():
                    continue
                pts = [(px[j], py[j]) for j in tri]
                xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
                if max(xs) < 0 or min(xs) > w or max(ys) < 0 or min(ys) > h:
                    continue
                polys.append((float(tz.mean()), pts, col))
    polys.sort(key=lambda p: -p[0])                   # painter: far to near
    for _, pts, col in polys:
        draw.polygon(pts, fill=col)
    return im


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=os.path.join(HERE, "pairs", "stig_YqWk45Sawt4"))
    ap.add_argument("--item", default="stig")
    ap.add_argument("--category", default="Chair")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--overlay", action="store_true",
                    help="add a 50/50 blend panel to check alignment")
    a = ap.parse_args()

    meta = json.load(open(os.path.join(a.pairs, "poses.json")))
    frames = [f for f in meta["frames"] if len(f["parts"]) >= 2]
    # spread picks across the assembly: early / middle / late
    picks = [frames[int(r * (len(frames) - 1))]
             for r in np.linspace(0.15, 0.95, a.n)]

    mesh_dir = os.path.join(REPO, "IKEA-Manuals-at-Work", "data", "parts",
                            a.category, a.item)
    meshes = {int(os.path.splitext(os.path.basename(p))[0]): load_obj(p)
              for p in glob.glob(os.path.join(mesh_dir, "*.obj"))}
    print(f"{len(meshes)} part meshes, {len(frames)} usable frames")

    for k, fr in enumerate(picks):
        real = Image.open(os.path.join(a.pairs, fr["image"])).convert("RGB")
        w, h = real.size
        schem = render_frame(fr, meshes, w, h)
        panels = [schem, real]
        if a.overlay:
            panels.append(Image.blend(schem, real, 0.5))
        out = Image.new("RGB", (w * len(panels), h + 34), (15, 15, 18))
        d = ImageDraw.Draw(out)
        labels = ["3D model (annotated pose)", "real frame (same pose)", "50/50 overlay"]
        for i, (p, lab) in enumerate(zip(panels, labels)):
            out.paste(p, (i * w, 34))
            d.text((i * w + 10, 9), lab, fill=(220, 220, 230))
        d.text((w * len(panels) - 330, 9),
               f"frame {fr['frame_id']} · step {fr['step_id']} · parts {fr['parts']}",
               fill=(150, 160, 180))
        name = os.path.join(HERE, f"pair_{k}_frame{fr['frame_id']:06d}.jpg")
        out.save(name, quality=90)
        print("wrote", name)


if __name__ == "__main__":
    main()
