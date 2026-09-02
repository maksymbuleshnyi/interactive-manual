#!/usr/bin/env python3
"""Visualize the style-free conditioning maps (depth / normal / part-ID) that
would replace schematic RGB as the agent<->renderer contract.

Renders them from the dataset meshes at one annotated real-frame pose, next to
the real frame, so the mapping is concrete.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
from PIL import Image, ImageDraw

import render_pose_pairs as rpp

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))


def render_maps(fr: dict, meshes: dict, w: int, h: int):
    """Painter-rasterize depth, camera-space normal, and part-ID maps."""
    depth_im = Image.new("L", (w, h), 0)
    norm_im = Image.new("RGB", (w, h), (128, 128, 255))
    pid_im = Image.new("RGB", (w, h), (0, 0, 0))
    d_draw, n_draw, p_draw = (ImageDraw.Draw(depth_im),
                              ImageDraw.Draw(norm_im), ImageDraw.Draw(pid_im))
    polys = []
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
            vc = v @ R.T + t
            z = vc[:, 2]
            px = K[0, 0] * vc[:, 0] / np.maximum(z, 1e-6) + K[0, 2]
            py = K[1, 1] * vc[:, 1] / np.maximum(z, 1e-6) + K[1, 2]
            for tri in tris:
                tz = z[tri]
                if (tz <= 1e-4).any():
                    continue
                pts = [(px[j], py[j]) for j in tri]
                xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
                if max(xs) < 0 or min(xs) > w or max(ys) < 0 or min(ys) > h:
                    continue
                # camera-space triangle normal
                a, b, c = vc[tri]
                n = np.cross(b - a, c - a)
                nn = np.linalg.norm(n)
                if nn < 1e-12:
                    continue
                n = n / nn
                if n[2] > 0:            # face the camera
                    n = -n
                polys.append((float(tz.mean()), pts, n, pid))
    polys.sort(key=lambda p: -p[0])
    if not polys:
        return depth_im, norm_im, pid_im
    zs = [p[0] for p in polys]
    z_near, z_far = min(zs), max(zs)
    for zmean, pts, n, pid in polys:
        shade = int(255 * (1.0 - (zmean - z_near) / max(z_far - z_near, 1e-6)))
        d_draw.polygon(pts, fill=shade)                       # near=bright
        rgb = tuple(int((c + 1) * 127.5) for c in (n[0], -n[1], -n[2]))
        n_draw.polygon(pts, fill=rgb)                          # normal -> RGB
        p_draw.polygon(pts, fill=rpp.PALETTE[pid % len(rpp.PALETTE)])
    return depth_im, norm_im, pid_im


def main() -> None:
    pairs_dir = os.path.join(HERE, "pairs", "stig_YqWk45Sawt4")
    meta = json.load(open(os.path.join(pairs_dir, "poses.json")))
    def n_ids(f):   # parts entries may be comma groups like '0,4,7'
        return sum(len(str(p).split(",")) for p in f["parts"])
    frames = sorted(meta["frames"], key=n_ids)
    fr = frames[-1]                                   # most parts in view

    mesh_dir = os.path.join(REPO, "IKEA-Manuals-at-Work", "data", "parts",
                            "Chair", "stig")
    meshes = {int(os.path.splitext(os.path.basename(p))[0]): rpp.load_obj(p)
              for p in glob.glob(os.path.join(mesh_dir, "*.obj"))}

    real = Image.open(os.path.join(pairs_dir, fr["image"])).convert("RGB")
    w, h = real.size
    depth, norm, pid = render_maps(fr, meshes, w, h)

    panels = [(real, "real frame (target)"),
              (depth.convert("RGB"), "DEPTH map (condition)"),
              (norm, "NORMAL map (condition)"),
              (pid, "PART-ID map (condition)")]
    out = Image.new("RGB", (w * 2, h * 2 + 68), (15, 15, 18))
    d = ImageDraw.Draw(out)
    for i, (p, lab) in enumerate(panels):
        x, y = (i % 2) * w, 34 + (i // 2) * (h + 34)
        out.paste(p, (x, y))
        d.text((x + 10, y - 25), lab, fill=(220, 220, 230))
    path = os.path.join(HERE, f"geometry_maps_frame{fr['frame_id']:06d}.jpg")
    out.save(path, quality=90)
    print("wrote", path)


if __name__ == "__main__":
    main()
