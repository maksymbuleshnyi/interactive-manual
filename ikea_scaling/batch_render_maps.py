#!/usr/bin/env python3
"""Batch-render style-free conditioning maps + loss masks for every extracted
pair, for all items.

Per annotated frame, writes into maps/<pairs_dirname>/:
    frame_XXXXXX_depth.png    L    near=bright, per-frame normalized
    frame_XXXXXX_normal.png   RGB  camera-space normals
    frame_XXXXXX_pid.png      RGB  flat color per part
    frame_XXXXXX_mask.png     L    union of part RLE masks (255 = furniture)

The three maps are the CONDITION (style-free geometry, same contract any 3D
engine can emit); the mask restricts the training loss to furniture pixels so
people/clutter in the real frames never enter the objective.

Usage:
  python batch_render_maps.py                 # all pairs/ dirs, all frames
  python batch_render_maps.py --limit 20      # smoke test
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from multiprocessing import Pool

import numpy as np
from PIL import Image

import render_pose_pairs as rpp
from render_geometry_maps import render_maps

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))


# --------------------------------------------------- COCO compressed RLE

def rle_decode(counts: str, h: int, w: int) -> np.ndarray:
    """Decode COCO compressed RLE string -> HxW uint8 {0,1}. Pure python."""
    cnts: list[int] = []
    i = 0
    while i < len(counts):
        x, k, more = 0, 0, True
        while more:
            c = ord(counts[i]) - 48
            x |= (c & 0x1F) << (5 * k)
            more = bool(c & 0x20)
            i += 1
            k += 1
            if not more and (c & 0x10):
                x |= -1 << (5 * k)
        if len(cnts) > 2:
            x += cnts[-2]
        cnts.append(x)
    img = np.zeros(h * w, dtype=np.uint8)
    pos, val = 0, 0
    for c in cnts:
        img[pos:pos + c] = val
        pos += c
        val = 1 - val
    return img.reshape((w, h)).T          # column-major storage


def frame_mask(fr: dict, h: int, w: int) -> np.ndarray:
    out = np.zeros((h, w), dtype=np.uint8)
    for m in (fr.get("mask_rle") or []):
        if isinstance(m, dict) and "counts" in m:
            mh, mw = m["size"]
            dec = rle_decode(m["counts"], mh, mw)
            if (mh, mw) != (h, w):
                dec = np.asarray(Image.fromarray(dec * 255)
                                 .resize((w, h), Image.NEAREST)) // 255
            out |= dec
    return out * 255


# --------------------------------------------------------- worker setup

_G: dict = {}


def _init(pairs_dir: str, item: str, category: str):
    mesh_dir = os.path.join(REPO, "IKEA-Manuals-at-Work", "data", "parts",
                            category, item)
    _G["meshes"] = {int(os.path.splitext(os.path.basename(p))[0]): rpp.load_obj(p)
                    for p in glob.glob(os.path.join(mesh_dir, "*.obj"))}
    _G["pairs_dir"] = pairs_dir
    _G["out_dir"] = os.path.join(HERE, "maps", os.path.basename(pairs_dir))
    os.makedirs(_G["out_dir"], exist_ok=True)


def _one(fr: dict) -> str | None:
    stem = f"frame_{fr['frame_id']:06d}"
    out = _G["out_dir"]
    done = all(os.path.exists(os.path.join(out, f"{stem}_{s}.png"))
               for s in ("depth", "normal", "pid", "mask"))
    if done:
        return None
    real_p = os.path.join(_G["pairs_dir"], fr["image"])
    if not os.path.exists(real_p):
        return f"{stem}: real frame missing"
    with Image.open(real_p) as im:
        w, h = im.size
    depth, normal, pid = render_maps(fr, _G["meshes"], w, h)
    depth.save(os.path.join(out, f"{stem}_depth.png"))
    normal.save(os.path.join(out, f"{stem}_normal.png"))
    pid.save(os.path.join(out, f"{stem}_pid.png"))
    Image.fromarray(frame_mask(fr, h, w)).save(
        os.path.join(out, f"{stem}_mask.png"))
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-glob", default=os.path.join(HERE, "pairs", "*"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    data = json.load(open(os.path.join(REPO, "IKEA-Manuals-at-Work", "data",
                                       "data.json")))
    cat_of = {it["name"]: it["category"] for it in data}

    for pairs_dir in sorted(glob.glob(a.pairs_glob)):
        poses_p = os.path.join(pairs_dir, "poses.json")
        if not os.path.exists(poses_p):
            continue
        meta = json.load(open(poses_p))
        item = meta["item"]
        frames = meta["frames"][: a.limit]
        print(f"[{item}] {len(frames)} frames -> maps/{os.path.basename(pairs_dir)}")
        with Pool(a.workers, initializer=_init,
                  initargs=(pairs_dir, item, cat_of[item])) as pool:
            for i, err in enumerate(pool.imap_unordered(_one, frames, chunksize=8)):
                if err:
                    print("  !", err)
                if (i + 1) % 200 == 0:
                    print(f"  {i + 1}/{len(frames)}", flush=True)
    print("done")


if __name__ == "__main__":
    main()
