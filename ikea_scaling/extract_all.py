#!/usr/bin/env python3
"""Extract EVERY usable item from IKEA-Manuals-at-Work, not just the seven
already on disk.

The dataset holds 32 distinct items / ~21k annotated frames with part meshes
for all of them; the current pairs/ covers 7 items / ~7k frames. The scaling
claim needs the rest: with 7 families, holding out enough objects for a test
set leaves almost nothing to scale over.

Per (item, video) this drives the existing single-video pipeline:
  1. extract_pairs.py  --item X --video-rank R   (yt-dlp + exact frames)
  2. delete source.mp4                            (2 GB per ~10 videos)
then one batch_render_maps.py pass for every new pairs dir, and finally
make_scaling_index.py to rebuild the tranches.

Resumable: a pairs dir with poses.json is skipped, so rerunning after a
network failure only does what is missing. Videos do disappear from YouTube;
a failed download is logged and skipped, never fatal, and the report at the
end says exactly what the dataset now contains.

  python extract_all.py --dry-run             # plan only: items, videos, est
  python extract_all.py                       # everything missing
  python extract_all.py --items poang_2,pahl  # just these
  python extract_all.py --videos-per-item 2   # cap sequences per item
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "IKEA-Manuals-at-Work", "data", "data.json")


def videos_of(entry: dict) -> list[tuple[str, int]]:
    """[(video_url, n_annotated_frames)] sorted by frame count, descending.
    Mirrors extract_pairs.collect_video_frames so ranks agree with it."""
    vids: dict[str, int] = {}
    for st in entry.get("steps", []):
        for ve in (st.get("video") or []):
            vid = ve.get("video_id")
            if vid:
                vids[vid] = vids.get(vid, 0) + len(ve.get("frames") or [])
    return sorted(vids.items(), key=lambda kv: -kv[1])


def pairs_dir_for(item: str, url: str) -> str:
    return os.path.join(HERE, "pairs", f"{item}_{url.split('v=')[-1][:16]}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--items", default="",
                    help="comma-separated item names; default every item")
    ap.add_argument("--videos-per-item", type=int, default=3,
                    help="videos per item, best-annotated first; more "
                         "sequences beat more frames of one sequence")
    ap.add_argument("--min-frames", type=int, default=30,
                    help="skip videos with fewer annotated frames than this")
    ap.add_argument("--keep-video", action="store_true",
                    help="keep source.mp4 (default: delete after extraction)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-maps", action="store_true",
                    help="skip the map-render + index rebuild at the end")
    a = ap.parse_args()

    data = json.load(open(DATA))
    # entries can share a name across categories (applaro Bench/Chair/Table
    # share one video pool); collapse by name so a video is fetched once
    by_name: dict[str, dict] = {}
    for e in data:
        by_name.setdefault(e["name"], e)
    names = ([s.strip() for s in a.items.split(",") if s.strip()]
             if a.items else sorted(by_name))
    unknown = [n for n in names if n not in by_name]
    if unknown:
        raise SystemExit(f"unknown items {unknown}")

    plan: list[tuple[str, int, str, int]] = []   # (item, rank, url, frames)
    done: list[tuple[str, str, int]] = []
    for name in names:
        for rank, (url, n) in enumerate(videos_of(by_name[name])):
            if rank >= a.videos_per_item or n < a.min_frames:
                continue
            d = pairs_dir_for(name, url)
            if os.path.exists(os.path.join(d, "poses.json")):
                done.append((name, url, n))
            else:
                plan.append((name, rank, url, n))

    est = sum(p[3] for p in plan)
    print(f"[plan] {len(done)} sequences already extracted, "
          f"{len(plan)} to fetch (~{est} frames, "
          f"~{est * 115 // 1024} MB of jpg)")
    for name, rank, url, n in plan:
        print(f"   {name:14s} rank {rank}  {n:5d} frames  {url}")
    if a.dry_run or not plan:
        if not plan:
            print("[plan] nothing to do")
        return

    ok, failed = [], []
    t0 = time.time()
    for i, (name, rank, url, n) in enumerate(plan):
        print(f"\n[{i+1}/{len(plan)}] {name} rank {rank} "
              f"({n} frames)", flush=True)
        r = subprocess.run(
            [sys.executable, os.path.join(HERE, "extract_pairs.py"),
             "--item", name, "--video-rank", str(rank)])
        d = pairs_dir_for(name, url)
        if r.returncode == 0 and os.path.exists(
                os.path.join(d, "poses.json")):
            ok.append((name, url))
            src = os.path.join(d, "source.mp4")
            if not a.keep_video and os.path.exists(src):
                os.remove(src)
        else:
            # a dead or region-locked video is data loss, not a crash: note
            # it and move on, the report below keeps the inventory honest
            failed.append((name, url, r.returncode))
            print(f"[fail] {name} {url} (rc {r.returncode})", flush=True)

    print(f"\n[extract] {len(ok)} ok, {len(failed)} failed, "
          f"{(time.time()-t0)/60:.0f} min")
    for name, url, rc in failed:
        print(f"   FAILED {name:14s} {url}  rc={rc}")

    if a.no_maps:
        return
    if ok:
        print("\n[maps] rendering conditioning maps for all pairs", flush=True)
        subprocess.run([sys.executable,
                        os.path.join(HERE, "batch_render_maps.py")],
                       check=True)
        print("\n[index] rebuilding scaling tranches", flush=True)
        subprocess.run([sys.executable,
                        os.path.join(HERE, "make_scaling_index.py"),
                        "--rebuild"], check=True)

    inv: dict[str, int] = {}
    for d in sorted(os.listdir(os.path.join(HERE, "pairs"))):
        pj = os.path.join(HERE, "pairs", d, "poses.json")
        if os.path.exists(pj):
            m = json.load(open(pj))
            inv[m["item"]] = inv.get(m["item"], 0) + m["n_frames"]
    print(f"\n[inventory] {len(inv)} items, {sum(inv.values())} frames:")
    for k in sorted(inv, key=lambda k: -inv[k]):
        print(f"   {k:14s} {inv[k]:6d}")


if __name__ == "__main__":
    main()
