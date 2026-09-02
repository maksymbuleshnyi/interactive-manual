#!/usr/bin/env python3
"""Extract (real frame, per-part pose) pairs from IKEA-Manuals-at-Work videos.

This is the GROUNDING half of the sim-to-real formulation:

    agent's 3D scene, re-posed to frame     real video frame
    ---------------------------------  -->  ----------------
        (input: clean schematic)          (target: realistic)

The dataset's per-frame annotations (part 6-DoF extrinsics + intrinsics +
masks) are just numbers -- they are NOT tied to the dataset's crude meshes.
Applied to the agent's nicer parts they produce aligned schematic renders in
the agent's own visual style, pixel-registered to the real frame.

This script does the dataset side:
  1. picks the item's video with the most annotated frames
  2. downloads it (yt-dlp)
  3. extracts exactly the annotated frames (by frame_id, chunked select)
  4. writes pairs/<video>/frame_XXXXXX.jpg + poses.json

poses.json is the CONTRACT the agent-geometry renderer consumes. Per frame:
  { "frame_id", "frame_time", "parts": ["3","6"],
    "intrinsics": [3x3 per part], "extrinsics": [3x4 per part],
    "mask_rle": [...per part...] }

What the AGENT side must export to close the loop (not this script's job):
  - GLB/OBJ with each part as a separately named node
  - part_map.json: {agent_node_name: dataset_part_id}
  - one rigid registration per part (agent canonical frame -> dataset
    canonical frame), fit once, e.g. coarse manual + ICP refine

Usage:
  python extract_pairs.py --item stig                 # best video
  python extract_pairs.py --item stig --video-rank 1  # second-best, etc.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
DATA = os.path.join(REPO, "IKEA-Manuals-at-Work", "data", "data.json")


def collect_video_frames(item: dict) -> dict[str, list[dict]]:
    """video_id -> list of annotated frame dicts (all steps merged)."""
    vids: dict[str, list[dict]] = {}
    for st in item.get("steps", []):
        for ve in (st.get("video") or []):
            vid = ve.get("video_id")
            fps = ve.get("fps")
            for fr in (ve.get("frames") or []):
                rec = dict(fr)
                rec["_fps"] = fps
                rec["_step_id"] = st.get("step_id")
                vids.setdefault(vid, []).append(rec)
    for v in vids.values():
        v.sort(key=lambda r: r["frame_id"])
    return vids


def download(video_url: str, out_path: str) -> None:
    if os.path.exists(out_path):
        print(f"[dl] cached: {out_path}")
        return
    subprocess.run(
        ["yt-dlp", "-f", "bestvideo[height<=720][ext=mp4]/best[ext=mp4]/best",
         "--merge-output-format", "mp4", "-o", out_path, video_url],
        check=True)


def probe_frame_count(video_path: str) -> int:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
         "-show_entries", "stream=nb_read_frames", "-of",
         "default=nokey=1:noprint_wrappers=1", video_path],
        capture_output=True, text=True, check=True)
    return int(out.stdout.strip())


def extract_frames(video_path: str, frame_ids: list[int], out_dir: str) -> int:
    """Extract exact frames by index, in chunks (one ffmpeg pass per chunk)."""
    os.makedirs(out_dir, exist_ok=True)
    n_vid = probe_frame_count(video_path)
    in_range = [f for f in frame_ids if f < n_vid]
    if len(in_range) < len(frame_ids):
        print(f"[warn] {len(frame_ids) - len(in_range)} annotated frames beyond "
              f"video end ({n_vid} frames) — dropped (annotation made on a "
              f"longer/different-fps upload)")
    frame_ids = in_range
    todo = [f for f in frame_ids
            if not os.path.exists(os.path.join(out_dir, f"frame_{f:06d}.jpg"))]
    CHUNK = 80
    done = 0
    for i in range(0, len(todo), CHUNK):
        chunk = todo[i:i + CHUNK]
        expr = "+".join(f"eq(n\\,{f})" for f in chunk)
        tmp_pat = os.path.join(out_dir, "_tmp_%04d.jpg")
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", video_path,
             "-vf", f"select='{expr}'", "-vsync", "0", "-q:v", "2", tmp_pat],
            check=True)
        # select preserves frame order -> rename sequential tmp to frame ids
        produced = sorted(p for p in os.listdir(out_dir) if p.startswith("_tmp_"))
        assert len(produced) == len(chunk), \
            f"chunk extraction mismatch: wanted {len(chunk)} got {len(produced)}"
        for tmp, fid in zip(produced, sorted(chunk)):
            os.rename(os.path.join(out_dir, tmp),
                      os.path.join(out_dir, f"frame_{fid:06d}.jpg"))
        done += len(chunk)
        print(f"[frames] {done}/{len(todo)} extracted", flush=True)
    return len(frame_ids)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--item", default="stig")
    ap.add_argument("--video-rank", type=int, default=0,
                    help="0 = video with most annotated frames")
    ap.add_argument("--out", default=os.path.join(HERE, "pairs"))
    a = ap.parse_args()

    data = json.load(open(DATA))
    item = next(x for x in data if x["name"] == a.item)
    vids = collect_video_frames(item)
    ranked = sorted(vids.items(), key=lambda kv: -len(kv[1]))
    print(f"{a.item}: {len(ranked)} annotated videos")
    for i, (vid, frames) in enumerate(ranked):
        mark = " <-- selected" if i == a.video_rank else ""
        print(f"  [{i}] {len(frames):5d} frames  {vid}{mark}")
    vid_url, frames = ranked[a.video_rank]

    vid_key = vid_url.split("v=")[-1][:16]
    out_dir = os.path.join(a.out, f"{a.item}_{vid_key}")
    os.makedirs(out_dir, exist_ok=True)

    video_path = os.path.join(out_dir, "source.mp4")
    download(vid_url, video_path)

    ids = [fr["frame_id"] for fr in frames]
    extract_frames(video_path, ids, out_dir)

    poses = []
    for fr in frames:
        if not os.path.exists(os.path.join(out_dir, f"frame_{fr['frame_id']:06d}.jpg")):
            continue          # dropped (beyond video end) — keep index honest
        poses.append({
            "frame_id": fr["frame_id"],
            "frame_time": fr["frame_time"],
            "step_id": fr["_step_id"],
            "fps": fr["_fps"],
            "parts": fr["parts"],
            "intrinsics": fr["intrinsics"],
            "extrinsics": fr["extrinsics"],
            "mask_rle": fr.get("mask"),
            "image": f"frame_{fr['frame_id']:06d}.jpg",
        })
    with open(os.path.join(out_dir, "poses.json"), "w") as f:
        json.dump({"item": a.item, "video": vid_url, "n_frames": len(poses),
                   "frames": poses}, f)
    print(f"\n[done] {len(poses)} pairs -> {out_dir}")
    print("       poses.json = contract for the agent-geometry renderer")


if __name__ == "__main__":
    main()
