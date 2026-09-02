#!/usr/bin/env python3
"""Build object-held-out tranche indices for the object-count scaling curve.

The question this exists to answer: does held-out-OBJECT performance improve
as the number of training objects grows, and where does it saturate?

`make_index.py` cannot answer it. Its val split is a contiguous frame span
per item, so every item and every video sequence appears in both halves. That
measures interpolation between neighbouring frames of an object the model was
trained on, which is not the claim. Numbers from it look good and mean nothing
about a new object.

Three splits per tranche, and the gap between the last two IS the measurement:

    train        frames from the N training objects
    val_seen     held-out frames of those SAME objects  (fitting check)
    test_unseen  every frame of the held-out objects    (the actual claim)

Two design decisions worth knowing about, because they are the difference
between a curve that means something and one that does not:

**Variants are grouped.** poang_1 / poang_2, mammut_1 / mammut_2, applaro /
applaro_3 are the same product line. Training on one and testing on another is
not a held-out object. Items are split on the base name with the trailing
_<digits> stripped, so a whole family lands on one side.

**The frame budget is fixed across tranches by default.** Otherwise object
count and dataset size grow together and the curve cannot say which one
helped. With --budget, N=4 and N=24 both see the same number of frames, so
the only variable is diversity. Pass --no-budget for the other experiment
(more objects AND more data), but do not read it as a diversity result.

  python make_scaling_index.py                       # default tranches
  python make_scaling_index.py --tranches 4,8,16     # only these
  python make_scaling_index.py --no-budget           # use every frame
  python make_scaling_index.py --list                # what is on disk, then stop

Writes scaling/index_n<N>.json per tranche plus scaling/manifest.json.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "examples_cache.json")
OUT_DIR = os.path.join(HERE, "scaling")
VAL_FRAC = 0.10
MIN_COV = 0.005          # object smaller than 0.5% of frame: nothing to learn


def base_name(item: str) -> str:
    """poang_2 -> poang, bekvam_3017 -> bekvam, stig -> stig."""
    return re.sub(r"_\d+$", "", item)


def _thumb(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L").resize((64, 64),
                                                           Image.BILINEAR),
                      dtype=np.float32) / 255.0


def build_sequence(name: str) -> list[dict]:
    """Scan one maps/<seq> dir into examples with coverage/edge/motion scores.

    Same features as make_index.py so the trainer's sampling behaves
    identically; only the splitting differs.
    """
    maps_dir = os.path.join(HERE, "maps", name)
    pairs_dir = os.path.join(HERE, "pairs", name)
    pj = os.path.join(pairs_dir, "poses.json")
    if not os.path.exists(pj):
        return []
    item = json.load(open(pj))["item"]
    stems = sorted(p[:-10] for p in os.listdir(maps_dir)
                   if p.endswith("_depth.png"))
    exs: list[dict] = []
    prev_d = prev_m = None
    for stem in stems:
        fid = stem.split("frame_")[1]
        ex = {"item": item, "base": base_name(item), "seq": name,
              "real": f"pairs/{name}/frame_{fid}.jpg",
              "depth": f"maps/{name}/{stem}_depth.png",
              "normal": f"maps/{name}/{stem}_normal.png",
              "pid": f"maps/{name}/{stem}_pid.png",
              "mask": f"maps/{name}/{stem}_mask.png"}
        if not all(os.path.exists(os.path.join(HERE, ex[k]))
                   for k in ("real", "depth", "normal", "pid", "mask")):
            continue
        d = _thumb(os.path.join(HERE, ex["depth"]))
        m = _thumb(os.path.join(HERE, ex["mask"])) > 0.5
        ex["cov"] = round(float(m.mean()), 4)
        lo = np.asarray(Image.fromarray((d * 255).astype("uint8"))
                        .resize((8, 8), Image.BILINEAR)
                        .resize((64, 64), Image.BILINEAR),
                        dtype=np.float32) / 255.0
        ex["edge"] = round(float(np.abs(d - lo).mean()), 5)
        ex["mot"] = 0.0 if prev_d is None else \
            round(float(np.abs(d - prev_d).mean())
                  + float((m ^ prev_m).mean()), 5)
        prev_d, prev_m = d, m
        exs.append(ex)
    return exs


def load_examples(rebuild: bool = False) -> dict[str, list[dict]]:
    """seq name -> examples, cached. New sequences are added incrementally."""
    cache: dict[str, list[dict]] = {}
    if os.path.exists(CACHE) and not rebuild:
        cache = json.load(open(CACHE))
    seqs = sorted(os.path.basename(p)
                  for p in glob.glob(os.path.join(HERE, "maps", "*"))
                  if os.path.isdir(p))
    fresh = 0
    for name in seqs:
        if name in cache:
            continue
        exs = build_sequence(name)
        if exs:
            cache[name] = exs
            fresh += 1
            print(f"  scanned {name}: {len(exs)} frames", flush=True)
    for gone in [k for k in cache if k not in seqs]:
        del cache[gone]
    if fresh or rebuild:
        json.dump(cache, open(CACHE, "w"))
        print(f"  cached {len(cache)} sequences -> "
              f"{os.path.relpath(CACHE, HERE)}", flush=True)
    return cache


def group_by_base(cache: dict[str, list[dict]]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for exs in cache.values():
        for ex in exs:
            if ex["cov"] >= MIN_COV:
                out.setdefault(ex["base"], []).append(ex)
    return out


def split_seen(exs: list[dict], rng: random.Random) -> tuple[list, list]:
    """Hold out a contiguous middle span per sequence: neighbouring video
    frames are near-duplicates, so a random frame split leaks."""
    train, val = [], []
    by_seq: dict[str, list[dict]] = {}
    for e in exs:
        by_seq.setdefault(e["seq"], []).append(e)
    for seq, rows in by_seq.items():
        rows.sort(key=lambda r: r["real"])
        k = max(1, int(len(rows) * VAL_FRAC))
        mid = len(rows) // 2
        val += rows[mid: mid + k]
        train += rows[:mid] + rows[mid + k:]
    return train, val


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tranches", default="",
                    help="comma-separated object counts; default is every "
                         "power-of-two step that fits")
    ap.add_argument("--holdout", type=int, default=6,
                    help="object families reserved for test_unseen")
    ap.add_argument("--holdout-items", default="",
                    help="comma-separated base names to hold out, overriding "
                         "the automatic pick")
    ap.add_argument("--budget", type=int, default=0,
                    help="train frames per tranche; 0 = auto (the largest "
                         "budget every tranche can meet)")
    ap.add_argument("--no-budget", action="store_true",
                    help="use every frame; object count and data size then "
                         "grow together and the curve is confounded")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rebuild", action="store_true", help="rescan maps/")
    ap.add_argument("--list", action="store_true",
                    help="print what is on disk and exit")
    a = ap.parse_args()

    rng = random.Random(a.seed)
    print("[scan] loading examples", flush=True)
    cache = load_examples(a.rebuild)
    fams = group_by_base(cache)
    if not fams:
        raise SystemExit("no examples found; run extract_all.py first")

    order = sorted(fams, key=lambda k: -len(fams[k]))
    print(f"\n{len(fams)} object families on disk "
          f"({sum(len(v) for v in fams.values())} usable frames):")
    for k in order:
        items = sorted({e["item"] for e in fams[k]})
        seqs = len({e["seq"] for e in fams[k]})
        print(f"   {k:14s} {len(fams[k]):6d} frames  {seqs:2d} seq  "
              f"{'/'.join(items)}")
    if a.list:
        return

    # ---- held-out families: spread over the size range, not just the tail.
    # Taking the N smallest would make the test set the hardest, sparsest
    # objects and the curve would measure that instead of transfer.
    if a.holdout_items:
        test_fams = [s.strip() for s in a.holdout_items.split(",") if s.strip()]
        missing = [f for f in test_fams if f not in fams]
        if missing:
            raise SystemExit(f"unknown families: {missing}")
    else:
        k = min(a.holdout, max(1, len(fams) // 3))
        idx = np.linspace(0, len(order) - 1, k).round().astype(int)
        test_fams = [order[i] for i in sorted(set(idx.tolist()))]
    train_pool = [f for f in order if f not in test_fams]
    if not train_pool:
        raise SystemExit("holdout consumed every family")

    test_unseen = [e for f in test_fams for e in fams[f]]
    print(f"\nheld out ({len(test_fams)}): {', '.join(test_fams)}"
          f"  -> {len(test_unseen)} test_unseen frames")
    print(f"train pool ({len(train_pool)}): {', '.join(train_pool)}")

    # ---- tranche sizes
    if a.tranches:
        sizes = [int(s) for s in a.tranches.split(",") if s.strip()]
    else:
        sizes, n = [], 4
        while n < len(train_pool):
            sizes.append(n)
            n *= 2
        sizes.append(len(train_pool))
    sizes = sorted({min(n, len(train_pool)) for n in sizes if n >= 1})

    # nested tranches: N=8 contains N=4, so the only thing that changed
    # between two points on the curve is the objects that were added
    per_fam = {f: len(fams[f]) for f in train_pool}
    if a.no_budget:
        budget = 0
    elif a.budget:
        budget = a.budget
    else:
        # the largest budget the SMALLEST tranche can meet without
        # oversampling any one family beyond what it has
        budget = min(sum(sorted((per_fam[f] for f in train_pool[:n]),
                                reverse=True))
                     for n in sizes)
        budget = int(budget * 0.9)          # leave room for the val carve-out

    os.makedirs(OUT_DIR, exist_ok=True)
    manifest = {"seed": a.seed, "holdout": test_fams,
                "budget": budget, "tranches": []}
    print(f"\nbudget: {budget or 'none (every frame)'} train frames per tranche")
    print(f"\n{'N':>3} {'train':>7} {'val_seen':>9} {'test_unseen':>12}  families")

    for n in sizes:
        chosen = train_pool[:n]
        rows = [e for f in chosen for e in fams[f]]
        tr, va = split_seen(rows, rng)
        if budget:
            # even quota per family, then top up from whoever has spare, so a
            # tranche with few large objects is not silently favoured
            by_fam: dict[str, list[dict]] = {}
            for e in tr:
                by_fam.setdefault(e["base"], []).append(e)
            for v in by_fam.values():
                rng.shuffle(v)
            quota, picked = budget // len(chosen), []
            for f in chosen:
                picked += by_fam.get(f, [])[:quota]
            spare = [e for f in chosen for e in by_fam.get(f, [])[quota:]]
            rng.shuffle(spare)
            picked += spare[:max(0, budget - len(picked))]
            tr = picked
        out = {"train": tr, "val_seen": va, "test_unseen": test_unseen,
               "n_objects": n, "objects": chosen, "holdout": test_fams}
        path = os.path.join(OUT_DIR, f"index_n{n:02d}.json")
        json.dump(out, open(path, "w"))
        manifest["tranches"].append(
            {"n": n, "path": os.path.relpath(path, HERE),
             "train": len(tr), "val_seen": len(va),
             "test_unseen": len(test_unseen), "objects": chosen})
        print(f"{n:3d} {len(tr):7d} {len(va):9d} {len(test_unseen):12d}  "
              f"{', '.join(chosen)}")

    json.dump(manifest, open(os.path.join(OUT_DIR, "manifest.json"), "w"),
              indent=2)
    print(f"\nwrote {len(sizes)} tranches -> {os.path.relpath(OUT_DIR, HERE)}/")
    print("test_unseen is IDENTICAL across tranches, so the curve is "
          "comparable point to point.")


if __name__ == "__main__":
    main()
