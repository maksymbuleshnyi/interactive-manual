#!/usr/bin/env python3
"""Run the object-count scaling curve: train one checkpoint per tranche,
evaluate every checkpoint on the SAME held-out objects, plot the curve.

This is the experiment behind the paper's scaling section. The tranches are
nested and share one frozen test set, so the only thing that changes between
two points is which objects the model saw. The paper's rule is enforced here,
not remembered: an index whose test objects intersect its train objects is
refused.

Run on the GPU box, from real_grounding/:

  python run_scaling.py --dry-run          # print the commands, do nothing
  python run_scaling.py                    # every tranche in scaling/
  python run_scaling.py --tranches 4,8     # a subset
  python run_scaling.py --steps 6000       # cheaper first pass

Each tranche writes scaling_runs/nNN/ with the checkpoint, eval_test_unseen
and eval_val_seen artifacts, and a contact sheet. Finished tranches (eval
artifact present) are skipped, so the sweep is resumable. curve.json and
curve.txt summarize at the end; the val_seen-vs-test_unseen gap per N is the
generalization measurement itself.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def sh(cmd: list[str], dry: bool) -> None:
    print("  $ " + " ".join(cmd), flush=True)
    if not dry:
        subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tranches", default="",
                    help="comma-separated N values; default all in scaling/")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--seq-len", type=int, default=8)
    ap.add_argument("--per-object", type=int, default=24)
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--out", default=os.path.join(HERE, "scaling_runs"))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    idxs = sorted(glob.glob(os.path.join(HERE, "scaling", "index_n*.json")))
    if a.tranches:
        want = {int(s) for s in a.tranches.split(",")}
        idxs = [p for p in idxs
                if int(os.path.basename(p)[7:9]) in want]
    if not idxs:
        raise SystemExit("no tranche indices; run make_scaling_index.py first")

    # one shared test set or the curve is not a curve
    holds = {tuple(json.load(open(p)).get("holdout", [])) for p in idxs}
    if len(holds) != 1:
        raise SystemExit(f"tranches disagree on holdout: {holds}; "
                         "rebuild with make_scaling_index.py")
    for p in idxs:
        d = json.load(open(p))
        leak = set(d.get("objects", [])) & set(d.get("holdout", []))
        if leak:
            raise SystemExit(f"{p}: train/test leak {sorted(leak)}")

    print(f"[sweep] {len(idxs)} tranches, shared holdout: "
          f"{', '.join(holds.pop())}\n")
    trainer = os.path.join(HERE, "box", "train_renderer.py")
    if not os.path.exists(trainer):                 # repo layout: flat
        trainer = os.path.join(HERE, "train_renderer.py")
    evaler = os.path.join(HERE, "eval_scaling.py")

    for p in idxs:
        n = json.load(open(p))["n_objects"]
        run = os.path.join(a.out, f"n{n:02d}")
        ckpt = os.path.join(run, "controlnet")
        done = os.path.join(run, "eval_test_unseen.json")
        print(f"[tranche n={n}] -> {run}")
        if os.path.exists(done):
            print("  already evaluated, skipping\n")
            continue
        if not os.path.exists(os.path.join(ckpt, "config.json")):
            sh([sys.executable, trainer, "--data", HERE, "--index", p,
                "--out", run, "--steps", str(a.steps), "--bs", str(a.bs),
                "--seq-len", str(a.seq_len)], a.dry_run)
        else:
            print("  checkpoint exists, skipping training")
        for split in ("test_unseen", "val_seen"):
            sh([sys.executable, evaler, "--cn", ckpt, "--index", p,
                "--split", split, "--per-object", str(a.per_object),
                "--res", str(a.res), "--out", run], a.dry_run)
        print()

    if a.dry_run:
        return

    # ---- assemble the curve
    curve = []
    for p in idxs:
        n = json.load(open(p))["n_objects"]
        run = os.path.join(a.out, f"n{n:02d}")
        row = {"n_objects": n}
        for split in ("test_unseen", "val_seen"):
            jp = os.path.join(run, f"eval_{split}.json")
            if os.path.exists(jp):
                s = json.load(open(jp))["summary"]
                for m, v in s.items():
                    row[f"{split}.{m}"] = v["mean"]
                    row[f"{split}.{m}.ci95"] = v["ci95"]
        curve.append(row)
    cj = os.path.join(a.out, "curve.json")
    json.dump(curve, open(cj, "w"), indent=2)

    lines = [f"{'N':>3} {'test l1':>9} {'seen l1':>9} {'gap':>7}   "
             f"(test lpips / dino where available)"]
    for r in curve:
        t = r.get("test_unseen.l1_obj")
        s = r.get("val_seen.l1_obj")
        gap = (t - s) if (t is not None and s is not None) else None
        extra = " ".join(f"{k.split('.')[1]}={r[k]:.3f}" for k in r
                         if k.startswith("test_unseen.") and
                         k.count(".") == 1 and "l1" not in k)
        lines.append(f"{r['n_objects']:>3} "
                     f"{t if t is not None else float('nan'):>9.4f} "
                     f"{s if s is not None else float('nan'):>9.4f} "
                     f"{gap if gap is not None else float('nan'):>7.4f}   "
                     f"{extra}")
    txt = "\n".join(lines)
    open(os.path.join(a.out, "curve.txt"), "w").write(txt + "\n")
    print(txt)
    print(f"\n[sweep] curve -> {cj}")
    print("Read: 'gap' shrinking as N grows = transfer improving with object "
          "diversity.\nFlat test with shrinking gap only via rising seen "
          "error = it never transferred.")


if __name__ == "__main__":
    main()
