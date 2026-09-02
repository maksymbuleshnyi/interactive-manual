# Object-count scaling on real assembly videos

The F1 prototype (`../f1_assembly`) shows the mechanism on one object with
rendered targets. This directory is the generalization experiment the paper
needs: the same structural-conditioning idea trained across many objects with
**real video frames** as targets, evaluated only on **objects held out from
training**, as a function of how many objects were trained on.

Data comes from [IKEA-Manuals-at-Work](https://github.com/yunongliu1/IKEA-video-manual):
32 furniture items, ~21k internet-video frames annotated with per-part 6-DoF
poses, intrinsics, and masks, plus part meshes for every item. The
annotations let us render pixel-registered depth / normal / part-ID maps from
the meshes at each real frame's pose, which is exactly the contract an agent's
blockout would emit.

## Pipeline

Everything is resumable; rerun any step after a failure.

```bash
git clone https://github.com/yunongliu1/IKEA-video-manual ../IKEA-Manuals-at-Work
# (obtain their data/ per their instructions: data.json, parts/)

python extract_all.py --dry-run     # what will be fetched, per item
python extract_all.py               # yt-dlp + exact annotated frames + maps
python make_scaling_index.py        # object-held-out tranche indices
python run_scaling.py --dry-run     # the whole sweep, printed
python run_scaling.py               # train + eval every tranche (GPU)
```

| script | what it does |
| --- | --- |
| `extract_all.py` | every usable (item, video): download, extract exactly the annotated frames, drop the video, render maps, rebuild indices |
| `extract_pairs.py` | one video: frames + `poses.json` (the renderer contract) |
| `batch_render_maps.py` / `render_geometry_maps.py` | depth, camera-space normal, part-ID, mask per frame, rasterized from the part meshes at the annotated pose |
| `make_scaling_index.py` | nested tranches (N = 4, 8, 16, ...) over one frozen set of held-out object families |
| `train_renderer.py` | 12-channel ControlNet (depth, normal, part-ID, history), sequence training with history chaining, `--index` selects a tranche |
| `eval_scaling.py` | held-out-object metrics: object-region L1/PSNR, LPIPS, DINOv2, aggregated per family with bootstrap CIs over families |
| `run_scaling.py` | the sweep: one checkpoint per tranche, both eval splits, `curve.json` |
| `infer_renderer.py` | single frame: any depth map + any reference photo |

## The two rules that make the curve mean something

**Variants are one family.** `poang_1` and `poang_2` are the same product
line; training on one and "generalizing" to the other is leakage. Splits are
made on the base name, and `eval_scaling.py` refuses to run on an index whose
train and test families intersect.

**The frame budget is fixed across tranches.** By default every tranche
trains on the same number of frames, so N objects is the only variable. The
alternative (`--no-budget`) grows data with objects and cannot separate
diversity from volume.

Each tranche reports `val_seen` (held-out frames of trained objects) and
`test_unseen` (all frames of held-out objects). The gap between them, per N,
is the generalization measurement.

## Provenance

Frames are extracted from public YouTube videos referenced by the dataset's
annotations, for research use, following the dataset's own collection method.
Videos are deleted after frame extraction; some may become unavailable, which
`extract_all.py` logs and skips. Check the dataset's license before
redistributing extracted frames.
