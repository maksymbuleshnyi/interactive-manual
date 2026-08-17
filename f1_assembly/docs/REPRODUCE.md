# Reproducing the F1 result

Zero to a running demo. Roughly two hours of wall clock, most of it capture
and training, plus whatever it takes to obtain the STLs.

Every stage ends with a check. They are there because two full training runs
were wasted on data that looked fine and was not, so verify the inputs before
spending the GPU.

---

## 0. Prerequisites

```bash
pip install -r requirements.txt      # torch, diffusers, fastapi, trimesh, ...
```

Put the 27 kit STLs in `viewer/Separate STLs/`. The filenames must match the
`file` key of each entry in `viewer_manifest.json`. Nothing works without them:
they are the training target.

**Check:**

```bash
cd viewer && python3 -c "
import json, os
m = json.load(open('viewer_manifest.json'))
f = [v['file'] for v in m['parts'].values() if v.get('file')]
print(sum(os.path.exists(x) for x in f), 'of', len(f), 'STLs present')"
```

Expect `27 of 27`.

---

## 1. Open the manual

```bash
cd viewer && python3 capture_server.py 8078 traindata
open "http://localhost:8078/index.html"
```

The left panel is a line-art schematic built from primitives; the right is a
lit workshop scene using the real CAD. Hover names parts, the step list walks
the build, and arrows plus glue dots come from measured contact patches, not
from guesses.

**Check:** all 23 parts load, the step list runs 1 to 17, and the wheels enter
from outside the car.

---

## 2. Capture the training data

Chrome, tab in the foreground. Background tabs get throttled and the capture
crawls.

```bash
open "http://localhost:8078/index.html?traj=1"    # 32 interactions, 4352 frames
# wait for the tab title to read TRAJECTORIES DONE, then
open "http://localhost:8078/index.html?pairs=1"   # 1632 independent viewpoints
# optional, only for presentation videos
open "http://localhost:8078/index.html?sch=1&seq=papaya_v0"
```

About 45 minutes. Each frame writes four images: depth (`cond_`), the
photographic target (`tgt_`), an exact object mask (`mask_`) and the flat
per-part colour map (`alb_`). Trajectories also record the camera action and
the assembly step, and mark `cut` on the frame where a new part appears.

**Check, and do not skip this one:**

```bash
python3 - <<'EOF'
import json, numpy as np
from PIL import Image
root = 'viewer/traindata'
m = json.load(open(root + '/trajectories.json'))
fr = [f for s in m['seqs'] for f in s['frames']][::37]
rows = []
for f in fr:
    d = np.asarray(Image.open(f"{root}/cond_{f['stem']}.png").convert('L'))
    a = np.asarray(Image.open(f"{root}/alb_{f['stem']}.png").convert('RGB'))
    rows.append((f['zoom'], d.max(), a.max()))
rows = np.array(rows)
for lo, hi in [(0.3,0.5),(0.5,0.7),(0.7,0.9),(0.9,1.1),(1.1,1.4)]:
    s = rows[(rows[:,0] >= lo) & (rows[:,0] < hi)]
    if len(s):
        print(f'zoom {lo}-{hi}: depth max {s[:,1].mean():5.0f}, '
              f'albedo max {s[:,2].mean():5.0f}, n={len(s)}')
EOF
```

Depth max should be well over 100 in **every** zoom band, including the
closest. If the close bands read single digits, the near plane has collapsed
and those frames are black. Albedo max should be over 100 everywhere too; near
zero means the flat colour pass rendered nothing.

If the capture came from an older version with no `mask_` files,
`renderer/bench_mask.py --all <traindata>` reconstructs them from the depth.
Current captures render the mask exactly and do not need it.

---

## 3. Train

```bash
cd ../renderer
python train_memory_cn.py --data ../viewer/traindata \
    --out memory_cn9 --steps 12000 --bs 8 --res 384 2>&1 | tee train.log
```

About an hour on an H200, roughly 16 epochs over 5200 frames. One whole
colourway (`british`) is held out, never single frames, so the question it
answers is whether it works on an appearance it has not seen.

**Check the first three lines:**

- `[model] conditioning 3 -> 9 channels (depth, albedo, memory)`
- `[data:train] ... frames` with **no** albedo warning
- `[rollout] step 1000: N frames of self-generated memory cached` shortly after

Ignore the loss. Diffusion training loss is dominated by which timestep was
drawn and sits near 0.27 whether the model learned everything or nothing. It
will look flat for the entire run. That is normal and says nothing.

Judge from `memory_cn9/clip_00N000.mp4` instead, written every 1000 steps: 36
frames of the held-out colourway, rendered the way it is served, with each
frame's memory being the model's own previous output. Same frames and seed
every time, so the clips are directly comparable. Flicker, trails and drift
are visible there and invisible in a still.

---

## 4. Measure

```bash
python eval_memory.py --cn memory_cn9/controlnet --data ../viewer/traindata
```

Runs each held-out frame twice with identical noise and fixed timesteps, once
with the memory channels filled and once zeroed. Same frame, same noise, one
variable.

- change near 0%, wins near 50%: the model is ignoring the memory
- negative change, wins above 50%: the memory is being used, and by how much

Side-by-side video against the truth:

```bash
python render_video.py --cn memory_cn9/controlnet --data ../viewer/traindata \
    --seq british_v0 --out heldout --res 384 --steps 8 --cfg 1.0
```

Add `--no-memory` for the A/B, `--left sch --panels 2 --upscale 768` for a
presentation cut, `--steps 24 --cfg 4.5` for maximum quality at about 1 fps.

---

## 5. Serve

```bash
python serve_render.py --memory-cn memory_cn9/controlnet --res 384 \
    --port 8903 --ws websockets-sansio --warmup \
    --fast --steps 4 --tiny-vae --compile
```

Confirm it prints `9 conditioning channels (depth, albedo, memory)` and
`LCM few-step mode`. If it says 6 channels it loaded an older checkpoint; if
peft is missing, `--fast` is silently ignored.

From the laptop, if the GPU is remote:

```bash
ssh -N -L 8903:<node>:8903 <user>@<cluster>
open "http://localhost:8078/index.html#api=http://localhost:8903"
```

Hard-reload the page. A cached copy of older JS sends the previous payload
format and the albedo is dropped.

**Check the per-frame log:** `[render] ws 115 ms memory +mem +alb`. The `+alb`
is the one that matters. `NO-ALBEDO` means a stale page, `ALBEDO-BLACK` means
the flat pass rendered nothing.

Useful knobs at serve time, no retraining: `--mem-scale 0` disables memory,
`--mem-reset` controls how stale memory may be before it is dropped,
`--ip-scale` weakens the reference photo, `--cfg 1.0` halves the cost per
frame, `--steps` trades quality for speed.

---

## Where the time goes

| stage | wall clock |
| --- | --- |
| capture, trajectories + pairs | ~45 min |
| training, 12000 steps at 384 | ~1 h on an H200 |
| eval | ~3 min |
| a 136-frame video | 2 to 5 min |

Serving runs at about 8 fps at 384. Transport plus browser readback set a
floor of 25 to 35 ms per frame, so this design tops out near 30 fps however
cheap the model gets.
