# What building this taught us

Written down because most of it was learned the expensive way, and because
several items generalise well beyond this object.

---

## Measurement

**Diffusion training loss tells you nothing about progress.** With a timestep
sampled uniformly per step, epsilon-MSE is dominated by which noise level was
drawn, not by model quality. Ours sat between 0.25 and 0.29 for an entire
12000-step run and would have looked identical had the model learned nothing.
Most of that loss is irreducible: at high `t` even a perfect model scores near
1.0, at low `t` near 0.

What works instead:

- **A paired A/B with the variance removed.** Run the same held-out frame
  twice, same noise, same fixed timesteps, changing exactly one input. That is
  `eval_memory.py`, and it answers "are the memory channels used at all", which
  the training loss cannot.
- **A clip, not a still.** Flicker, trails and drift are temporal, so a still
  frame cannot show them. Training writes a short held-out clip every 1000
  steps, rendered the way it is served, on the same frames with the same seed
  so checkpoints are comparable.

**Verify the inputs before spending the GPU.** Two full runs were wasted on
data that looked plausible and was not. A five-line script that measures the
data (depth contrast per zoom band, albedo brightness, object pixel area) would
have caught both in seconds. Every capture stage in `REPRODUCE.md` now ends
with such a check.

---

## The bug that cost the most

The blockout depth was rendered with three.js `MeshDepthMaterial`, which writes
the hardware depth buffer. That buffer is `1/w` distributed: nearly all of its
precision sits just in front of the near plane. The capture set
`near = max(1, dist - fitS*0.9)`, so on a close-up, where `dist` is smaller
than `fitS*0.9`, near clamped to **1mm** while far stayed several hundred mm
out. Every surface then landed at `z ≈ 0.998` and the depth map came out black.

**44% of the training frames had a depth map that never exceeded 60/255. In
one zoom band the maximum value was 3.7.** The model was trained on nearly half
garbage and learned to hallucinate. Worse, it made *memory look actively
harmful*: with no signal in the depth, the only informative input was the
memory picture, which is the model's own previous output, so it became a closed
feedback loop that reinforced whatever it invented.

The fix is a four-line shader writing **linear view-space depth** normalised
between near and far. Same contrast at every distance. The general lesson: a
depth buffer is a rendering artifact, not a measurement, and its nonlinearity
is invisible until you view something up close.

---

## Memory, and why the obvious version fails

The goal was appearance consistency across frames. Three designs, in the order
we tried them:

**Chaining (bad).** Start the sampler from the previous frame's latent at half
strength. Consistent for a few frames, then every frame inherits the previous
one's defects and the look drifts, so a periodic purge is forced, which snaps
the image back visibly. The user's description was exact: "it goes weird for
about 10 frames, then it all cancels." Consistency and drift were the same
mechanism, and the purge was the tax.

**Memory as conditioning (better).** Feed the previous frame in extra
ControlNet input channels instead of as the starting latent. Each frame is
still denoised from pure noise, so nothing compounds and no purge is needed.
The new channels are zero-initialised, so an untrained model behaves exactly
like stock ControlNet and training can only add.

**But it smeared.** The model painted the car where it *used* to be. The cause
is subtle and worth generalising: the ControlNet's first layer sees depth and
memory **at the same pixel**, and in training the memory was always the frame
immediately before, a few degrees away. At every pixel, "what the memory shows
here" and "what the answer is here" agreed. Gradient descent found the cheapest
rule that fits: **copy the memory at this pixel**. It never needed to consult
the depth for *where* the object is. At serving time, where memory can be many
degrees stale, that rule paints the old silhouette onto empty bench.

Two changes broke the copy rule:

- memory drawn from anywhere earlier in the step, spanning up to ~70° of
  camera motion rather than one frame
- 12% of the time, deliberately the **wrong** frame, with the target still
  correct, so copying becomes a losing strategy and the only thing that still
  pays is "position from the depth, appearance from the memory"

**Exposure bias.** Even then, training fed *captured* frames as memory: clean,
correct, perfectly consistent with the target. Serving feeds the model its own
flawed render. That is teacher forcing evaluated autoregressively, and it is a
standard failure. Corrupting the memory (blur, shift, noise, holes) imitates
noise, not the model's *specific* mistakes, which are the ones that compound.
The real fix is to unroll: every 200 steps the trainer replays five consecutive
frames, rendering each and passing its own output forward. That costs about
1.3× amortised, against 10× for a full window unroll.

---

## Do not make the model remember what the scene already knows

Colour drifted frame to frame because colour lived only in the weights and was
re-decided every frame. But the page **knows** every part's colour, it assigns
them from a palette. So it renders a flat per-part colour pass and feeds it as
three more conditioning channels. Colour stops being a decision and becomes an
input, and drift becomes impossible rather than merely discouraged.

The general form: anything the deterministic side of the system already knows
should be an input, not a thing the model reconstructs. It also buys sharpness,
because of the next point.

**MSE answers uncertainty with blur.** When the model is unsure about a detail,
the loss-minimising output is the *average* of the plausible details, and the
average of many sharp possibilities is soft. Every bit of uncertainty left in
the conditioning is paid for in blur. Removing uncertainty is more effective
than fighting the blur downstream.

---

## Render what you know; do not infer it

Object masks were originally recovered from the depth by fitting a plane: a
bench under a perspective camera is exactly affine in the hardware depth
buffer, so a robust quantile fit finds it and whatever stands off it is the
object. It worked, and it was still the wrong approach. It broke on close-ups
where no bench is visible, and it broke completely once depth became linear,
since a plane is not affine in linear view-space depth.

The scene can render the mask exactly: same camera, parts only, bench hidden.
One extra pass, zero heuristics. The fitting code survives only as a fallback
for older captures.

---

## Data framing matters more than expected

The VAE downsamples 8×. A car occupying 17% of a 512 frame leaves a wing 6px
wide, which is under one latent pixel: the model literally cannot see what it
is asked to draw. Two changes took the object from 17% to 30% of the frame:

- the framing padding was **absolute** (105mm added to the bbox diagonal), so
  on a step introducing one small part it dominated and pushed the camera far
  back
- the zoom range was simply too far out

Also: **capture at the resolution you train at.** The first dataset was
captured at 384 and upscaled to 512 for training, so the model learned from
blurred targets for free.

---

## Train and serve must share one code path

The depth the model trains on and the depth it is served must be produced by
the same function. In this codebase there is exactly one `grabPass`, used by
the capture harness and the live client alike, so a change to the depth
encoding cannot silently desynchronise them. The same rule applies to the
caption: it is rebuilt from the same manifest with the same logic on both
sides.

Where the two *did* diverge, the failures were quiet. A stale browser cache
sending the previous socket payload, for instance, drops the albedo silently
and colour drift returns with no error anywhere. That is why the server now
logs `+alb` per frame and warns loudly on `NO-ALBEDO` and `ALBEDO-BLACK`.

---

## Speed

Cost per frame is `steps × (2 if classifier-free guidance else 1)` model
passes. The demo went from 0.8 fps to 8 fps by attacking both factors:
guidance is nearly pointless once the ControlNet is fine-tuned on one object,
since the conditioning already pins the output, and few-step sampling handles
the rest.

Two caveats worth knowing:

- **The generic LCM-LoRA fights a fine-tuned model.** It was distilled from
  stock SD 1.5. At 2 to 4 steps the sampler commits to low-frequency content,
  which is colour, almost immediately. Distilling a few-step version of *your*
  model is the principled fix.
- **There is an architectural floor.** Browser readback plus the round trip
  costs 25 to 35 ms per frame, so this design tops out near 30 fps no matter
  how cheap the model becomes. Distillation cannot break that floor; a
  different transport would be needed.

---

## Still open

- **Colour stability is not fully solved.** Candidates: the LCM sampler, the
  IP-Adapter reference at 0.7 competing with the albedo, and memory feedback.
  Each is one server restart to isolate, and it has not been done.
- **Diffusion is fighting us on two axes at once,** consistency and speed. A
  deterministic recurrent renderer, distilled from this model as teacher, gives
  frame-to-frame consistency by construction rather than by patch, and runs at
  a single forward pass per frame. `train_state_student.py` in the research
  tree is a start.
- **Generalisation is untested.** This is overfit to one kit by design. Whether
  it transfers to a second object, and what the scaling curve looks like, is
  the actual research question and nothing here answers it.
