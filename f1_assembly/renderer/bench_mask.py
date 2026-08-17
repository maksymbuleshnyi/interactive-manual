#!/usr/bin/env python3
"""Object mask from a blockout depth map that already contains the bench.

The capture renders the parts sitting on a bench plane, so the depth map is
not object-on-black and no threshold separates them. But a plane under a
perspective camera has a depth buffer value that is exactly affine in screen
coordinates, so fitting an affine ramp robustly recovers the bench, and
whatever stands off that ramp toward the camera is the object.

Run directly to write a check image:
  python bench_mask.py <traindata dir> [stem ...]
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter


def object_mask(depth: np.ndarray, tol: float = 6.0,
                tau: float = 0.15) -> np.ndarray:
    """depth: HxW float array, brighter = nearer (three.js MeshDepthMaterial).
    Returns a float mask, 1 on the parts.

    Fitted as a LOWER ENVELOPE rather than a least-squares average: every
    part stands in front of the bench, never behind it, so an average fit is
    dragged up by the object and a close-up that fills the frame loses its
    bench entirely. Quantile regression at tau keeps the ramp underneath the
    data, which stays correct even when only a corner of bench is visible,
    because the plane's depth is exactly affine across the whole image."""
    h, w = depth.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    valid = (depth > 2).ravel()
    d = depth.ravel()[valid]
    if d.size < 200:
        return np.zeros_like(depth)
    A = np.stack([xx.ravel()[valid], yy.ravel()[valid],
                  np.ones(d.size, np.float32)], 1)
    sub = slice(None, None, 3)                 # the fit does not need every px
    As, ds = A[sub], d[sub]
    coef = np.linalg.lstsq(As, ds, rcond=None)[0]
    for _ in range(20):
        res = ds - As @ coef
        wq = np.where(res > 0, tau, 1 - tau) / np.maximum(np.abs(res), 1.0)
        sw = np.sqrt(wq)[:, None]
        coef = np.linalg.lstsq(As * sw, ds * sw[:, 0], rcond=None)[0]
    res = d - A @ coef
    full = np.zeros(h * w, bool)
    full[valid] = res > tol
    # a tight close-up can leave no bench in frame at all, and then the fitted
    # ramp is a plane through the car itself. If almost nothing sits ON the
    # ramp, there was no bench to find, so everything solid is object.
    if (np.abs(res) < tol).sum() < 0.10 * h * w:
        full[valid] = True
    im = Image.fromarray((full.reshape(h, w) * 255).astype(np.uint8))
    im = im.filter(ImageFilter.MinFilter(3)).filter(ImageFilter.MaxFilter(5))
    return np.asarray(im).astype(np.float32) / 255.0


def cache_all(root: str) -> int:
    """Write mask_<stem>.png for every conditioning frame in the folder, so
    training loads a PNG instead of refitting a plane per sample."""
    import os
    names = sorted(f[5:-4] for f in os.listdir(root)
                   if f.startswith('cond_') and f.endswith('.png'))
    made = 0
    for i, s in enumerate(names):
        src = os.path.join(root, 'cond_' + s + '.png')
        out = os.path.join(root, 'mask_' + s + '.png')
        # a recapture reuses the same stems, so an older mask is a WRONG mask,
        # not a cached one
        if os.path.exists(out) and os.path.getmtime(out) >= os.path.getmtime(src):
            continue
        dp = np.asarray(Image.open(os.path.join(root, 'cond_' + s + '.png'))
                        .convert('L')).astype(np.float32)
        Image.fromarray((object_mask(dp) * 255).astype(np.uint8)).save(out)
        made += 1
        if made % 200 == 0:
            print(f'  {made} masks ...', flush=True)
    print(f'{len(names)} frames, {made} masks written to {root}')
    return made


def _main():
    import json
    import os
    import sys
    if sys.argv[1] == '--all':
        cache_all(sys.argv[2])
        return
    root = sys.argv[1]
    stems = sys.argv[2:]
    if not stems:
        meta = json.load(open(os.path.join(root, 'trajectories.json')))
        stems = [meta['seqs'][i]['frames'][j]['stem']
                 for i in (0, 5, 11) for j in (0, 30, 60)]
    cols = []
    for s in stems:
        dp = np.asarray(Image.open(os.path.join(root, 'cond_' + s + '.png'))
                        .convert('L')).astype(np.float32)
        m = object_mask(dp)
        tg = np.asarray(Image.open(os.path.join(root, 'tgt_' + s + '.png'))
                        .convert('RGB')).astype(np.float32)
        over = tg * (0.35 + 0.65 * m[..., None])
        cols.append(np.concatenate(
            [np.repeat(dp[..., None], 3, 2),
             np.repeat(m[..., None] * 255, 3, 2), over], 0))
        print(f'{s}: object covers {m.mean()*100:5.1f}% of the frame')
    out = np.concatenate(cols, 1).astype(np.uint8)
    p = os.path.join(root, 'mask_check.png')
    Image.fromarray(out).save(p)
    print('wrote', p, '(rows: depth, mask, target dimmed outside the mask)')


if __name__ == '__main__':
    _main()
