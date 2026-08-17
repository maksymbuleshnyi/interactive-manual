#!/usr/bin/env python3
"""One training example for the blocks-to-photo model, drawn from the real F1
assets: the agent's blockout on the left, the frame it must produce on the
right, plus the text and reference photo that carry identity and appearance.

  python3 make_training_example.py
"""
import glob
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'real_grounding'))
from assemble import load_part                          # noqa: E402
from render_geometry_maps import render_maps            # noqa: E402

RES = 420
STEP_TITLE = 'Rear tyres'             # the step this example belongs to


# ---------- the agent's proxy: boxes and cylinders, no CAD ----------
def box(w, h, d, x=0, y=0, z=0):
    c = np.array([[i, j, k] for i in (-w/2, w/2) for j in (-h/2, h/2)
                  for k in (-d/2, d/2)]) + [x, y, z]
    f = [[0,1,3],[0,3,2],[4,6,7],[4,7,5],[0,4,5],[0,5,1],
         [2,3,7],[2,7,6],[0,2,6],[0,6,4],[1,5,7],[1,7,3]]
    return c, np.array(f)


def cyl(r, l, axis, x=0, y=0, z=0, seg=16):
    a = np.linspace(0, 2*np.pi, seg, endpoint=False)
    v = np.vstack([np.stack([r*np.cos(a), r*np.sin(a), np.full(seg, l/2)], 1),
                   np.stack([r*np.cos(a), r*np.sin(a), np.full(seg, -l/2)], 1),
                   [[0, 0, l/2]], [[0, 0, -l/2]]])
    f = []
    for i in range(seg):
        j = (i+1) % seg
        f += [[i,j,seg+j],[i,seg+j,seg+i],[2*seg,j,i],[2*seg+1,seg+i,seg+j]]
    if axis == 'x':
        v = v[:, [2,1,0]]
    if axis == 'y':
        v = v[:, [0,2,1]]
    return v + [x, y, z], np.array(f)


def merge(parts):
    V, F, o = [], [], 0
    for v, f in parts:
        V.append(v); F.append(f + o); o += len(v)
    return np.vstack(V), np.vstack(F)


def proxy(name, dims):
    dx, dy, dz = dims
    if 'Tyre' in name:
        r = max(dx, dy)/2
        return merge([cyl(r, dz, 'z'), cyl(r*0.58, dz*1.04, 'z')])
    if 'Inner Hub' in name:
        return merge([cyl(max(dx, dy)/2, dz, 'z')])
    if 'Wheel Hub' in name:
        r = max(dx, dy)/2
        return merge([cyl(r, dz*0.55, 'z', 0, 0, dz*0.2), cyl(r*0.35, dz, 'z')])
    if 'Axle' in name:
        return merge([cyl(max(dx, dy)/2, dz, 'z')])
    if 'Pin' in name:
        return merge([cyl(max(dy, dz)/2, dx, 'x')])
    if name == 'Mid Body':
        return merge([box(dx, dy*0.16, dz*0.62, 0, -dy*0.40, 0),
                      box(dx*0.86, dy*0.46, dz*0.34, -dx*0.04, -dy*0.08, 0),
                      box(dx*0.30, dy*0.26, dz*0.26, dx*0.22, dy*0.16, 0),
                      box(dx*0.26, dy*0.34, dz*0.20, -dx*0.06, dy*0.26, 0),
                      box(dx*0.34, dy*0.26, dz*0.16, -dx*0.30, dy*0.14, 0),
                      box(dx*0.44, dy*0.34, dz*0.22, -dx*0.10, -dy*0.14, dz*0.36),
                      box(dx*0.44, dy*0.34, dz*0.22, -dx*0.10, -dy*0.14, -dz*0.36)])
    return merge([box(dx, dy, dz)])


def look(az, el, radius, target):
    a, e = np.deg2rad(az), np.deg2rad(el)
    eye = target + radius*np.array([np.cos(e)*np.sin(a), np.sin(e),
                                    np.cos(e)*np.cos(a)])
    fwd = target - eye
    fwd /= np.linalg.norm(fwd)
    r = np.cross(fwd, [0, 1, 0]); r /= np.linalg.norm(r)
    u = np.cross(r, fwd)
    R = np.stack([r, -u, fwd])
    E = np.zeros((3, 4)); E[:, :3] = R; E[:, 3] = -R @ eye
    return E, R


def main():
    fin = json.load(open(os.path.join(HERE, 'assembly_final.json')))
    man = json.load(open(os.path.join(HERE, 'viewer_manifest.json')))
    P = man['parts']
    step = next(s for s in man['steps'] if s['title'] == STEP_TITLE)
    visible = []
    for s in man['steps']:
        visible += [n for n in s['new'] if not s.get('layout')]
        if s['title'] == STEP_TITLE:
            break

    prox, detail, groups = {}, {}, []
    for i, n in enumerate(visible):
        spec = fin[n]
        M, t = np.array(spec['matrix']), np.array(spec['pos'])
        v, f = proxy(n, P[n]['dims'])
        prox[i] = (v + t, f)                       # proxies are axis-aligned
        v2, f2 = load_part(n, spec)
        detail[i] = (v2 @ M.T + t, f2)
        groups.append((n, P[n]['group']))

    allv = np.concatenate([v for v, _ in detail.values()])
    c = (allv.min(0) + allv.max(0))/2
    span = float(np.linalg.norm(allv.max(0) - allv.min(0)))
    E, R = look(46, 22, 1.5*span, c)
    K = [[560, 0, RES/2], [0, 560, RES/2], [0, 0, 1]]

    # unique colour per part: the stock 10-colour palette wraps around and
    # would make two different parts indistinguishable in the id map
    import render_pose_pairs as rpp
    import colorsys
    n_parts = len(visible)
    rpp.PALETTE = [tuple(int(255*v) for v in
                         colorsys.hsv_to_rgb(i/max(1, n_parts), 0.85, 0.95))
                   for i in range(n_parts)]
    PALETTE = rpp.PALETTE

    def render(meshes):
        fr = {'parts': [str(k) for k in meshes],
              'intrinsics': [K]*len(meshes),
              'extrinsics': [E.tolist()]*len(meshes)}
        return render_maps(fr, meshes, RES, RES)

    p_depth, p_norm, p_pid = render(prox)
    d_depth, d_norm, d_pid = render(detail)

    # material map: colour by what the part is made of, not which part it is
    MATCOL = {'body': (70, 120, 210), 'accent': (70, 120, 210),
              'hub': (200, 170, 60), 'tyre': (200, 70, 90)}
    pid_a = np.asarray(p_pid).astype(np.int16)
    mat = np.zeros((RES, RES, 3), np.uint8)
    inst = np.zeros((RES, RES, 3), np.uint8)
    for i, (n, g) in enumerate(groups):
        col = np.array(PALETTE[i % len(PALETTE)], np.int16)
        m = (np.abs(pid_a - col).sum(2) < 12)
        mat[m] = MATCOL.get(g, (120, 120, 120))
        inst[m] = col.astype(np.uint8)
    mask = (np.asarray(p_depth).astype(np.float32) > 6)

    # the target the model must produce (photographic frame; here the graphics
    # render stands in for it)
    nrm = np.asarray(d_norm).astype(np.float32)/255*2 - 1
    dd = np.asarray(d_depth).astype(np.float32)
    L = np.array([0.45, 0.62, 0.65]); L /= np.linalg.norm(L)
    L2 = np.array([-0.5, 0.4, -0.4]); L2 /= np.linalg.norm(L2)
    lit = np.clip(nrm @ (R @ L), 0, 1) + 0.35*np.clip(nrm @ (R @ L2), 0, 1)
    shade = np.clip(0.38 + 0.95*lit, 0, 1.6)
    # albedo per material, so tyres read as rubber and hubs as metal
    ALB = {'body': (0.28, 0.30, 0.33), 'accent': (0.85, 0.36, 0.10),
           'hub': (0.42, 0.45, 0.48), 'tyre': (0.075, 0.075, 0.085)}
    d_pid_a = np.asarray(d_pid).astype(np.int16)
    alb = np.zeros((RES, RES, 3), np.float32)
    alb[:] = ALB['body']
    for i, (n, g) in enumerate(groups):
        col = np.array(PALETTE[i % len(PALETTE)], np.int16)
        m2 = (np.abs(d_pid_a - col).sum(2) < 12)
        alb[m2] = ALB.get(g, ALB['body'])
    tgt = np.clip(alb*shade[..., None]*255*1.35, 0, 255)
    bench = np.linspace(150, 92, RES)[:, None]
    bg = np.repeat(bench, RES, 1)[..., None]*np.array([1.0, 0.72, 0.48])
    tgt = np.where((dd > 6)[..., None], tgt, bg)

    panels = [
        (p_depth.convert('RGB'), 'INPUT 1  depth of the agent blocks'),
        (Image.fromarray(inst), 'INPUT 2  which part is which'),
        (Image.fromarray(mat), 'INPUT 3  what it is made of'),
        (Image.fromarray((mask*255).astype(np.uint8)).convert('RGB'),
         'INPUT 4  object vs bench'),
    ]
    ref_files = sorted(glob.glob(os.path.join(HERE, 'grounding', '*_loose_v0.png')))
    ref = Image.open(ref_files[0]).convert('RGB').resize((RES, RES)) \
        if ref_files else Image.new('RGB', (RES, RES), (30, 30, 34))
    panels.append((ref, 'INPUT 5  reference photo (their bench)'))
    panels.append((Image.fromarray(tgt.astype(np.uint8)),
                   'TARGET  the photograph (graphics render stands in here)'))

    try:
        F1 = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 15)
        FB = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 20)
        FS = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 13)
    except Exception:
        F1 = FB = FS = ImageFont.load_default()

    W = RES*3
    sheet = Image.new('RGB', (W, 64 + (RES+30)*2 + 74), (14, 14, 18))
    d = ImageDraw.Draw(sheet)
    d.text((14, 12), 'ONE TRAINING EXAMPLE  ·  step "%s"  ·  one camera'
           % STEP_TITLE, font=FB, fill=(255, 255, 255))
    d.text((14, 38), 'left: everything the agent can produce from its own '
           'blocks. right: what the camera should have seen',
           font=FS, fill=(150, 158, 172))
    for k, (im, lab) in enumerate(panels):
        x, y = (k % 3)*RES, 64 + (k//3)*(RES+30)
        sheet.paste(im.resize((RES, RES)), (x, y))
        d.rectangle([x, y+RES, x+RES, y+RES+30], fill=(24, 26, 32))
        d.text((x+8, y+RES+8), lab, font=F1,
               fill=(255, 205, 90) if 'TARGET' in lab else (215, 220, 230))
    y0 = 64 + (RES+30)*2 + 6
    d.text((14, y0), 'INPUT 6  caption:  "a printed Formula One monocoque '
           'with rear suspension and both rear wheels fitted, black rubber '
           'tyres on metal hubs, on a wooden bench, photo"',
           font=F1, fill=(215, 220, 230))
    d.text((14, y0+24), 'INPUT 7  previous frame (for stability while the '
           'camera moves)      LOSS  compare the produced frame with the '
           'target, weighted onto the object', font=F1, fill=(215, 220, 230))
    d.text((14, y0+48), 'the model must add everything the blocks lack: the '
           'curved flanks, the tread and sidewall of each tyre, the spokes '
           'in the hub face, print texture, contact shadow', font=FS, fill=(150, 158, 172))
    out = os.path.join(HERE, 'training_example.jpg')
    sheet.save(out, quality=93)
    print('saved', out)


if __name__ == '__main__':
    main()
