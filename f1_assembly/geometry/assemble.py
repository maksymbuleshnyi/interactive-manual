#!/usr/bin/env python3
"""F1 car assembly workspace: applies per-part transforms from
assembly.json to the print-bed STLs and renders orthographic 3-views of the
result, for iterative placement against the manual photos.

Car frame: X = forward (nose +X), Y = up, Z = driver's left.
assembly.json: {part: {"rot": [rx,ry,rz] degrees applied as Rz@Ry@Rx,
                       "pos": [x,y,z] mm of the part's BBOX CENTER,
                       "src": "file.stl" (optional, defaults to part name),
                       "tyre": 0-3 (optional: quadrant of Tyres.stl)}}

  python3 assemble.py            -> assembled_views.jpg
  python3 assemble.py --parts "Mid Body,Front Body"   (subset)
"""
import argparse
import json
import os
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'real_grounding'))
from render_geometry_maps import render_maps            # noqa: E402
from PIL import Image, ImageDraw                        # noqa: E402

STL_DIR = os.path.join(ROOT, 'Separate STLs')


def read_stl(path):
    with open(path, 'rb') as f:
        f.seek(80)
        n = struct.unpack('<I', f.read(4))[0]
        data = np.frombuffer(f.read(n * 50), dtype=np.uint8)
        tri = data.reshape(n, 50)[:, 12:48].copy().view('<f4').reshape(n, 3, 3)
        return tri.astype(np.float64)


def cluster_decimate(tri, cells=110):
    v = tri.reshape(-1, 3)
    mn, mx = v.min(0), v.max(0)
    h = max((mx - mn).max() / cells, 1e-6)
    key = np.floor((v - mn) / h).astype(np.int64)
    kid = key[:, 0] * 73856093 ^ key[:, 1] * 19349663 ^ key[:, 2] * 83492791
    uniq, inv = np.unique(kid, return_inverse=True)
    cent = np.zeros((len(uniq), 3))
    cnt = np.zeros(len(uniq))
    np.add.at(cent, inv, v)
    np.add.at(cnt, inv, 1)
    cent /= cnt[:, None]
    f = inv.reshape(-1, 3)
    good = (f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])
    return cent, f[good]


AXES = {'x': np.array([1., 0, 0]), 'y': np.array([0., 1, 0]),
        'z': np.array([0., 0, 1]),
        '-x': np.array([-1., 0, 0]), '-y': np.array([0., -1, 0]),
        '-z': np.array([0., 0, -1])}


def map_matrix(mapping):
    """mapping = images of print x,y,z as car-frame axes,
    e.g. ["-z","y","x"] means print X -> car -Z, print Y -> car Y,
    print Z -> car X. Columns of R are those images: v_car = R @ v_print."""
    R = np.stack([AXES[m] for m in mapping], axis=1)
    assert abs(np.linalg.det(R) - 1) < 1e-6, \
        f'improper/reflection mapping {mapping}'
    return R


def rot_matrix(rx, ry, rz):
    cx, sx = np.cos(np.deg2rad(rx)), np.sin(np.deg2rad(rx))
    cy, sy = np.cos(np.deg2rad(ry)), np.sin(np.deg2rad(ry))
    cz, sz = np.cos(np.deg2rad(rz)), np.sin(np.deg2rad(rz))
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


_tyre_cache = None


def load_part(name, spec):
    global _tyre_cache
    src = spec.get('src', name + '.stl')
    if 'tyre' in spec:
        if _tyre_cache is None:
            tri = read_stl(os.path.join(STL_DIR, 'Tyres.stl'))
            c = tri.mean(1)                       # per-tri centroid
            mid = (tri.reshape(-1, 3).min(0) + tri.reshape(-1, 3).max(0)) / 2
            _tyre_cache = {}
            for q, (sx_, sy_) in enumerate([(-1, -1), (1, -1), (-1, 1),
                                            (1, 1)]):
                m = ((c[:, 0] - mid[0]) * sx_ > 0) & \
                    ((c[:, 1] - mid[1]) * sy_ > 0)
                _tyre_cache[q] = tri[m]
        tri = _tyre_cache[spec['tyre']]
    else:
        tri = read_stl(os.path.join(STL_DIR, src))
    v, f = cluster_decimate(tri)
    c = (v.min(0) + v.max(0)) / 2
    return v - c, f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--parts', default=None,
                    help='comma-separated subset to render')
    ap.add_argument('--out', default=os.path.join(HERE,
                                                  'assembled_views.jpg'))
    ap.add_argument('--res', type=int, default=540)
    a = ap.parse_args()

    asm = json.load(open(os.path.join(HERE, 'assembly.json')))
    names = [n.strip() for n in a.parts.split(',')] if a.parts else list(asm)

    meshes = {}
    for i, n in enumerate(names):
        spec = asm[n]
        v, f = load_part(n, spec)
        if 'map' in spec:
            R = map_matrix(spec['map'])
        else:
            R = rot_matrix(*spec.get('rot', [0, 0, 0]))
        if 'rot2' in spec:                 # fine-tune after axis mapping
            R = rot_matrix(*spec['rot2']) @ R
        v = v @ R.T + np.array(spec.get('pos', [0, 0, 0]), dtype=float)
        meshes[i] = (v, f)

    # axis gizmo: +X red-ish bar (part id 100), +Y green (101), +Z blue (102)
    def bar(d, L=60, w=1.5):
        d = np.array(d, float)
        ax = np.argmax(np.abs(d))
        lo, hi = np.minimum(d * 0, d * L), np.maximum(d * 0, d * L)
        lo[lo == 0] -= w
        hi[hi == 0] += w
        corners = np.array([[x, y, z] for x in (lo[0], hi[0])
                            for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        faces = []
        idx = {tuple(c): k for k, c in enumerate(corners)}
        quads = [(0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4),
                 (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5)]
        for q in quads:
            faces += [[q[0], q[1], q[2]], [q[0], q[2], q[3]]]
        return corners, np.array(faces)
    meshes[100] = bar([1, 0, 0])
    meshes[101] = bar([0, 1, 0])
    meshes[102] = bar([0, 0, 1])

    all_v = np.concatenate([v for v, _ in meshes.values()])
    c = (all_v.min(0) + all_v.max(0)) / 2
    span = float(np.linalg.norm(all_v.max(0) - all_v.min(0)))
    RES = a.res
    views = [(90, 4, 'front-on (camera on +X: nose toward you)'),
             (0, 4, 'side (camera +Z: nose RIGHT)'),
             (180, 4, 'side flip (camera -Z: nose LEFT)'),
             (0, 88, 'top')]
    tiles = []
    for az, el, lab in views:
        a_, e_ = np.deg2rad(az), np.deg2rad(el)
        eye = c + 1.5 * span * np.array([np.cos(e_) * np.sin(a_),
                                         np.sin(e_),
                                         np.cos(e_) * np.cos(a_)])
        fwd = c - eye
        fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, [0, 1, 0])
        right /= np.linalg.norm(right)
        up = np.cross(right, fwd)
        Rm = np.stack([right, -up, fwd])
        E = np.zeros((3, 4))
        E[:, :3], E[:, 3] = Rm, -Rm @ eye
        K = [[700, 0, RES / 2], [0, 700, RES / 2], [0, 0, 1]]
        fr = {'parts': [str(p) for p in meshes],
              'intrinsics': [K] * len(meshes),
              'extrinsics': [E.tolist()] * len(meshes)}
        _, normal, pid = render_maps(fr, meshes, RES, RES)
        t = Image.new('RGB', (RES * 2, RES + 24), (16, 16, 20))
        t.paste(normal, (0, 24))
        t.paste(pid, (RES, 24))
        ImageDraw.Draw(t).text((6, 5), lab, fill=(235, 235, 235))
        tiles.append(t)
    sheet = Image.new('RGB', (RES * 2, (RES + 24) * 4 + 20), (16, 16, 20))
    for k, t in enumerate(tiles):
        sheet.paste(t, (0, k * (RES + 24)))
    from render_pose_pairs import PALETTE
    d = ImageDraw.Draw(sheet)
    leg = '   '.join(f'{n}={i}' for i, n in enumerate(names))
    d.text((6, (RES + 24) * 4 + 4),
           'pid colors by index: ' + leg + '   gizmo: 100=+X 101=+Y 102=+Z',
           fill=(220, 220, 220))
    sheet.save(a.out, quality=92)
    print('saved', a.out, '| parts:', len(names))
    for i, n in enumerate(names):
        print(f'  pid {i}: {n} -> palette', PALETTE[i % len(PALETTE)]
              if isinstance(PALETTE, (list, tuple)) else '?')


if __name__ == '__main__':
    main()
