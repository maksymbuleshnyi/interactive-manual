#!/usr/bin/env python3
"""Contact-patch extraction between assembled parts.

Every part sits in car coordinates (assembly_final.json, registered against
the manufacturer's assembled STL), so where two parts actually touch is
measurable rather than guessed: sample part A's surface, keep the points
within TOL of part B's surface, cluster them, and describe each patch.

Each patch yields what a manual needs to state precisely:
  centroid  -> where to put the glue
  normal    -> which face it goes on
  shape     -> bead along a line / ring / thin coat / single drop
  points    -> where to draw the glue marks in the viewer

  python3 contacts.py            -> glue_sites.json
"""
import json
import os
import sys
from collections import defaultdict, deque

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from register_parts import read_stl                     # noqa: E402
from assemble import load_part                          # noqa: E402

TOL = 1.0          # mm: surfaces closer than this are "touching"
CELL = 2.0         # mm: clustering cell for separating patches
MIN_PTS = 12       # discard specks

# joints the manual actually calls out (a = moving part, b = it mates to)
JOINTS = [
    ("Locating Pin 1", "Mid Body", "press"),
    ("Locating Pin 2", "Mid Body", "press"),
    ("Locating Pin 3", "Mid Body", "press"),
    ("Back Body", "Mid Body", "glue"),
    ("Front Body", "Mid Body", "glue"),
    ("Left Rear Suspension", "Back Body", "glue"),
    ("Right Rear Suspension", "Back Body", "glue"),
    ("Front Mini-Wing", "Front Body", "glue"),
    ("Halo", "Front Body", "glue"),
    ("Rear Axle", "Rear Wheel Hub 1", "glue"),
    ("Rear Axle", "Rear Wheel Hub 2", "glue"),
    ("Rear Tyre L", "Rear Wheel Hub 1", "press"),
    ("Rear Tyre R", "Rear Wheel Hub 2", "press"),
    ("Front Wheel Inner Hub A 1", "Front Wheel Inner Hub B 1", "glue"),
    ("Front Wheel Inner Hub B 1", "Front Wheel Inner Hub A 1", "glue"),
    ("Front Wheel Inner Hub A 2", "Front Wheel Inner Hub B 2", "glue"),
    ("Front Wheel Inner Hub B 2", "Front Wheel Inner Hub A 2", "glue"),
    ("Front Wheel Hub 1", "Front Wheel Inner Hub A 1", "glue"),
    ("Front Wheel Hub 2", "Front Wheel Inner Hub A 2", "glue"),
    ("Front Tyre L", "Front Wheel Hub 1", "press"),
    ("Front Tyre R", "Front Wheel Hub 2", "press"),
    ("Side Camera Mount", "Mid Body", "glue"),
    ("Side Camera Mount 2", "Mid Body", "glue"),
    ("Top Camera Mount", "Mid Body", "glue"),
    ("Halo", "Mid Body", "glue"),
    ("Rear Stabiliser Fin", "Mid Body", "glue"),
]


STL_DIR = os.path.join(os.path.dirname(HERE), 'Separate STLs')


def surface_points(name, spec, n_max=150000):
    """Dense surface sample of a placed part, from the RAW mesh (decimation
    would erase exactly the small mating features we are looking for)."""
    if 'tyre' in spec:
        tri = read_stl(os.path.join(HERE, f"tyre_q{spec['tyre']}.stl"))
    else:
        tri = read_stl(os.path.join(STL_DIR, spec.get('src', name + '.stl')))
    v = tri.reshape(-1, 3)
    c = (v.min(0) + v.max(0)) / 2
    P = np.vstack([v, tri.mean(1)]) - c
    if len(P) > n_max:
        P = P[np.random.RandomState(0).choice(len(P), n_max, replace=False)]
    return P @ np.array(spec['matrix']).T + np.array(spec['pos'])


def near_mask(A, B, tol):
    """Boolean mask over A: within ~tol of B's surface (voxel occupancy)."""
    lo = np.minimum(A.min(0), B.min(0)) - 3 * tol
    hi = np.maximum(A.max(0), B.max(0)) + 3 * tol
    dims = np.floor((hi - lo) / tol).astype(int) + 3
    occ = np.zeros(dims, bool)
    ib = np.floor((B - lo) / tol).astype(int)
    occ[ib[:, 0], ib[:, 1], ib[:, 2]] = True
    for ax in range(3):
        occ |= np.roll(occ, 1, axis=ax) | np.roll(occ, -1, axis=ax)
    ia = np.floor((A - lo) / tol).astype(int)
    ok = np.all((ia >= 0) & (ia < dims), axis=1)
    out = np.zeros(len(A), bool)
    ii = ia[ok]
    out[ok] = occ[ii[:, 0], ii[:, 1], ii[:, 2]]
    return out


def cluster(P, cell=CELL):
    """Grid-connected clustering."""
    keys = np.floor(P / cell).astype(np.int64)
    grid = defaultdict(list)
    for i, k in enumerate(map(tuple, keys)):
        grid[k].append(i)
    seen, groups = set(), []
    for k in list(grid):
        if k in seen:
            continue
        q, comp = deque([k]), []
        seen.add(k)
        while q:
            c = q.popleft()
            comp += grid[c]
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        nk = (c[0] + dx, c[1] + dy, c[2] + dz)
                        if nk in grid and nk not in seen:
                            seen.add(nk)
                            q.append(nk)
        if len(comp) >= MIN_PTS:
            groups.append(np.array(comp))
    return groups


def farthest_points(P, k):
    """Even spread of k marker points across the patch."""
    idx = [int(np.argmin(np.linalg.norm(P - P.mean(0), axis=1)))]
    d = np.linalg.norm(P - P[idx[0]], axis=1)
    while len(idx) < min(k, len(P)):
        i = int(np.argmax(d))
        idx.append(i)
        d = np.minimum(d, np.linalg.norm(P - P[i], axis=1))
    return P[idx]


def marks_for(P, k=14):
    """Where to draw the glue. A ring-shaped contact, which is what every
    press fit and bore joint gives, must be marked at even angles: farthest
    point sampling clumps on a ring and reads as sloppy."""
    c = P.mean(0)
    u, s, vt = np.linalg.svd(P - c, full_matrices=False)
    axis = vt[2]                        # ring plane normal, or shaft axis
    e1, e2 = vt[0], vt[1]
    rel = P - c
    # test both interpretations: points around a circle in the (e1,e2) plane
    x, y = rel @ e1, rel @ e2
    r = np.hypot(x, y)
    if len(P) > 12 and r.mean() > 1.0 and r.std() / max(r.mean(), 1e-6) < 0.45:
        rad = float(np.median(r))
        ang = np.linspace(0, 2 * np.pi, k, endpoint=False)
        out = [c + e1 * (rad * np.cos(t)) + e2 * (rad * np.sin(t))
               for t in ang]
        # keep the ring on the surface: snap each mark to its nearest point
        snapped = []
        for q in out:
            j = int(np.argmin(np.linalg.norm(P - q, axis=1)))
            snapped.append(P[j])
        return np.array(snapped)
    return farthest_points(P, k)


def describe(P, ext, area):
    """How the glue should be applied, from the patch's shape."""
    ratio = ext[0] / max(ext[1], 1e-3)
    if area < 12:
        return "single small drop"
    if ratio > 3.0:
        return "thin bead along the line of contact"
    if ext[2] < 0.6 and area > 60:
        return "thin even coat over the whole face"
    return "thin ring of glue around the contact"


# how each part actually goes on, where geometry alone is ambiguous
FROM_ABOVE = {'Halo', 'Rear Stabiliser Fin', 'Top Camera Mount'}
THREADED = {'Front Mini-Wing'}                    # through a slot, side to side
OUTBOARD = {'Left Rear Suspension', 'Right Rear Suspension',
            'Side Camera Mount', 'Side Camera Mount 2'}   # in from the side
CLAMSHELL = {'Front Wheel Inner Hub A 1': +1, 'Front Wheel Inner Hub B 1': -1,
             'Front Wheel Inner Hub A 2': +1, 'Front Wheel Inner Hub B 2': -1}


def insertion_axis(a, b, patch, pos_of):
    pa = np.array(pos_of[a], float)
    pb = np.array(pos_of.get(b, pos_of[a]), float)
    if a in FROM_ABOVE:
        return np.array([0.0, 1.0, 0.0])
    if a in THREADED:
        return np.array([0.0, 0.0, 1.0])
    if a in CLAMSHELL:                    # each half closes in from its side
        off = pa - pb
        k = int(np.argmax(np.abs(off)))
        v = np.zeros(3)
        v[k] = np.sign(off[k]) or 1.0
        return v
    if a in OUTBOARD or 'Tyre' in a or 'Wheel Hub' in a or 'Inner Hub' in a \
            or a == 'Rear Axle':
        z = pa[2] if abs(pa[2]) > 3 else pb[2]
        return np.array([0.0, 0.0, 1.0 if z >= 0 else -1.0])
    off = pa - pb
    src = off if np.linalg.norm(off) > 15 else np.array(patch['normal'], float)
    k = int(np.argmax(np.abs(src)))
    v = np.zeros(3)
    v[k] = np.sign(src[k]) or 1.0
    return v


# joints the geometry cannot express: a part threaded through a slot has no
# outside contact patch, so the annotation is authored once and kept here
MANUAL = {
    'Front Mini-Wing|Front Body': dict(
        kind='glue', how='one small drop in the slot, then slide it through',
        offset=[0, 0, 0], insert=[0, 0, 1], area=30,
        marks_along='z', n_marks=5, span=12),
    'Side Camera Mount|Mid Body': dict(
        kind='glue', how='one drop in the locating hole',
        offset=[0, -2, -2], insert=[0, 0, 1], area=20,
        marks_along='x', n_marks=3, span=2.5),
    'Side Camera Mount 2|Mid Body': dict(
        kind='glue', how='one drop in the locating hole',
        offset=[0, -2, 2], insert=[0, 0, -1], area=20,
        marks_along='x', n_marks=3, span=2.5),
    'Top Camera Mount|Mid Body': dict(
        kind='glue', how='one drop in the locating hole',
        offset=[0, -3.5, 0], insert=[0, 1, 0], area=22,
        marks_along='z', n_marks=3, span=2.5),
    'Front Body|Mid Body': dict(
        kind='glue', how='glue inside the square socket, not on the outer seam',
        offset=[-30, 2, 0], insert=[1, 0, 0], area=90,
        marks_along='y', n_marks=6, span=7),
}


def add_manual(out, fin):
    for key, m in MANUAL.items():
        a, b = key.split('|')
        if key in out and out[key]['patches'] and \
                out[key]['patches'][0]['area'] > 20:
            continue                     # geometry found a real patch, keep it
        pos = np.array(fin[a]['pos'], float) + np.array(m['offset'], float)
        ax = {'x': 0, 'y': 1, 'z': 2}[m['marks_along']]
        marks = []
        for t in np.linspace(-m['span'], m['span'], m['n_marks']):
            p = pos.copy(); p[ax] += t
            marks.append([round(float(x), 2) for x in p])
        out[key] = {'a': a, 'b': b, 'kind': m['kind'], 'manual': True,
                    'patches': [{'centroid': [round(float(x), 2) for x in pos],
                                 'normal': m['insert'], 'insert': m['insert'],
                                 'joint': 'slot', 'extent': [m['span'], 2, 2],
                                 'area': m['area'], 'n_pts': 0,
                                 'how': m['how'], 'marks': marks}]}
        print(f'  {key:50s} annotated by hand')


def main():
    fin = json.load(open(os.path.join(HERE, 'assembly_final.json')))
    pts = {n: surface_points(n, s) for n, s in fin.items()}
    print('sampled parts:', {k: len(v) for k, v in list(pts.items())[:3]},
          '...', flush=True)

    out = {}
    for a, b, kind in JOINTS:
        if a not in pts or b not in pts:
            print(f'  skip {a} -> {b} (missing)')
            continue
        A, B = pts[a], pts[b]
        # cheap reject: bounding boxes must overlap within tol
        if (A.min(0) > B.max(0) + TOL).any() or (B.min(0) > A.max(0) + TOL).any():
            print(f'  {a:26s} -> {b:22s} no bbox overlap')
            continue
        m = near_mask(A, B, TOL)
        C = A[m]
        if len(C) < MIN_PTS:
            print(f'  {a:26s} -> {b:22s} no contact found ({m.sum()} pts)')
            continue
        a_centre = A.mean(0)
        patches = []
        for g in cluster(C):
            Q = C[g]
            c = Q.mean(0)
            u, s, vt = np.linalg.svd(Q - c, full_matrices=False)
            ext = np.sqrt(np.maximum(s ** 2 / max(len(Q) - 1, 1), 0)) * 2
            nrm = vt[2]
            area = float(np.pi * ext[0] * ext[1])
            # insertion axis: along the shaft for a cylindrical contact,
            # perpendicular to the face for a flat one
            cylindrical = (ext[1] / max(ext[2], 1e-3) < 1.7
                           and ext[0] / max(ext[1], 1e-3) > 1.5)
            ins = vt[0] if cylindrical else nrm
            if float(ins @ (a_centre - c)) < 0:      # point it "outwards"
                ins = -ins
            patches.append({
                'centroid': [round(float(x), 2) for x in c],
                'normal': [round(float(x), 3) for x in nrm],
                'insert': [round(float(x), 3) for x in ins],
                'joint': 'shaft' if cylindrical else 'face',
                'extent': [round(float(x), 2) for x in ext],
                'area': round(area, 1),
                'n_pts': int(len(Q)),
                'how': describe(Q, ext, area),
                'marks': [[round(float(x), 2) for x in p]
                          for p in marks_for(Q, 14)],
            })
        if not patches:
            print(f'  {a:26s} -> {b:22s} contact too small to cluster '
                  f'({len(C)} pts)', flush=True)
            continue
        patches.sort(key=lambda p: -p['area'])
        out[f'{a}|{b}'] = {'a': a, 'b': b, 'kind': kind,
                           'patches': patches[:3]}
        tot = sum(p['area'] for p in patches[:3])
        print(f'  {a:26s} -> {b:22s} {len(patches)} patch(es), '
              f'area~{tot:6.1f}mm2  [{patches[0]["how"]}]', flush=True)

    add_manual(out, fin)
    pos_of = {n: s['pos'] for n, s in fin.items()}
    for j in out.values():
        for p in j['patches']:
            v = insertion_axis(j['a'], j['b'], p, pos_of)
            p['insert'] = [round(float(x), 3) for x in v]
    with open(os.path.join(HERE, 'glue_sites.json'), 'w') as f:
        json.dump(out, f, indent=1)
    print('saved glue_sites.json:', len(out), 'joints')


if __name__ == '__main__':
    main()
