#!/usr/bin/env python3
"""Register every separate part STL against the fully assembled reference
STL: congruent-triangle matching (sorted edge-length triples) proposes rigid
transforms, a dilated occupancy grid of the assembly verifies them densely.
Fully vectorized. Recovers EXACT poses including wheel camber / wing angles.

Output: registration.json  {part: [{"R": 3x3, "t": [xyz], "inliers": f}]}
"""
import glob
import json
import os
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ASSEMBLY = os.path.join(ROOT, '2025+F1+Car+STL+Assembly (1).stl')
VOX = 0.6                      # occupancy voxel (mm); tolerance ~= 2*VOX


def read_stl(path):
    with open(path, 'rb') as f:
        f.seek(80)
        n = struct.unpack('<I', f.read(4))[0]
        data = np.frombuffer(f.read(n * 50), dtype=np.uint8)
        tri = data.reshape(n, 50)[:, 12:48].copy().view('<f4').reshape(n, 3, 3)
        return tri.astype(np.float64)


def edge_triples(tri):
    return np.stack([np.linalg.norm(tri[:, 1] - tri[:, 0], axis=1),
                     np.linalg.norm(tri[:, 2] - tri[:, 1], axis=1),
                     np.linalg.norm(tri[:, 0] - tri[:, 2], axis=1)], axis=1)


def canonical(tri1, e1):
    opp = np.array([2, 0, 1])
    return tri1[opp[np.argsort(-e1)]]


def rigid_from_tris(P, Q):
    cp, cq = P.mean(0), Q.mean(0)
    H = (P - cp).T @ (Q - cq)
    U, S, Vt = np.linalg.svd(H)
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, cq - R @ cp


def main():
    print('loading assembly ...', flush=True)
    A = read_stl(ASSEMBLY)
    VAll = A.reshape(-1, 3)
    lo = VAll.min(0) - 2
    print(f'assembly: {len(A)} tris', flush=True)

    # dilated occupancy grid (verification): mark voxels of tri VERTICES +
    # edge midpoints, then dilate by 1 voxel in each axis via shifts
    pts = np.concatenate([VAll, (A[:, 0] + A[:, 1]) / 2,
                          (A[:, 1] + A[:, 2]) / 2, (A[:, 0] + A[:, 2]) / 2])
    idx = np.floor((pts - lo) / VOX).astype(np.int32)
    dims = idx.max(0) + 3
    occ = np.zeros(dims, dtype=bool)
    occ[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    occ_d = occ.copy()
    for ax in range(3):
        occ_d |= np.roll(occ_d, 1, axis=ax) | np.roll(occ_d, -1, axis=ax)
    print(f'occupancy grid {dims} ({occ_d.mean()*100:.1f}% full)', flush=True)

    def inlier_frac(q):
        k = np.floor((q - lo) / VOX).astype(np.int32)
        ok = np.all((k >= 0) & (k < dims), axis=1)
        r = np.zeros(len(q), bool)
        kk = k[ok]
        r[ok] = occ_d[kk[:, 0], kk[:, 1], kk[:, 2]]
        return r.mean()

    # triangle lookup: quantized sorted edge triples -> assembly tri ids
    eA = edge_triples(A)
    keyA = np.round(np.sort(eA, axis=1) / 0.05).astype(np.int64)
    packA = keyA[:, 0] * 97003969 + keyA[:, 1] * 9973 + keyA[:, 2]
    order = np.argsort(packA)
    packS = packA[order]
    print('lookup ready', flush=True)

    def lookup(key):
        p = key[0] * 97003969 + key[1] * 9973 + key[2]
        i0 = np.searchsorted(packS, p, 'left')
        i1 = np.searchsorted(packS, p, 'right')
        return order[i0:i1]

    rng = np.random.RandomState(0)
    out = {}
    files = sorted(glob.glob(os.path.join(ROOT, 'Separate STLs', '*.stl')))
    files += sorted(glob.glob(os.path.join(HERE, 'tyre_q*.stl')))
    for path in files:
        name = os.path.splitext(os.path.basename(path))[0]
        if name == 'Tyres':
            continue
        P = read_stl(path)
        eP = edge_triples(P)
        srt = np.sort(eP, axis=1)
        distinct = (srt[:, 1] - srt[:, 0] > 0.08) & \
                   (srt[:, 2] - srt[:, 1] > 0.08)
        area = np.linalg.norm(np.cross(P[:, 1] - P[:, 0],
                                       P[:, 2] - P[:, 0]), axis=1)
        cand = np.nonzero(distinct & (area > np.percentile(area, 50)))[0]
        if len(cand) == 0:
            cand = np.nonzero(distinct)[0]
        rng.shuffle(cand)
        Vp = np.unique(P.reshape(-1, 3).round(2), axis=0)
        probe = Vp[rng.choice(len(Vp), min(400, len(Vp)), replace=False)]

        found = []
        for ti in cand[:100]:
            key = np.round(np.sort(eP[ti]) / 0.05).astype(np.int64)
            hits = lookup(key)
            if len(hits) == 0 or len(hits) > 150:
                continue
            cp = canonical(P[ti], eP[ti])
            for ai in hits:
                ca = canonical(A[ai], eA[ai])
                R, t = rigid_from_tris(cp, ca)
                if any(np.abs(t - f[1]).max() < 1.0
                       and np.abs(R - f[0]).max() < 1e-2 for f in found):
                    continue
                fr = inlier_frac(probe @ R.T + t)
                if fr > 0.9:
                    found.append((R, t, fr))
        out[name] = [{'R': np.round(f[0], 6).tolist(),
                      't': np.round(f[1], 4).tolist(),
                      'inliers': round(float(f[2]), 3)} for f in found]
        print(f'{name:28s} instances={len(found)} '
              f'inl={[round(float(f[2]),2) for f in found]}', flush=True)

    with open(os.path.join(HERE, 'registration.json'), 'w') as fjs:
        json.dump(out, fjs)
    print('saved registration.json', flush=True)


if __name__ == '__main__':
    main()
