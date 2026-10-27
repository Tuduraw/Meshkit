"""Shared-texel check (0.7.0): two VISIBLE surfaces that sample the same texels while lying in different places.

A projection atlas (top / side / front views of the model) gives every surface the texels of its projection. Where
two surfaces overlap in that projection - two rows of interleaved road wheels, a wheel face and the hull side behind
it, a mudguard over a track - only one of them can own a texel; the other one shows a picture painted for a place
it is not (the hub of the wheel behind drawn across the wheel in front, camouflage meant for the hull on a wheel).
The texture painter cannot see this; the preview only shows it from the angle where the wrong owner is visible.

check() renders the model from many directions to find the faces that are seen at all, rasterises the UV triangles of
those faces into the texture grid and, texel by texel, compares the 3D points that the faces map there. A texel is a
conflict when two visible faces of different shells map it to points further apart than `tol` (mirror images across
X are accepted: left and right sides sharing one picture is a normal economy - pass mirror_ok=False to report them
too). Flat swatch faces (all UVs equal) are ignored.

    from meshkit import texshare
    rep = texshare.check(model)            # model.uvs / model.texture from read_obj(..., texture=png)
    rep["texels"], rep["pairs"][:5]        # conflicting texels; worst part/shell pairs with where they are

Command line: python -m meshkit texshare model.obj --texture model.png [--parts '$wheel_*'] [--skip '$track_*'] [-o sheet.png]
Targets: 0 for parts that are painted with a picture of their own (wheels, markings); left/right sharing is allowed.
"""
import math

import numpy as np

from . import seethrough
from . import render as R


def _faces(model, skip=()):
    uvs = model.uvs
    out = []        # (part, shell, face, P (k,3), UV (k,2))
    for pname in model.order:
        if pname in skip:
            continue
        for si, sh in enumerate(model.parts[pname].shells):
            V = sh.V()
            for fi, f in enumerate(sh.faces):
                uv = uvs.get((pname, si, fi)) if uvs else None
                if uv is None:
                    continue
                out.append((pname, si, fi, V[list(f)], np.asarray(uv, float)))
    return out


def visible_faces(model, faces, n_views=24, res=300):
    """Render id buffers from n_views directions (far + near each, like seethrough).
    Returns ({face index: pixel count}, [(eye, target, fov, idbuf, zbuf)])."""
    tris, ids = [], []
    for k, (_, _, _, P, UV) in enumerate(faces):
        for t in range(1, len(P) - 1):
            idx = [0, t, t + 1]
            tris.append((P[idx], None, np.array([200, 200, 200], np.float32))); ids.append(k)
    seen = {}; bufs = []
    for (_, eye, tgt, fov) in seethrough.views(model, n_views):
        _, idb, zb = R.render(tris, None, W=res, H=res, eye=eye, target=tgt, fov=fov, ids=ids, return_ids=True,
                              return_depth=True)
        v, c = np.unique(idb[idb >= 0], return_counts=True)
        for a, b in zip(v, c):
            seen[int(a)] = seen.get(int(a), 0) + int(b)
        bufs.append((eye, tgt, fov, idb, zb))
    return seen, bufs


def _seen_points(points, face, bufs):
    """True where the 3D point of `face` (array) is the visible surface (or next to it) in at least one view."""
    out = np.zeros(len(points), bool)
    for (eye, tgt, fov, idb, zb) in bufs:
        H, W = idb.shape
        e, r, u, f = R.look_at(eye, tgt)
        rel = points - e
        xc, yc, zc = rel @ r, rel @ u, rel @ f
        ok = zc > 0.05
        fl = 0.5 * H / math.tan(math.radians(fov) / 2)
        zs = np.where(ok, zc, 1.0)
        sx = (W / 2 + fl * xc / zs).astype(np.int64); sy = (H / 2 - fl * yc / zs).astype(np.int64)
        for dx, dy in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
            x = sx + dx; y = sy + dy
            inb = ok & (x >= 0) & (x < W) & (y >= 0) & (y < H)
            hit = np.zeros(len(points), bool)
            yy, xx = y[inb], x[inb]
            # the point's own face is drawn there and the point is not behind it (grazing views map many hidden
            # points of a large face onto the few pixels where its visible part is drawn)
            hit[inb] = (idb[yy, xx] == face[inb]) & (zc[inb] <= zb[yy, xx] * 1.003 + 0.02)
            out |= hit
        if out.all():
            break
    return out


def check(model, n_views=24, res=300, skip=(), tol=0.03, mirror_ok=True, min_pixels=4, max_pairs=40, only=None):
    """only: part names - report only conflicts that involve at least one of them (all parts still occlude)."""
    tex = model.texture
    th, tw = (tex.shape[0], tex.shape[1]) if tex is not None else (1024, 1024)
    faces = _faces(model, skip)
    seen, bufs = visible_faces(model, faces, n_views, res)
    vis = [k for k, n in seen.items() if n >= min_pixels]
    owner = np.full(th * tw, -1, np.int64)
    opos = np.zeros((th * tw, 3), np.float32)
    C_cell, C_a, C_b, C_pa, C_pb = [], [], [], [], []
    for k in sorted(vis, key=lambda k: -seen[k]):
        pname, si, fi, P, UV = faces[k]
        px = np.stack([UV[:, 0] * tw, (1.0 - UV[:, 1]) * th], 1)
        if np.ptp(px[:, 0]) < 0.5 and np.ptp(px[:, 1]) < 0.5:
            continue                                  # flat swatch
        for t in range(1, len(P) - 1):
            idx = [0, t, t + 1]
            p = px[idx]; Q = P[idx]
            x0 = max(int(math.floor(p[:, 0].min())), 0); x1 = min(int(math.ceil(p[:, 0].max())), tw - 1)
            y0 = max(int(math.floor(p[:, 1].min())), 0); y1 = min(int(math.ceil(p[:, 1].max())), th - 1)
            if x1 < x0 or y1 < y0:
                continue
            X, Y = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
            (ax, ay), (bx, by), (cx, cy) = p
            area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
            if abs(area) < 1e-9:
                continue
            w0 = ((bx - X) * (cy - Y) - (by - Y) * (cx - X)) / area
            w1 = ((cx - X) * (ay - Y) - (cy - Y) * (ax - X)) / area
            w2 = 1.0 - w0 - w1
            ins = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
            if not ins.any():
                continue
            cell = (Y[ins].astype(np.int64)) * tw + X[ins].astype(np.int64)
            pts = (w0[ins, None] * Q[0] + w1[ins, None] * Q[1] + w2[ins, None] * Q[2]).astype(np.float32)
            free = owner[cell] < 0
            owner[cell[free]] = k; opos[cell[free]] = pts[free]
            o = owner[cell]
            other = o != k
            if not other.any():
                continue
            oc = cell[other]; op = opos[oc]; mp = pts[other]; ok_ = o[other]
            d = np.linalg.norm(op - mp, axis=1)
            if mirror_ok:
                d = np.minimum(d, np.linalg.norm(op - mp * np.array([-1, 1, 1], np.float32), axis=1))
            same_shell = np.array([faces[j][0] == pname and faces[j][1] == si for j in ok_])
            hit = (d > tol) & ~same_shell
            if only is not None and pname not in only:
                hit &= np.array([faces[j][0] in only for j in ok_])
            if hit.any():
                C_cell.append(oc[hit]); C_a.append(ok_[hit]); C_b.append(np.full(int(hit.sum()), k))
                C_pa.append(op[hit]); C_pb.append(mp[hit])
    bad = np.zeros(th * tw, bool)
    pairs = {}
    if C_cell:
        cell = np.concatenate(C_cell); fa = np.concatenate(C_a); fb = np.concatenate(C_b)
        pa = np.concatenate(C_pa).astype(float); pb = np.concatenate(C_pb).astype(float)
        both = _seen_points(pa, fa, bufs) & _seen_points(pb, fb, bufs)
        bad[cell[both]] = True
        for j, k, q in zip(fa[both], fb[both], pb[both]):
            key = (faces[j][:2], faces[k][:2])
            e = pairs.setdefault(key, [0, np.full(3, np.inf), np.full(3, -np.inf)])
            e[0] += 1; e[1] = np.minimum(e[1], q); e[2] = np.maximum(e[2], q)
    plist = [{"owner": f"{a[0]}#{a[1]}", "other": f"{b[0]}#{b[1]}", "texels": n,
              "other_lo": [round(float(v), 3) for v in lo], "other_hi": [round(float(v), 3) for v in hi]}
             for (a, b), (n, lo, hi) in pairs.items()]
    plist.sort(key=lambda r: -r["texels"])
    by_part = {}
    for r in plist:
        for side in ("owner", "other"):
            p = r[side].split("#")[0]
            by_part[p] = by_part.get(p, 0) + r["texels"]
    return {"texels": int(bad.sum()), "visible_faces": len(vis), "faces": len(faces),
            "by_part": dict(sorted(by_part.items(), key=lambda t: -t[1])), "pairs": plist[:max_pairs],
            "mask": bad.reshape(th, tw)}


def summary(rep):
    return {k: v for k, v in rep.items() if k != "mask"}


def sheet(model, rep):
    """The texture with conflicting texels in magenta (PIL Image)."""
    from PIL import Image
    tex = model.texture
    img = np.array(tex[..., :3] if tex is not None else np.full(rep["mask"].shape + (3,), 128, np.uint8))
    img = (img * 0.45).astype(np.uint8)
    img[rep["mask"]] = (255, 0, 255)
    return Image.fromarray(img)
