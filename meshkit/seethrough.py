"""See-through check: where does a viewer look at the back of a surface?

The mod (and meshkit's preview) draws every face from its front side only, and a texel with alpha 0 is a hole.
So wherever a line of sight passes a hole - an opening cut into the skin, a missing or flipped face, a gap between
two parts, an inner face that shares the texels of an opening - and then meets a surface from behind, the viewer
sees straight through the model (the background), where something should be.

For each viewpoint, one pass over the triangles keeps four depth buffers:
  CO  nearest front face with an opaque texel (alpha > 0)      - what the mod shows (glass counts as solid)
  BO  nearest face of either side with an opaque texel
  CG  nearest front face with alpha >= 250 (glass is a hole too)
  BG  nearest face of either side with alpha >= 250
  "open"  pixel: CO empty, BO covered - the back of an opaque surface is in view: a defect.
  "glass" pixel: CG empty, BG covered (and not "open") - the back of a surface seen through glazing; the parts that
          carry the glass themselves are left out (single-skin canopies show no frames from inside - accepted).
An opening that only shows the sky beyond (an open cockpit seen from the side) is not reported: every face on such
a line of sight is a hole.

    from meshkit import seethrough
    rep = seethrough.check(model)                    # uses model.uvs / model.texture when present
    rep["open"], rep["glass"], rep["where"]          # pixel counts over all views; worst places by part
    seethrough.sheet(model, rep).save("st.png")      # the worst views, open pixels magenta, glass cyan

Command line: python -m meshkit seethrough model.obj [--texture tex.png] [-o st.png]
Rough targets (default 64 views of 360 px): open 0-3 px in all (a stray edge pixel), glass a few tens at most.
"""
import math

import numpy as np

GLASS_ALPHA = 250


def _triangles(model, skip=()):
    """Fan triangles of every face: P (n,3,3), UV (n,3,2), part index per triangle, part names."""
    uvs = getattr(model, "uvs", None)
    P, UV, part = [], [], []
    names = []
    for pname in model.order:
        if pname in skip:
            continue
        pi = len(names); names.append(pname)
        for si, sh in enumerate(model.parts[pname].shells):
            V = sh.V()
            for fi, face in enumerate(sh.faces):
                uv = np.asarray(uvs[(pname, si, fi)], float) if uvs else np.zeros((len(face), 2))
                for k in range(1, len(face) - 1):
                    idx = [0, k, k + 1]
                    P.append(V[[face[i] for i in idx]]); UV.append(uv[idx]); part.append(pi)
    return np.array(P, float).reshape(-1, 3, 3), np.array(UV, float).reshape(-1, 3, 2), np.array(part, np.int64), names


def look_at(eye, target, up=(0, 1, 0)):
    eye = np.asarray(eye, float); target = np.asarray(target, float)
    f = target - eye; f /= np.linalg.norm(f)
    r = np.cross(f, up); r /= np.linalg.norm(r)
    u = np.cross(r, f)
    return eye, r, u, f


def views(model, n=32, far=1.15, near=0.55):
    """Viewpoints all round the model (a Fibonacci sphere of n directions), each from far (fov 40) and from near (fov
    50). Distances are given in bounding-box diagonals from the box centre: far 1.15 (2.3 bounding-sphere radii), near
    0.55 (1.1 radii, just outside the model: close looks into cockpits, gun positions and gaps)."""
    lo, hi = model.bbox(); c = (np.asarray(lo) + np.asarray(hi)) / 2; R = float(np.linalg.norm(np.asarray(hi) - np.asarray(lo)))
    out = []
    ga = math.pi * (3 - math.sqrt(5))
    for i in range(n):
        y = 1 - 2 * (i + 0.5) / n; rr = math.sqrt(max(0.0, 1 - y * y)); th = ga * i
        d = np.array([rr * math.cos(th), y, rr * math.sin(th)])
        out.append((f"far{i}", c + d * R * far, c, 40.0))
        out.append((f"near{i}", c + d * R * near, c, 50.0))
    return out


def _clip_near(Q, UVq, zn):
    """Sutherland-Hodgman clip of a camera-space triangle against z = zn (keeps z > zn)."""
    out_p, out_t = [], []
    for i in range(3):
        a, b = Q[i], Q[(i + 1) % 3]; ta, tb = UVq[i], UVq[(i + 1) % 3]
        ia, ib = a[2] > zn, b[2] > zn
        if ia:
            out_p.append(a); out_t.append(ta)
        if ia != ib:
            s = (zn - a[2]) / (b[2] - a[2])
            out_p.append(a + s * (b - a)); out_t.append(ta + s * (tb - ta))
    return out_p, out_t


def raster4(P, UV, alpha, eye, target, fov, W, H, zn=0.05):
    """The four id / depth buffers (CO, BO, CG, BG) of one view; ids are triangle indices (-1 = nothing)."""
    eye, r, u, f = look_at(eye, target)
    rel = P - eye
    C = np.stack([rel @ r, rel @ u, rel @ f], -1)                     # (n, 3, 3) camera space
    fl = 0.5 * H / math.tan(math.radians(fov) / 2)
    th, tw = alpha.shape[:2]
    zb = np.full((4, H, W), np.inf); ids = np.full((4, H, W), -1, np.int64)

    def draw(t, sx, sy, iz, uz, vz, front):
        x0 = max(int(math.floor(sx.min())), 0); x1 = min(int(math.ceil(sx.max())), W - 1)
        y0 = max(int(math.floor(sy.min())), 0); y1 = min(int(math.ceil(sy.max())), H - 1)
        if x0 > x1 or y0 > y1:
            return
        d = (sy[1] - sy[2]) * (sx[0] - sx[2]) + (sx[2] - sx[1]) * (sy[0] - sy[2])
        if abs(d) < 1e-12:
            return
        X, Y = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
        w0 = ((sy[1] - sy[2]) * (X - sx[2]) + (sx[2] - sx[1]) * (Y - sy[2])) / d
        w1 = ((sy[2] - sy[0]) * (X - sx[2]) + (sx[0] - sx[2]) * (Y - sy[2])) / d
        w2 = 1 - w0 - w1
        inside = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
        if not inside.any():
            return
        q = w0 * iz[0] + w1 * iz[1] + w2 * iz[2]
        z = 1.0 / np.maximum(q, 1e-12)
        uu = (w0 * uz[0] + w1 * uz[1] + w2 * uz[2]) / q
        vv = (w0 * vz[0] + w1 * vz[1] + w2 * vz[2]) / q
        tx = np.clip((uu * tw).astype(np.int64), 0, tw - 1); ty = np.clip(((1 - vv) * th).astype(np.int64), 0, th - 1)
        a = alpha[ty, tx]
        sl = (slice(y0, y1 + 1), slice(x0, x1 + 1))
        for k, ok in ((0, front), (1, True), (2, front), (3, True)):
            if not ok:
                continue
            m = inside & ((a > 0) if k < 2 else (a >= GLASS_ALPHA)) & (z < zb[k][sl])
            if m.any():
                zb[k][sl][m] = z[m]; ids[k][sl][m] = t

    for t in range(len(P)):
        Q = C[t]
        if np.all(Q[:, 2] <= zn):
            continue
        if np.all(Q[:, 2] > zn):
            polys = [(Q, UV[t])]
        else:
            pp, tt = _clip_near(Q, UV[t], zn)
            if len(pp) < 3:
                continue
            polys = [(np.array([pp[0], pp[k], pp[k + 1]]), np.array([tt[0], tt[k], tt[k + 1]])) for k in range(1, len(pp) - 1)]
        for Qp, Tp in polys:
            iz = 1.0 / Qp[:, 2]
            sx = W / 2 + fl * Qp[:, 0] * iz; sy = H / 2 - fl * Qp[:, 1] * iz
            area = (sx[1] - sx[0]) * (-(sy[2] - sy[0])) - (-(sy[1] - sy[0])) * (sx[2] - sx[0])
            if abs(area) < 1e-9:
                continue
            draw(t, sx, sy, iz, Tp[:, 0] * iz, Tp[:, 1] * iz, area > 0)
    return ids, zb, (eye, r, u, f, fl)


def check(model, n_views=32, res=360, skip=(), keep_masks=4):
    """Pixels where the back of a surface is in view, over n_views directions x (far, near). Returns
    {"open": px, "glass": px, "views": n, "where": [{part, kind, px, views, lo, hi, med}], "worst": [...]}.
    `where` gives the part whose back face is seen and where (model coordinates); `worst` keeps the masks of the
    views with most pixels for sheet()."""
    P, UV, part, names = _triangles(model, skip)
    if not len(P):
        return {"open": 0, "glass": 0, "views": 0, "where": [], "worst": [], "skip": list(skip)}
    tex = getattr(model, "texture", None)
    if tex is not None and np.asarray(tex).ndim == 3 and np.asarray(tex).shape[2] == 4:
        alpha = np.ascontiguousarray(np.asarray(tex)[..., 3]).astype(np.uint8)
    else:
        alpha = np.full((1, 1), 255, np.uint8)
    # parts carrying glass (partial alpha): their own back faces seen through glass are not reported
    th, tw = alpha.shape
    cuv = UV.mean(1)
    a_c = alpha[np.clip(((1 - cuv[:, 1]) * th).astype(int), 0, th - 1), np.clip((cuv[:, 0] * tw).astype(int), 0, tw - 1)]
    glass_tri = (a_c > 0) & (a_c < GLASS_ALPHA)
    glazed_part = np.zeros(len(names) + 1, bool)
    for pi in np.unique(part[glass_tri]):
        glazed_part[pi] = True
    glazed = np.append(glazed_part[part], False)                  # index -1 -> False
    hits = {"open": [], "glass": []}
    worst = []
    vs = views(model, n_views)
    for name, eye, tgt, fov in vs:
        ids, zb, (e, r, u, f, fl) = raster4(P, UV, alpha, eye, tgt, fov, res, res)
        d_open = (ids[0] < 0) & (ids[1] >= 0)
        d_glass = (ids[2] < 0) & (ids[3] >= 0) & ~d_open & ~glazed[ids[3]]
        if not (d_open.any() or d_glass.any()):
            continue
        for kind, m, k in (("open", d_open, 1), ("glass", d_glass, 3)):
            if not m.any():
                continue
            ys, xs = np.nonzero(m)
            dirs = f[None] + r[None] * ((xs + 0.5 - res / 2) / fl)[:, None] - u[None] * ((ys + 0.5 - res / 2) / fl)[:, None]
            pts = e[None] + dirs * zb[k][ys, xs][:, None]
            for t, p in zip(ids[k][ys, xs], pts):
                hits[kind].append((name, int(part[t]), p))
        worst.append((int(d_open.sum()), int(d_glass.sum()), name, eye, tgt, fov, d_open, d_glass))
    where = []
    for kind in ("open", "glass"):
        agg = {}
        for name, pi, p in hits[kind]:
            g = agg.setdefault(pi, [0, [], set()])
            g[0] += 1; g[1].append(p); g[2].add(name)
        for pi, (cnt, pts, vset) in sorted(agg.items(), key=lambda kv: -kv[1][0]):
            pts = np.array(pts)
            where.append({"kind": kind, "part": names[pi], "px": cnt, "views": len(vset),
                          "lo": np.round(pts.min(0), 2).tolist(), "hi": np.round(pts.max(0), 2).tolist(),
                          "med": np.round(np.median(pts, 0), 2).tolist()})
    worst.sort(key=lambda w: -(w[0] + 0.3 * w[1]))
    return {"open": len(hits["open"]), "glass": len(hits["glass"]), "views": len(vs), "res": res, "where": where,
            "worst": worst[:keep_masks], "skip": list(skip)}


def summary(rep):
    """The report without the masks (JSON-friendly)."""
    out = {k: v for k, v in rep.items() if k != "worst"}
    out["worst_views"] = [{"view": w[2], "open": w[0], "glass": w[1]} for w in rep.get("worst", [])]
    return out


def sheet(model, rep, cols=2):
    """The worst views of a check() report with the open pixels in magenta and the glass ones in cyan (a few stray
    pixels are ringed so they can be found)."""
    from . import render
    from PIL import Image, ImageDraw
    worst = rep.get("worst") or []
    if not worst:
        return None
    res = rep.get("res", 360)
    skip = tuple(rep.get("skip") or ())
    grey = np.array([180.0, 180.0, 180.0], np.float32)
    tris = render.model_tris(model, model.uvs, None, skip=skip) if getattr(model, "uvs", None) else \
        [(t[0], None, grey) for t in _flat_tris(model, skip)]
    tex = getattr(model, "texture", None)
    imgs, labels = [], []
    for no, ng, name, eye, tgt, fov, mo, mg in worst:
        img = render.render(tris, tex, W=res, H=res, eye=eye, target=tgt, fov=fov, bg=(196, 210, 226)).copy()
        img[mg] = (0, 200, 255); img[mo] = (255, 0, 255)
        if no + ng <= 40:
            pil = Image.fromarray(img); dr = ImageDraw.Draw(pil)
            for m, col in ((mg, (0, 200, 255)), (mo, (255, 0, 255))):
                for y, x in zip(*np.nonzero(m)):
                    dr.ellipse((x - 7, y - 7, x + 7, y + 7), outline=col, width=2)
            img = np.array(pil)
        imgs.append(img); labels.append(f"{name}: open {no} px, glass {ng} px")
    return render.contact_sheet(imgs, labels, cols=cols, title=f"{model.name or 'model'}: back faces in view "
                                                                "(magenta) / through glass (cyan)")


def _flat_tris(model, skip=()):
    out = []
    for pname in model.order:
        if pname in skip:
            continue
        for sh in model.parts[pname].shells:
            V = sh.V()
            for face in sh.faces:
                for k in range(1, len(face) - 1):
                    out.append((V[[face[0], face[k], face[k + 1]]],))
    return out
