"""Geometric attachment test: which closed shells actually touch or interpenetrate.

Two shells are 'attached' when an edge of one pierces a triangle of the other, a vertex of one
lies inside the other (ray parity), or a vertex of one lies on a face of the other (within
`tol`).  Shells are then grouped (union-find); anything not in the main group is floating and
would show a visible gap (light between parts).  Much stricter than the AABB test in check.py.

    python3 attach.py <vid> [pose]      (pose = a preview pose name, e.g. "gear up")
"""
import sys
import numpy as np

RAY = np.array([0.5772, 0.5775, 0.5776])
RAY = RAY / np.linalg.norm(RAY)


def _tris(sh):
    P = sh.V()
    T = []
    for f in sh.faces:
        for k in range(1, len(f) - 1):
            T.append((f[0], f[k], f[k + 1]))
    T = np.array(T, int)
    E = set()
    for f in sh.faces:
        for i in range(len(f)):
            a, b = f[i], f[(i + 1) % len(f)]
            E.add((min(a, b), max(a, b)))
    return P, P[T], np.array(sorted(E), int)


def _seg_tri(p0, p1, tri, eps=1e-9):
    """Segments (S,3)x2 vs triangles (T,3,3) -> bool (S,T) of proper crossings."""
    d = (p1 - p0)[:, None, :]
    v0 = tri[None, :, 0, :]; e1 = tri[None, :, 1, :] - v0; e2 = tri[None, :, 2, :] - v0
    h = np.cross(d, e2)
    a = np.einsum('stk,stk->st', e1, h)
    ok = np.abs(a) > eps
    f = np.where(ok, 1.0 / np.where(ok, a, 1.0), 0.0)
    s = p0[:, None, :] - v0
    u = f * np.einsum('stk,stk->st', s, h)
    q = np.cross(s, e1)
    v = f * np.einsum('stk,stk->st', d, q)
    t = f * np.einsum('stk,stk->st', e2, q)
    return ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t >= 0) & (t <= 1)


def _inside(P, tri):
    """Ray-parity point-in-closed-mesh for points P (N,3)."""
    if len(P) == 0 or len(tri) == 0:
        return np.zeros(len(P), bool)
    far = P + RAY * 1e4
    hits = _seg_tri(P, far, tri)
    return (hits.sum(1) % 2) == 1


def _on_face(P, tri, tol):
    """Points lying on a triangle (plane distance < tol, projection inside the triangle)."""
    if len(P) == 0 or len(tri) == 0:
        return np.zeros(len(P), bool)
    v0 = tri[:, 0]; e1 = tri[:, 1] - v0; e2 = tri[:, 2] - v0
    n = np.cross(e1, e2); nl = np.linalg.norm(n, axis=1); good = nl > 1e-12
    n = n[good] / nl[good, None]; v0 = v0[good]; e1 = e1[good]; e2 = e2[good]
    w = P[:, None, :] - v0[None]
    dist = np.einsum('ptk,tk->pt', w, n)
    proj = w - dist[..., None] * n[None]
    d00 = np.einsum('tk,tk->t', e1, e1); d01 = np.einsum('tk,tk->t', e1, e2); d11 = np.einsum('tk,tk->t', e2, e2)
    d20 = np.einsum('ptk,tk->pt', proj, e1); d21 = np.einsum('ptk,tk->pt', proj, e2)
    den = d00 * d11 - d01 * d01
    den = np.where(np.abs(den) < 1e-18, 1e-18, den)
    v = (d11 * d20 - d01 * d21) / den; w2 = (d00 * d21 - d01 * d20) / den
    e = 0.02
    return ((np.abs(dist) < tol) & (v >= -e) & (w2 >= -e) & (v + w2 <= 1 + e)).any(1)


def _box_overlap(lo1, hi1, lo2, hi2, tol):
    return np.all(lo1 <= hi2 + tol) and np.all(lo2 <= hi1 + tol)


def _tri_in_box(tri, lo, hi, tol):
    tl = tri.min(1); th = tri.max(1)
    return np.all(tl <= hi + tol, 1) & np.all(th >= lo - tol, 1)


def attached(A, B, tol=0.004, chunk=4000):
    PA, TA, EA = A
    PB, TB, EB = B
    loA, hiA = PA.min(0), PA.max(0)
    loB, hiB = PB.min(0), PB.max(0)
    if not _box_overlap(loA, hiA, loB, hiB, tol):
        return False
    lo = np.maximum(loA, loB) - tol; hi = np.minimum(hiA, hiB) + tol
    tA = TA[_tri_in_box(TA, lo, hi, tol)]
    tB = TB[_tri_in_box(TB, lo, hi, tol)]
    if len(tA) == 0 or len(tB) == 0:
        return False
    # vertices on / inside the other shell
    for P, tri_full, tri in ((PA, TB, tB), (PB, TA, tA)):
        sel = np.all((P >= lo) & (P <= hi), 1)
        Q = P[sel]
        if len(Q):
            for i in range(0, len(Q), 512):
                if _on_face(Q[i:i + 512], tri, tol).any() or _inside(Q[i:i + 512], tri_full).any():
                    return True
    # edges piercing faces
    for P, E, tri in ((PA, EA, tB), (PB, EB, tA)):
        p0 = P[E[:, 0]]; p1 = P[E[:, 1]]
        el = np.minimum(p0, p1); eh = np.maximum(p0, p1)
        sel = np.all(el <= hi, 1) & np.all(eh >= lo, 1)
        p0 = p0[sel]; p1 = p1[sel]
        step = max(1, chunk // max(1, len(tri)))
        for i in range(0, len(p0), step):
            if _seg_tri(p0[i:i + step], p1[i:i + step], tri).any():
                return True
    return False


def groups(model, transforms=None, skip=(), tol=0.004):
    names, data = [], []
    for pname in model.order:
        if pname in skip:
            continue
        tf = (transforms or {}).get(pname)
        for si, sh in enumerate(model.parts[pname].shells):
            P, T, E = _tris(sh)
            if tf is not None:
                P = tf(P)
                T = P[np.array([[f[0], f[k], f[k + 1]] for f in sh.faces for k in range(1, len(f) - 1)], int)]
            names.append(f"{pname}[{si}]")
            data.append((P, T, E))
    n = len(names)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for i in range(n):
        for j in range(i + 1, n):
            if find(i) == find(j):
                continue
            if attached(data[i], data[j], tol):
                parent[find(i)] = find(j)
    g = {}
    for i in range(n):
        g.setdefault(find(i), []).append(names[i])
    return sorted(g.values(), key=len, reverse=True)


def _surface_points(sh, k):
    """Vertices plus a barycentric grid of order k on every triangle (k=0: vertices only)."""
    P = sh.V()
    if k <= 0:
        return P
    T = _tris(sh)[1]
    W = np.array([(a, b, k - a - b) for a in range(k + 1) for b in range(k + 1 - a)], float) / k
    S = np.einsum("wk,tkc->twc", W, T).reshape(-1, 3)
    return np.vstack([P, S])


def outside(model, transforms, parts, skip=(), also_static=(), skip_tags=(), samples=0, detail=False):
    """Fraction of each (moved) part's surface points lying outside every static shell - how much of a
    retracted undercarriage leg / wheel still hangs out in the airflow.  also_static: animated parts
    that count as structure here (e.g. folding outer wing panels holding the wheel wells).
    skip_tags: shells of the measured parts whose tags are all in this set are not measured (e.g. "door":
    bay doors are meant to lie on the skin).  samples: k > 0 also tests a barycentric grid of order k on
    every triangle, so a wheel poking through a skin between its vertices is caught.
    detail=True returns {part: {"outside": fraction, "points": [outside points (max 20)]}}."""
    static = []
    for pname in model.order:
        if pname in skip or pname in parts or (pname.startswith("$") and pname not in also_static):
            continue
        for sh in model.parts[pname].shells:
            static.append(_tris(sh)[1])
    res = {}
    for pname in parts:
        tf = (transforms or {}).get(pname)
        shells = [sh for sh in model.parts[pname].shells
                  if not (skip_tags and sh.tags and all(t in skip_tags for t in sh.tags))]
        if not shells:
            continue
        P = np.vstack([_surface_points(sh, samples) for sh in shells])
        if tf is not None:
            P = tf(P)
        ins = np.zeros(len(P), bool)
        for T in static:
            lo, hi = T.reshape(-1, 3).min(0), T.reshape(-1, 3).max(0)
            sel = ~ins & np.all((P >= lo) & (P <= hi), 1)
            if sel.any():
                ins[np.where(sel)[0][_inside(P[sel], T)]] = True
        frac = round(float(1 - ins.mean()), 3)
        res[pname] = {"outside": frac, "points": np.round(P[~ins][:20], 3).tolist()} if detail else frac
    return res


