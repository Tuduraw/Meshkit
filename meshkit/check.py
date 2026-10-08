"""Mesh validation: watertightness, winding, face sanity, poly budget, floating shells."""
import numpy as np
from collections import Counter


def face_normal_area(P):
    n = np.zeros(3)
    m = len(P)
    for i in range(m):
        a = P[i]; b = P[(i + 1) % m]
        n += np.cross(a, b)
    area = np.linalg.norm(n) / 2
    return (n / (2 * area) if area > 1e-12 else n), area


def check_shell(sh, label=""):
    issues = []
    P = sh.V()
    directed = Counter()
    for f in sh.faces:
        for i in range(len(f)):
            directed[(f[i], f[(i + 1) % len(f)])] += 1
    undirected = Counter()
    for (a, b), c in directed.items():
        undirected[(min(a, b), max(a, b))] += c
        if c > 1:
            issues.append(f"{label}: directed edge {a}->{b} used {c}x (inconsistent winding)")
    for (a, b), c in undirected.items():
        if c != 2:
            issues.append(f"{label}: edge {a}-{b} has {c} faces (open/non-manifold)")
        elif directed[(a, b)] != 1 or directed[(b, a)] != 1:
            issues.append(f"{label}: edge {a}-{b} winding mismatch")
    vol = sh.signed_volume()
    if vol <= 0:
        issues.append(f"{label}: non-positive volume {vol:.4g} (inside-out)")
    # face sanity
    size = max(np.ptp(P, 0).max(), 1e-6)
    for fi, f in enumerate(sh.faces):
        Q = P[list(f)]
        n, area = face_normal_area(Q)
        if area < 1e-7 * size * size:
            issues.append(f"{label}: degenerate face {fi} area={area:.3g}")
            continue
        if len(f) >= 4:
            c = Q.mean(0)
            dev = np.abs((Q - c) @ n).max()
            edge = max(np.linalg.norm(Q[(i + 1) % len(f)] - Q[i]) for i in range(len(f)))
            if dev > 0.08 * edge + 1e-4:
                issues.append(f"{label}: non-planar face {fi} ({len(f)} verts, dev={dev:.3f}, edge={edge:.3f})")
            # convexity (mod fans ngons from vertex 0; GPU splits quads 0-1-2 / 2-3-0)
            sgn = None
            for i in range(len(f)):
                a, b, cc = Q[i], Q[(i + 1) % len(f)], Q[(i + 2) % len(f)]
                cr = np.dot(np.cross(b - a, cc - b), n)
                # relative tolerance: collinear corners (exact in code, slightly off after a round trip
                # through a 6-decimal file) must not count as concave
                if abs(cr) < 1e-10 or abs(cr) < 1e-4 * np.linalg.norm(b - a) * np.linalg.norm(cc - b):
                    continue
                s = cr > 0
                if sgn is None:
                    sgn = s
                elif s != sgn:
                    issues.append(f"{label}: non-convex face {fi} ({len(f)} verts)")
                    break
        # first-three-vertex normal must agree with the polygon normal (lighting)
        n3 = np.cross(Q[1] - Q[0], Q[2] - Q[0])
        l01 = np.linalg.norm(Q[1] - Q[0]); l02 = np.linalg.norm(Q[2] - Q[0])
        if np.linalg.norm(n3) > max(1e-12, 1e-4 * l01 * l02) and np.dot(n3, n) <= 0:
            issues.append(f"{label}: face {fi} first-triangle normal flipped")
    return issues


def check_model(model, parts_rest_only=True):
    issues = []
    for pname in model.order:
        for si, sh in enumerate(model.parts[pname].shells):
            issues += check_shell(sh, f"{pname}[{si}]{('/' + sh.name) if sh.name else ''}")
    return issues


def islands(model, exclude=()):
    """Group shells by AABB overlap (a cheap 'is it attached to something' test)."""
    boxes = []
    for pname in model.order:
        if pname in exclude:
            continue
        for si, sh in enumerate(model.parts[pname].shells):
            lo, hi = sh.bbox(); boxes.append((f"{pname}[{si}]", lo - 1e-3, hi + 1e-3))
    n = len(boxes)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for i in range(n):
        for j in range(i + 1, n):
            if np.all(boxes[i][1] <= boxes[j][2]) and np.all(boxes[j][1] <= boxes[i][2]):
                parent[find(i)] = find(j)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(boxes[i][0])
    return sorted(groups.values(), key=len, reverse=True)


def tri_count_faces(faces):
    return sum(len(f) - 2 for f in faces)


def report(model, contacts=True, max_issues=60, tol=0.004, skip=()):
    """One dict summarising everything an agent needs to judge a model:
    size / counts, closed-shell issues (holes, winding, degenerate / non-planar / non-convex faces),
    and - with contacts=True - groups of shells that touch each other (more than one group = floating parts)."""
    s = model.summary()
    issues = check_model(model)
    kinds = Counter()
    for i in issues:
        for k in ("inconsistent winding", "open/non-manifold", "winding mismatch", "inside-out", "degenerate",
                  "non-planar", "non-convex", "first-triangle normal"):
            if k in i:
                kinds[k] += 1; break
        else:
            kinds["other"] += 1
    out = {"summary": {k: v for k, v in s.items() if k != "per_part"}, "parts": s["per_part"],
           "issue_count": len(issues), "issue_kinds": dict(kinds), "issues": issues[:max_issues]}
    if contacts:
        from . import attach
        g = attach.groups(model, skip=skip, tol=tol)
        out["contact_groups"] = len(g)
        out["floating"] = g[1:]
    out["ok"] = not issues and (not contacts or out["contact_groups"] <= 1)
    return out


def clearance(model, part, pivot, axis, angles=range(0, 360, 10), tags=None, ignore=(), tol=0.0):
    """Swept-motion test: rotate `part` about `axis` through `pivot` to every angle (degrees) and report the
    shells of other parts it touches - e.g. propeller blades against the airframe, a turret's gun against the
    hull, a folding wing against the fuselage. tags: only test the moving part's shells that carry one of
    these face tags (e.g. ("prop", "prop_tip") to ignore the spinner that is meant to touch its mount).
    ignore: parts left out of the test. Returns {"hits": [(angle, "other[shell]"), ...], "ok": bool}."""
    from . import attach
    from .geom import rot_axis
    pv = np.asarray(pivot, float)
    movers = [sh for sh in model.parts[part].shells if tags is None or any(t in tags for t in sh.tags)]
    others = []
    for pname in model.order:
        if pname == part or pname in ignore:
            continue
        for si, sh in enumerate(model.parts[pname].shells):
            others.append((f"{pname}[{si}]", attach._tris(sh)))
    hits = []
    for a in angles:
        M = rot_axis(axis, a)
        for sh in movers:
            P, T, E = attach._tris(sh)
            P2 = (P - pv) @ M.T + pv
            T2 = P2[np.array([[f[0], f[k], f[k + 1]] for f in sh.faces for k in range(1, len(f) - 1)], int)]
            for name, data in others:
                if attach.attached((P2, T2, E), data, tol):
                    hits.append((a, name))
    hits = sorted(set(hits))
    return {"hits": hits, "ok": not hits}


# ------------------------------------------------------------------------------------------- v0.3 checks
def penetration(model, parts_a, parts_b, transforms=None, samples=2, tags_a=None, tags_b=None):
    """How much of parts_a lies INSIDE parts_b (e.g. tailplanes cutting through nozzles, a turret module
    sunk into the hull, a gun barrel inside a mast).  Returns {part_a: fraction of its surface points
    inside any shell of parts_b}.  Small embeddings for attachment are fine; look for large values."""
    from . import attach
    T = [attach._tris(sh)[1] for p in parts_b for sh in model.parts[p].shells
         if tags_b is None or any(t in tags_b for t in sh.tags)]
    out = {}
    for pa in parts_a:
        shells = [sh for sh in model.parts[pa].shells if tags_a is None or any(t in tags_a for t in sh.tags)]
        if not shells:
            continue
        P = np.vstack([attach._surface_points(sh, samples) for sh in shells])
        tf = (transforms or {}).get(pa)
        if tf is not None:
            P = tf(P)
        ins = np.zeros(len(P), bool)
        for tri in T:
            lo, hi = tri.reshape(-1, 3).min(0), tri.reshape(-1, 3).max(0)
            sel = ~ins & np.all((P >= lo) & (P <= hi), 1)
            if sel.any():
                ins[np.where(sel)[0][attach._inside(P[sel], tri)]] = True
        out[pa] = round(float(ins.mean()), 3)
    return out


def compare_size(model, expected, tol=0.03, parts=None):
    """Compare the bounding box with published dimensions.  expected = {"length": m, "width": m,
    "height": m} (any subset; length = Z, width = X, height = Y).  Returns {key: (model, published,
    relative error, ok)} - keep stores, rails, drums and other add-ons in mind when a value is over."""
    lo, hi = model.bbox(parts)
    size = {"width": hi[0] - lo[0], "height": hi[1] - lo[1], "length": hi[2] - lo[2]}
    out = {}
    for k, v in expected.items():
        e = (size[k] - v) / v
        out[k] = (round(float(size[k]), 3), v, round(float(e), 3), abs(e) <= tol)
    return out


def marking_spot(model, center, radius, side=1, axis="x", n=9, max_relief=0.08, parts=None):
    """Will a painted marking (roundel, number) of `radius` centred at `center` sit on one flat-ish patch?
    Rays are cast inwards along `axis` from the `side` (+1 / -1) over an n x n grid of the disc.  Returns
    {"coverage": fraction of rays that hit, "relief": depth spread of the hits (m), "ok": bool}.
    Low coverage = the marking runs off the edge (e.g. past the end of a wing root); high relief = it is
    painted across a step and will look cut."""
    from . import attach
    ai = "xyz".index(axis)
    oth = [i for i in range(3) if i != ai]
    c = np.asarray(center, float)
    g = np.linspace(-radius, radius, n)
    pts = []
    for u in g:
        for v in g:
            if u * u + v * v <= radius * radius:
                p = c.copy(); p[oth[0]] += u; p[oth[1]] += v; pts.append(p)
    P = np.array(pts)
    far = 1e3
    P0 = P.copy(); P0[:, ai] = side * far
    P1 = P.copy(); P1[:, ai] = -side * far
    best = np.full(len(P), np.nan)
    for pname in (parts or model.order):
        for sh in model.parts[pname].shells:
            tri = attach._tris(sh)[1]
            d = P1 - P0
            for j in range(len(P)):
                v0 = tri[:, 0]; e1 = tri[:, 1] - v0; e2 = tri[:, 2] - v0
                h = np.cross(d[j], e2); a = np.einsum("tk,tk->t", e1, h)
                ok = np.abs(a) > 1e-12
                f = np.where(ok, 1 / np.where(ok, a, 1), 0)
                s = P0[j] - v0
                uu = f * np.einsum("tk,tk->t", s, h)
                q = np.cross(s, e1)
                vv = f * np.einsum("k,tk->t", d[j], q)
                t = f * np.einsum("tk,tk->t", e2, q)
                hit = ok & (uu >= 0) & (vv >= 0) & (uu + vv <= 1) & (t >= 0) & (t <= 1)
                if hit.any():
                    coord = (P0[j, ai] + d[j, ai] * t[hit]) * side
                    m = coord.max()
                    if np.isnan(best[j]) or m > best[j]:
                        best[j] = m
    hitm = ~np.isnan(best)
    cov = float(hitm.mean())
    rel = float(best[hitm].max() - best[hitm].min()) if hitm.any() else 0.0
    return {"coverage": round(cov, 3), "relief": round(rel, 3), "ok": cov >= 0.999 and rel <= max_relief}
