"""Robots (mechs): limb and armour shapes, joint rotations and proportion checks.

Gathered from the Mech Frame addon (tudursvehiclemod) - 9 stock frames, 3 transforming sets and an
expansion pack of 4 makers - see AI_GUIDE.md chapter 7 for how to use them. Coordinates as everywhere
in meshkit: +X left, +Y up, +Z forward, 1 unit = 1 m.

Shapes (all closed, outward shells):

    rrect / ring_frame        chamfered rectangle rings (8 points)
    stack(levels)             vertical loft of chamfered rectangles (blocks, hulls, torsos)
    seg(points, sizes)        chamfered-rectangle loft along a polyline (limb segments, struts)
    cyl / cbox / mirror / nozzle
    pent2d / pent_ring        rounded pentagon pointing forward (the "light" language)
    pstack / pseg / zloft     pentagon lofts: vertical, along a polyline, along Z pointing up (prows, spines)
    plate / slab / hbox       bolted armour plate, extruded convex polygon, faceted heavy lump
    tube / truss              bare frame: round tubes, an open lattice girder (skeletal frames)
    beam / girder / at        square-tube members; triangular-section girder with diagonal webs
    panel(outline, ...)       separate armour plate on stand-off posts (posts checked to land on it)
    claw(root, tip, w, h)     tapering blade (toes, spikes) in 6 triangles
    chine_ring / chine_seg / chine_stack
                              diamond / hexagonal sections with sharp chines (faceted stealth language)
    split_warped(shell)       triangulate warped / concave quads of a loft (keeps the checks clean)
    cast(stations)            rounded cast-armour lump from superellipse sections (fortress language)
    cast_seg(points, sizes)   the same along a polyline (thick limbs)

Joints:  R3, axis_angle, align, frame_rot, M4, quat, qmat, slerp
Checks:  volume, fill_ratio, proportions, limb_report
"""
import math

import numpy as np

from .geom import loft, cylinder, lathe, prism, extrude_xz, ring_superellipse

# ---------------------------------------------------------------------------------- rings and lofts


def rrect(cx, cz, hw, hd, c, y):
    """Chamfered rectangle ring (8 points) in the plane y = const."""
    c = max(0.004, min(c, hw * 0.45, hd * 0.45))
    pts = [(cx + hw - c, cz - hd), (cx + hw, cz - hd + c), (cx + hw, cz + hd - c), (cx + hw - c, cz + hd),
           (cx - hw + c, cz + hd), (cx - hw, cz + hd - c), (cx - hw, cz - hd + c), (cx - hw + c, cz - hd)]
    return np.array([(x, y, z) for x, z in pts], float)


def stack(levels, tag="armor", c=0.08):
    """Vertical loft of chamfered rectangles. levels: [(y, cx, cz, hw, hd[, c]), ...] bottom to top."""
    rings = []
    for lv in levels:
        y, cx, cz, hw, hd = lv[:5]
        cc = lv[5] if len(lv) > 5 else c
        rings.append(rrect(cx, cz, hw, hd, cc, y))
    return loft(rings, tag)


def ring_frame(center, u, v, hu, hv, c):
    """Chamfered rectangle (8 points) in the plane spanned by unit vectors u, v around center."""
    c = max(0.004, min(c, hu * 0.45, hv * 0.45))
    pts2 = [(hu - c, -hv), (hu, -hv + c), (hu, hv - c), (hu - c, hv), (-hu + c, hv), (-hu, hv - c), (-hu, -hv + c), (-hu + c, -hv)]
    center = np.asarray(center, float)
    return np.array([center + a * u + b * v for a, b in pts2])


def _frames(pts, side):
    """(u, v, w) per polyline point: w along the line, u toward `side` (kept perpendicular)."""
    side = np.asarray(side, float)
    out = []
    for i, p in enumerate(pts):
        d = (pts[1] - pts[0]) if i == 0 else (pts[-1] - pts[-2]) if i == len(pts) - 1 else (pts[i + 1] - pts[i - 1])
        w = d / np.linalg.norm(d)
        u = side - w * np.dot(side, w)
        if np.linalg.norm(u) < 1e-6:
            u = np.cross([0, 0, 1.0], w)
        u = u / np.linalg.norm(u)
        out.append((u, np.cross(w, u), w))
    return out


def seg(points, sizes, tag="armor", c=0.06, side=(1, 0, 0)):
    """Loft of chamfered rectangles along a polyline. sizes: [(half_side, half_depth[, chamfer]), ...]
    per point. side: preferred direction of the rings' first axis (kept perpendicular to the segment)."""
    pts = [np.asarray(p, float) for p in points]
    rings = []
    for i, (u, v, _) in enumerate(_frames(pts, side)):
        hs, hd = sizes[i][:2]
        cc = sizes[i][2] if len(sizes[i]) > 2 else c
        rings.append(ring_frame(pts[i], u, v, hs, hd, cc))
    return loft(rings, tag)


def cyl(p0, p1, r, n=8, tag="frame", r1=None):
    return cylinder(p0, p1, r, r1, n=n, tag=tag)


def cbox(cx, cy, cz, sx, sy, sz, tag="armor", c=0.05):
    """Chamfered box (chamfer on the vertical edges and a bevel ring top and bottom)."""
    b = min(c, sy * 0.3)
    return stack([(cy - sy / 2, cx, cz, sx / 2 - b, sz / 2 - b, c),
                  (cy - sy / 2 + b, cx, cz, sx / 2, sz / 2, c),
                  (cy + sy / 2 - b, cx, cz, sx / 2, sz / 2, c),
                  (cy + sy / 2, cx, cz, sx / 2 - b, sz / 2 - b, c)], tag)


def wedge_xz(poly_xz, y0, y1, tag="armor"):
    return extrude_xz(poly_xz, y0, y1, tag)


def mirror(shells):
    """Mirror shells across X (keeps the outward winding)."""
    return [s.mirrored_x() for s in shells]


def nozzle(p, d, r, length, tag="nozzle", n=8):
    """A bell nozzle opening toward d (outer cone, closed)."""
    p = np.asarray(p, float)
    d = np.asarray(d, float) / np.linalg.norm(d)
    return lathe(p, d, [(0, r * 0.7), (length * 0.5, r * 0.85), (length, r)], n=n, tag=tag)


# ------------------------------------------------------------------ pentagons (light language)

def _round_poly(pts, rnd):
    """Each corner of a convex polygon replaced by two points (rnd = fraction of the adjacent edges)."""
    n = len(pts)
    out = []
    for i in range(n):
        p = np.asarray(pts[i], float)
        a = np.asarray(pts[i - 1], float)
        b = np.asarray(pts[(i + 1) % n], float)
        out.append(p + (a - p) * rnd)
        out.append(p + (b - p) * rnd)
    return out


def pent2d(hw, front, rear, sh=0.4, rb=0.8, rnd=0.22, tip=0.0):
    """2D rounded pentagon (side, forward), pointing +forward: tip, two shoulders, two rear corners.
    tip > 0 flattens the tip into a short edge of that half width. 10 (or 12) points, CCW seen from +up."""
    if tip > 0:
        poly = [(tip, front), (-tip, front), (-hw, front * sh), (-hw * rb, -rear), (hw * rb, -rear), (hw, front * sh)]
    else:
        poly = [(0.0, front), (-hw, front * sh), (-hw * rb, -rear), (hw * rb, -rear), (hw, front * sh)]
    return _round_poly(poly, rnd)


def pent_ring(cx, cz, hw, front, rear, y, **kw):
    """Rounded pentagon in the plane y = const, pointing +Z, centred at (cx, cz)."""
    return np.array([(cx + a, y, cz + b) for a, b in pent2d(hw, front, rear, **kw)], float)


def pstack(levels, tag="armor", **kw):
    """Vertical loft of rounded pentagons. levels: [(y, cx, cz, hw, front, rear[, {overrides}]), ...]."""
    rings = []
    for lv in levels:
        o = dict(kw)
        if len(lv) > 6:
            o.update(lv[6])
        rings.append(pent_ring(lv[1], lv[2], lv[3], lv[4], lv[5], lv[0], **o))
    return loft(rings, tag)


def pseg(points, sizes, tag="armor", fwd=(0, 0, 1), **kw):
    """Loft of rounded pentagons along a polyline, pointing toward fwd (kept perpendicular to the
    segment). sizes: [(half_side, front, rear[, {overrides}]), ...] per point."""
    pts = [np.asarray(p, float) for p in points]
    fwd = np.asarray(fwd, float)
    rings = []
    for i, p in enumerate(pts):
        d = (pts[1] - pts[0]) if i == 0 else (pts[-1] - pts[-2]) if i == len(pts) - 1 else (pts[i + 1] - pts[i - 1])
        w = d / np.linalg.norm(d)
        v = fwd - w * np.dot(fwd, w)
        if np.linalg.norm(v) < 1e-6:
            v = np.cross(w, [1.0, 0, 0])
        v /= np.linalg.norm(v)
        u = np.cross(v, w)
        sz = sizes[i]
        o = dict(kw)
        if len(sz) > 3:
            o.update(sz[3])
        rings.append(np.array([p + a * u + b * v for a, b in pent2d(sz[0], sz[1], sz[2], **o)]))
    return loft(rings, tag)


def zloft(stations, tag="armor", tip=None, **kw):
    """Loft along Z of rounded pentagons pointing UP (ridge on top): a prow or a spine.
    stations: [(z, cx, cy, hw, up, down[, {overrides}]), ...]; tip: optional end point."""
    rings = []
    for st in stations:
        o = dict(kw)
        if len(st) > 6:
            o.update(st[6])
        z, cx, cy, hw, up, down = st[:6]
        rings.append(np.array([(cx + a, cy + b, z) for a, b in pent2d(hw, up, down, **o)], float))
    if tip is not None:
        rings.append(np.array([tip], float))
    return loft(rings, tag)


# ------------------------------------------------------------------ plates (heavy language)

def plate(center, u, v, hu, hv, t, tag="armor", c=0.06, bevel=0.35):
    """Armour plate: chamfered rectangle (hu x hv in the u, v plane) of thickness t along u x v,
    with the outer face (+n) bevelled."""
    center = np.asarray(center, float)
    u = np.asarray(u, float); u = u / np.linalg.norm(u)
    v = np.asarray(v, float); v = v - u * np.dot(u, v); v = v / np.linalg.norm(v)
    n = np.cross(u, v)
    b = min(t * bevel, hu * 0.3, hv * 0.3)
    rings = [ring_frame(center - n * t / 2, u, v, hu, hv, c),
             ring_frame(center + n * (t / 2 - b), u, v, hu, hv, c),
             ring_frame(center + n * t / 2, u, v, hu - b, hv - b, max(0.004, c - b * 0.5))]
    return loft(rings, tag)


def slab(poly_uv, center, u, v, t, tag="armor"):
    """Convex polygon (in the u, v plane around center) extruded by t along u x v (both faces)."""
    center = np.asarray(center, float)
    u = np.asarray(u, float); u = u / np.linalg.norm(u)
    v = np.asarray(v, float); v = v - u * np.dot(u, v); v = v / np.linalg.norm(v)
    n = np.cross(u, v)
    a = [center + x * u + y * v - n * t / 2 for x, y in poly_uv]
    b = [center + x * u + y * v + n * t / 2 for x, y in poly_uv]
    return prism(a, b, tag=tag)


def hbox(cx, cy, cz, sx, sy, sz, tag="armor", c=0.12, top=0.75, bottom=0.9):
    """Heavy block: strongly chamfered box whose top (and bottom) rings are narrowed - a faceted lump."""
    return stack([(cy - sy / 2, cx, cz, sx / 2 * bottom, sz / 2 * bottom, c * 0.8),
                  (cy - sy / 2 + sy * 0.18, cx, cz, sx / 2, sz / 2, c),
                  (cy + sy / 2 - sy * 0.22, cx, cz, sx / 2, sz / 2, c),
                  (cy + sy / 2, cx, cz, sx / 2 * top, sz / 2 * top, c * 0.7)], tag)


# ------------------------------------------------------------------ bare frame (skeletal language)

def tube(points, r, tag="frame", n=6, joint=1.25):
    """Round tube along a polyline: one cylinder per leg, a slightly fatter collar at every bend
    (the legs overlap the collars, so the tube reads as one welded piece; joint=0 leaves the collars
    out to save triangles). Returns a shell list."""
    pts = [np.asarray(p, float) for p in points]
    out = []
    for a, b in zip(pts[:-1], pts[1:]):
        out.append(cylinder(a, b, r, n=n, tag=tag))
    for p, q in zip(pts[1:-1], pts[2:]) if joint and joint > 0 else []:
        d = q - p
        d = d / np.linalg.norm(d)
        out.append(cylinder(p - d * r * 1.1, p + d * r * 1.1, r * joint, n=n, tag=tag))
    return out


def truss(p0, p1, width, depth=None, bays=3, r=0.03, chord_r=None, tag="frame", side=(1, 0, 0), n=5):
    """Open lattice girder from p0 to p1: four chords at the corners of a width x depth section and a
    zig-zag of diagonals on the two broad faces. The skeletal frames' limbs - light, see-through, but
    still closed shells (each member is a thin cylinder). Returns a shell list."""
    p0 = np.asarray(p0, float)
    p1 = np.asarray(p1, float)
    depth = width if depth is None else depth
    chord_r = r * 1.4 if chord_r is None else chord_r
    (u, v, w), = _frames([p0, p1], side)[:1]
    hw, hd = width / 2, depth / 2
    corners = [(hw, hd), (-hw, hd), (-hw, -hd), (hw, -hd)]
    out = [cylinder(p0 + a * u + b * v, p1 + a * u + b * v, chord_r, n=n, tag=tag) for a, b in corners]
    L = np.linalg.norm(p1 - p0)
    for face in (hd, -hd):
        for i in range(bays):
            t0, t1 = i / bays, (i + 1) / bays
            a = hw if i % 2 == 0 else -hw
            q0 = p0 + w * L * t0 + a * u + face * v
            q1 = p0 + w * L * t1 - a * u + face * v
            out.append(cylinder(q0, q1, r, n=n, tag=tag))
    return out


# ------------------------------------------------------------------ chines (faceted language)

# ------------------------------------------------------------------ braced frame and stand-off armour
# (load-bearing skeleton: the Vesper lesson - a bare rod looks like it snaps under its own weight)

def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def beam(a, b, r=0.045, tag="metal", n=4):
    """Square-section titanium member (4-sided cylinder): reads as a box beam, 12 triangles. Thin web
    members and posts use n=3 (8 triangles) - at their size the section does not show."""
    return cyl(np.asarray(a, float), np.asarray(b, float), r, n=n, tag=tag)


def girder(p0, p1, face, w=0.22, r=0.042, web_r=0.024, bays=2, tag="metal", webs=("f1", "f2")):
    """Triangular-section girder from p0 to p1: two chords on the flat face (towards `face`), one apex
    chord behind them and a Warren web of diagonals on the faces listed in `webs` ("f1", "f2": the two
    inclined faces to the apex; "flat": the plated face). The flat face carries the armour plates.
    Returns (shells, chord lines {f1, f2, apex: (start, end)}, (d, f, a)) with d
    along the girder, f the face normal and a across the face (f1 = +a side)."""
    p0 = np.asarray(p0, float)
    p1 = np.asarray(p1, float)
    d = unit(p1 - p0)
    f = np.asarray(face, float)
    f = unit(f - d * np.dot(f, d))
    a = unit(np.cross(d, f))
    corner = {"f1": f * w * 0.32 + a * w * 0.5, "f2": f * w * 0.32 - a * w * 0.5, "apex": -f * w * 0.45}
    lines = {k: (p0 + c, p1 + c) for k, c in corner.items()}
    out = [beam(*lines[k], r=r, tag=tag) for k in ("f1", "f2", "apex")]

    def zig(A, B):
        pts = []
        for i in range(bays + 1):
            pts.append(A[0] + (A[1] - A[0]) * (i / bays))
            if i < bays:
                pts.append(B[0] + (B[1] - B[0]) * ((i + 0.5) / bays))
        return [beam(pts[i], pts[i + 1], r=web_r, tag=tag, n=3) for i in range(len(pts) - 1)]

    for wb in webs:
        out += zig(lines["f1"], lines["f2"]) if wb == "flat" else zig(lines[wb], lines["apex"])
    return out, lines, (d, f, a)


def at(line, t):
    return line[0] + (line[1] - line[0]) * t


def _hull2d(pts):
    """Convex hull (counter-clockwise) of 2D points - panels are given as a loose outline."""
    pts = sorted(set((float(x), float(y)) for x, y in pts))

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lo, hi = [], []
    for p in pts:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0:
            lo.pop()
        lo.append(p)
    for p in reversed(pts):
        while len(hi) >= 2 and cross(hi[-2], hi[-1], p) <= 0:
            hi.pop()
        hi.append(p)
    return lo[:-1] + hi[:-1]


def _inside(poly, p, margin=0.015):
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        ex, ey = b[0] - a[0], b[1] - a[1]
        cr = ex * (p[1] - a[1]) - ey * (p[0] - a[0])
        if cr < margin * math.hypot(ex, ey):
            return False
    return True


def panel(outline, center, u, v, posts=(), t=0.035, tag="armor", post_r=0.022):
    """A separate armour plate on stand-off posts: a convex plate (outline in the u, v plane around
    center, thickness t) and one square post from each frame point in `posts` straight to the plate
    (the post foot must land on the plate - checked). Returns shells."""
    center = np.asarray(center, float)
    u = unit(u)
    v = np.asarray(v, float)
    v = unit(v - u * np.dot(u, v))
    n = np.cross(u, v)
    poly = _hull2d(outline)
    out = [slab(poly, center, u, v, t, tag)]
    for p in posts:
        p = np.asarray(p, float)
        dd = float(np.dot(center - p, n))
        q = p + n * dd
        uv = (float(np.dot(q - center, u)), float(np.dot(q - center, v)))
        if not _inside(poly, uv):
            raise ValueError(f"panel post foot {uv} misses the plate {poly}")
        out.append(cyl(p, q + n * np.sign(dd) * t * 0.3, post_r, n=3, tag="frame"))
    return out


def claw(root, tip, w, h, tag):
    """A claw blade: a flat rectangle at the root tapering to the tip point (6 triangles)."""
    root = np.asarray(root, float)
    tip = np.asarray(tip, float)
    d = unit(tip - root)
    a = unit(np.cross(d, (0, 1, 0)))
    b = np.cross(a, d)
    ring = np.array([root + a * w + b * h, root - a * w + b * h, root - a * w - b * h, root + a * w - b * h])
    return loft([ring, np.array([tip])], tag)


def split_warped(shell, tol=0.03):
    """Split every quad of a shell that is warped or not convex into two triangles (along the
    diagonal that keeps both halves facing the same way as the quad). Lofts whose sections change
    shape along the way (chined hexagons, bent limbs) give such quads; triangles keep the faceted
    look and pass the checks. Returns the shell."""
    V = [np.asarray(v, float) for v in shell.verts]
    faces, tags = [], []
    for f, t in zip(shell.faces, shell.tags):
        if len(f) != 4:
            faces.append(f); tags.append(t)
            continue
        Q = np.array([V[i] for i in f])
        n = np.cross(Q[2] - Q[0], Q[3] - Q[1])
        ln = np.linalg.norm(n)
        bad = ln < 1e-12
        if not bad:
            n = n / ln
            edge = max(np.linalg.norm(Q[(i + 1) % 4] - Q[i]) for i in range(4))
            bad = np.abs((Q - Q.mean(0)) @ n).max() > tol * edge
            if not bad:
                sg = [np.dot(np.cross(Q[(i + 1) % 4] - Q[i], Q[(i + 2) % 4] - Q[(i + 1) % 4]), n) > 0 for i in range(4)]
                bad = len(set(sg)) > 1
        if not bad:
            faces.append(f); tags.append(t)
            continue
        a, b, c, d = f
        # pick the shorter diagonal (the flatter, convex split for a gently warped quad)
        if np.linalg.norm(V[a] - V[c]) <= np.linalg.norm(V[b] - V[d]):
            faces += [(a, b, c), (a, c, d)]
        else:
            faces += [(a, b, d), (b, c, d)]
        tags += [t, t]
    shell.faces, shell.tags = faces, tags
    return shell


def chine2d(hw, up, down, k=0.55, kb=None):
    """Hexagonal section with sharp side chines: (side, up) points CCW. k: how far out the top facets
    reach before turning down to the chine (0..1, smaller = steeper roof); kb the same for the belly."""
    kb = k if kb is None else kb
    return [(hw, 0.0), (hw * k, up), (-hw * k, up), (-hw, 0.0), (-hw * kb, -down), (hw * kb, -down)]


def chine_ring(center, u, v, hw, up, down, **kw):
    center = np.asarray(center, float)
    return np.array([center + a * u + b * v for a, b in chine2d(hw, up, down, **kw)])


def chine_seg(points, sizes, tag="armor", up=(0, 1, 0), **kw):
    """Loft of chined hexagons along a polyline (limbs and nacelles of a faceted frame). sizes:
    [(half_width, up, down[, {overrides}]), ...]; `up` picks the roof side (kept perpendicular)."""
    pts = [np.asarray(p, float) for p in points]
    up = np.asarray(up, float)
    rings = []
    for i, p in enumerate(pts):
        d = (pts[1] - pts[0]) if i == 0 else (pts[-1] - pts[-2]) if i == len(pts) - 1 else (pts[i + 1] - pts[i - 1])
        w = d / np.linalg.norm(d)
        v = up - w * np.dot(up, w)
        if np.linalg.norm(v) < 1e-6:
            v = np.cross(w, [1.0, 0, 0])
        v /= np.linalg.norm(v)
        u = np.cross(v, w)
        sz = sizes[i]
        o = dict(kw)
        if len(sz) > 3:
            o.update(sz[3])
        rings.append(chine_ring(p, u, v, sz[0], sz[1], sz[2], **o))
    return split_warped(loft(rings, tag))


def chine_stack(levels, tag="armor", **kw):
    """Vertical loft whose sections are chined hexagons in plan (pointing +Z / -Z): levels
    [(y, cx, cz, hw, front, rear[, {overrides}]), ...] - a faceted torso or hull seen from above."""
    rings = []
    for lv in levels:
        o = dict(kw)
        if len(lv) > 6:
            o.update(lv[6])
        y, cx, cz, hw, front, rear = lv[:6]
        rings.append(np.array([(cx + a, y, cz + b) for a, b in chine2d(hw, front, rear, **o)], float))
    return split_warped(loft(rings, tag))


# ------------------------------------------------------------------ cast armour (fortress language)

def cast(stations, tag="armor", n=12, p=2.6, axis="z"):
    """Rounded lump of cast armour lofted from superellipse sections. stations along Z:
    [(z, cx, cy, hw, top, bottom[, p]), ...] (axis="z"), or along Y with (y, cx, cz, hw, front, rear[, p])
    (axis="y", sections in plan). p > 2 gives the squarish, thick-shouldered look of cast steel."""
    rings = []
    for st in stations:
        pp = st[6] if len(st) > 6 else p
        if axis == "z":
            z, cx, cy, hw, top, bot = st[:6]
            rings.append(ring_superellipse(cx, cy, z, hw, top, bot, n=n, p=pp))
        else:
            y, cx, cz, hw, front, rear = st[:6]
            r = ring_superellipse(cx, cz, y, hw, front, rear, n=n, p=pp)
            # ring_superellipse lies in z = const with (x, y): swap y/z so it lies in plan
            r = np.array([(x, y, zz) for x, zz, y in r])
            rings.append(r[::-1])
    return split_warped(loft(rings, tag))


def cast_seg(points, sizes, tag="armor", side=(1, 0, 0), n=12, p=2.6):
    """Cast-armour loft along a polyline (thick limbs of a fortress frame): superellipse sections
    sizes [(half_side, half_depth[, p]), ...], the first axis kept toward `side`."""
    pts = [np.asarray(q, float) for q in points]
    rings = []
    for i, (u, v, _) in enumerate(_frames(pts, side)):
        hs, hd = sizes[i][:2]
        pp = sizes[i][2] if len(sizes[i]) > 2 else p
        ring = []
        for k in range(n):
            th = 2 * math.pi * k / n
            c, s_ = math.cos(th), math.sin(th)
            a = hs * math.copysign(abs(c) ** (2.0 / pp), c)
            b = hd * math.copysign(abs(s_) ** (2.0 / pp), s_)
            ring.append(pts[i] + a * u + b * v)
        rings.append(np.array(ring))
    return split_warped(loft(rings, tag))


# ---------------------------------------------------------------------------------- rotations

def Rx(d):
    t = math.radians(d)
    c, s = math.cos(t), math.sin(t)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], float)


def Ry(d):
    t = math.radians(d)
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], float)


def Rz(d):
    t = math.radians(d)
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float)


def R3(rx=0.0, ry=0.0, rz=0.0):
    """Ry·Rx·Rz (degrees) as a 3x3 - the Euler order tudursvehiclemod uses for part rotations."""
    return Ry(ry) @ Rx(rx) @ Rz(rz)


def axis_angle(axis, deg):
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    t = math.radians(deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(t) * K + (1 - math.cos(t)) * K @ K


def align(frm, to):
    """Smallest rotation carrying direction frm onto direction to."""
    a = np.asarray(frm, float); a = a / np.linalg.norm(a)
    b = np.asarray(to, float); b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-9:
        if c > 0:
            return np.eye(3)
        perp = np.cross(a, [1, 0, 0])
        if np.linalg.norm(perp) < 1e-6:
            perp = np.cross(a, [0, 1, 0])
        return axis_angle(perp, 180.0)
    return axis_angle(v, math.degrees(math.atan2(np.linalg.norm(v), c)))


def frame_rot(u0, v0, u1, v1):
    """Rotation taking direction u0 -> u1 and (the plane of) v0 -> v1."""
    def basis(u, v):
        u = np.asarray(u, float); u = u / np.linalg.norm(u)
        v = np.asarray(v, float); v = v - u * np.dot(u, v); v = v / np.linalg.norm(v)
        return np.column_stack([u, v, np.cross(u, v)])
    return basis(u1, v1) @ basis(u0, v0).T


def M4(R=None, t=(0, 0, 0)):
    M = np.eye(4)
    if R is not None:
        M[:3, :3] = R
    M[:3, 3] = t
    return M


def quat(R):
    """3x3 rotation -> (x, y, z, w)."""
    m = R
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w = 0.25 * s; x = (m[2, 1] - m[1, 2]) / s; y = (m[0, 2] - m[2, 0]) / s; z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w = (m[2, 1] - m[1, 2]) / s; x = 0.25 * s; y = (m[0, 1] + m[1, 0]) / s; z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w = (m[0, 2] - m[2, 0]) / s; x = (m[0, 1] + m[1, 0]) / s; y = 0.25 * s; z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w = (m[1, 0] - m[0, 1]) / s; x = (m[0, 2] + m[2, 0]) / s; y = (m[1, 2] + m[2, 1]) / s; z = 0.25 * s
    q = np.array([x, y, z, w])
    return q / np.linalg.norm(q)


def qmat(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def slerp(q0, q1, t):
    q0 = np.asarray(q0, float); q1 = np.asarray(q1, float)
    d = float(np.dot(q0, q1))
    if d < 0:
        q1 = -q1; d = -d
    if d > 0.9995:
        q = q0 + (q1 - q0) * t
        return q / np.linalg.norm(q)
    th = math.acos(d)
    return (math.sin((1 - t) * th) * q0 + math.sin(t * th) * q1) / math.sin(th)


# ---------------------------------------------------------------------------------- checks

def volume(shells):
    """Enclosed volume (m^3) of closed outward shells (signed-tetrahedron sum, triangles fanned)."""
    total = 0.0
    for sh in shells:
        V = np.asarray(sh.verts, float)
        for f in sh.faces:
            a = V[f[0]]
            for i in range(1, len(f) - 1):
                total += float(np.dot(a, np.cross(V[f[i]], V[f[i + 1]]))) / 6.0
    return total


def _shells(model, groups=None):
    names = model.order if groups is None else [g for g in model.order if g in groups]
    return [sh for g in names for sh in model.parts[g].shells]


def fill_ratio(model, groups=None):
    """Solid volume / bounding-box volume of the chosen groups (all by default). A concept number -
    measured on one leg (examples/mech/concept_legs.py): skeletal ~0.05, light / faceted ~0.3,
    fortress ~0.45. Overlapping shells are counted twice, so treat it as a comparison between
    designs, not a physical volume."""
    shells = _shells(model, groups)
    if not shells:
        return 0.0
    pts = np.concatenate([np.asarray(s.verts, float) for s in shells])
    size = pts.max(0) - pts.min(0)
    box = float(np.prod(np.maximum(size, 1e-6)))
    return volume(shells) / box


def proportions(model, groups=None):
    """Outline numbers for comparing frames: size (x, y, z), width/height, depth/height, and the
    silhouette ratio front-width / side-depth (the light language wants < 0.8: slim from the front,
    long from the side; heavy frames sit near or above 1)."""
    shells = _shells(model, groups)
    pts = np.concatenate([np.asarray(s.verts, float) for s in shells])
    lo, hi = pts.min(0), pts.max(0)
    sx, sy, sz = (hi - lo)
    return {"size": [round(float(v), 3) for v in (sx, sy, sz)],
            "w_over_h": round(float(sx / sy), 3), "d_over_h": round(float(sz / sy), 3),
            "front_over_side": round(float(sx / sz), 3), "fill": round(fill_ratio(model, groups), 3)}


def limb_report(chains):
    """Per leg chain (dicts with hip / knee / ankle [/ knee2]): segment lengths, reach, rest bend
    (degrees from straight at each knee) and the thigh : shin ratio. Bends under ~10 degrees leave the
    IK no preferred direction - give every knee a visible rest bend toward its pole."""
    out = []
    for c in chains:
        pts = [np.asarray(c[k], float) for k in ("hip", "knee", "knee2", "ankle") if k in c]
        segs = [float(np.linalg.norm(b - a)) for a, b in zip(pts[:-1], pts[1:])]
        bends = []
        for a, b, cc in zip(pts[:-2], pts[1:-1], pts[2:]):
            d0, d1 = b - a, cc - b
            cosv = float(np.dot(d0, d1) / (np.linalg.norm(d0) * np.linalg.norm(d1)))
            bends.append(round(math.degrees(math.acos(max(-1.0, min(1.0, cosv)))), 1))
        out.append({"name": c.get("name", "?"), "segments": [round(s, 3) for s in segs], "reach": round(sum(segs), 3),
                    "bends": bends, "upper_over_lower": round(segs[0] / segs[-1], 3)})
    return out
