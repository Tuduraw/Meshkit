"""Closed-mesh geometry toolkit (from ww2gen, made standalone).

Native coordinate convention: +X = left, +Y = up, +Z = forward, 1 unit = 1 m.
This is the same as glTF and as Minecraft / tudursvehiclemod model space; other
conventions (Blender Z-up etc.) are converted at import / export time (see io_obj, io_gltf).

Every primitive returns a Shell: a CLOSED, consistently wound, outward-facing
triangle/quad/ngon mesh (orientation is auto-corrected from the signed volume).
A part is a list of shells; shells may intersect each other (that is how parts are
joined without holes), but every shell on its own is watertight, so the model never
shows a back face through a gap (the mod renders with back-face culling).

Each face carries a tag (string) that the texture stage uses to decide how it is
UV-mapped and coloured.
"""
import math
import numpy as np

EPS = 1e-9


def V(x, y, z):
    return np.array([x, y, z], dtype=float)


# ---------------------------------------------------------------------------
# rotations
# ---------------------------------------------------------------------------
def rot_x(deg):
    a = math.radians(deg); c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], float)


def rot_y(deg):
    a = math.radians(deg); c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], float)


def rot_z(deg):
    a = math.radians(deg); c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float)


def rot_axis(axis, deg):
    """Right-handed rotation about an arbitrary axis (same convention as JOML fromAxisAngleDeg)."""
    ax = np.asarray(axis, float); ax = ax / np.linalg.norm(ax)
    a = math.radians(deg); c, s = math.cos(a), math.sin(a); x, y, z = ax
    C = 1 - c
    return np.array([[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                     [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                     [z * x * C - y * s, z * y * C + x * s, c + z * z * C]], float)


def frame_from_axis(d):
    """Orthonormal (u, v, w) with w along d."""
    w = np.asarray(d, float); w = w / np.linalg.norm(w)
    ref = np.array([0.0, 1.0, 0.0]) if abs(w[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(ref, w); u /= np.linalg.norm(u)
    v = np.cross(w, u)
    return u, v, w


# ---------------------------------------------------------------------------
# Shell
# ---------------------------------------------------------------------------
class Shell:
    """One closed mesh component."""

    def __init__(self, verts=None, faces=None, tags=None, name=""):
        self.verts = [] if verts is None else [np.asarray(v, float) for v in verts]
        self.faces = [] if faces is None else [tuple(f) for f in faces]
        self.tags = [] if tags is None else list(tags)
        self.name = name

    # --- construction helpers ---
    def add_vert(self, p):
        self.verts.append(np.asarray(p, float))
        return len(self.verts) - 1

    def add_face(self, idx, tag):
        idx = tuple(int(i) for i in idx)
        # drop consecutive duplicates (collapsed corners)
        clean = []
        for i in idx:
            if not clean or clean[-1] != i:
                clean.append(i)
        if len(clean) > 1 and clean[0] == clean[-1]:
            clean.pop()
        if len(clean) >= 3:
            self.faces.append(tuple(clean)); self.tags.append(tag)

    # --- queries ---
    def V(self):
        return np.array(self.verts)

    def signed_volume(self):
        P = self.V(); vol = 0.0
        for f in self.faces:
            p0 = P[f[0]]
            for k in range(1, len(f) - 1):
                vol += np.dot(p0, np.cross(P[f[k]], P[f[k + 1]]))
        return vol / 6.0

    def reverse(self):
        self.faces = [tuple(reversed(f)) for f in self.faces]
        return self

    def orient_outward(self):
        if self.signed_volume() < 0:
            self.reverse()
        return self

    def bbox(self):
        P = self.V(); return P.min(0), P.max(0)

    def tri_count(self):
        return sum(len(f) - 2 for f in self.faces)

    # --- transforms (return self) ---
    def apply(self, M=None, t=(0, 0, 0), pivot=(0, 0, 0)):
        piv = np.asarray(pivot, float); tt = np.asarray(t, float)
        out = []
        for p in self.verts:
            q = p - piv
            if M is not None:
                q = M @ q
            out.append(q + piv + tt)
        self.verts = out
        if M is not None and np.linalg.det(M) < 0:
            self.reverse()
        return self

    def translate(self, x, y=0.0, z=0.0):
        return self.apply(None, (x, y, z))

    def rotate(self, M, pivot=(0, 0, 0)):
        return self.apply(M, (0, 0, 0), pivot)

    def mirrored_x(self, tag_map=None):
        s = self.copy()
        s.verts = [np.array([-p[0], p[1], p[2]]) for p in s.verts]
        s.reverse()
        if tag_map:
            s.tags = [tag_map.get(t, t) for t in s.tags]
        return s

    def copy(self):
        return Shell([p.copy() for p in self.verts], list(self.faces), list(self.tags), self.name)

    def triangulated(self):
        """Fan-split every polygon into triangles (for lofts whose quads are not planar)."""
        faces = []; tags = []
        for f, t in zip(self.faces, self.tags):
            for k in range(1, len(f) - 1):
                faces.append((f[0], f[k], f[k + 1])); tags.append(t)
        self.faces = faces; self.tags = tags
        return self

    def retag(self, tag, only=None):
        self.tags = [tag if (only is None or t == only) else t for t in self.tags]
        return self


# ---------------------------------------------------------------------------
# polygon utilities
# ---------------------------------------------------------------------------
def _plane_basis(pts):
    c = pts.mean(0)
    n = np.zeros(3)
    m = len(pts)
    for i in range(m):  # Newell normal
        a = pts[i]; b = pts[(i + 1) % m]
        n += np.array([(a[1] - b[1]) * (a[2] + b[2]), (a[2] - b[2]) * (a[0] + b[0]), (a[0] - b[0]) * (a[1] + b[1])])
    ln = np.linalg.norm(n)
    if ln < EPS:
        return c, None, None, None
    n /= ln
    u, v, _ = frame_from_axis(n)
    return c, u, v, n


def polygon_is_convex(pts):
    pts = np.asarray(pts, float)
    c, u, v, n = _plane_basis(pts)
    if n is None:
        return False
    q = np.stack([(pts - c) @ u, (pts - c) @ v], 1)
    m = len(q); sign = 0
    for i in range(m):
        a, b, cc = q[i], q[(i + 1) % m], q[(i + 2) % m]
        cr = (b[0] - a[0]) * (cc[1] - b[1]) - (b[1] - a[1]) * (cc[0] - b[0])
        if abs(cr) < 1e-10:
            continue
        s = 1 if cr > 0 else -1
        if sign == 0:
            sign = s
        elif s != sign:
            return False
    # planarity
    d = np.abs((pts - c) @ n)
    return d.max() < 1e-3 * max(1.0, np.ptp(pts, 0).max())


def add_cap(sh, ring_idx, tag, reverse=False):
    """Close a ring of vertex indices. Convex planar rings -> one ngon; otherwise a
    centroid fan. reverse=True gives the winding for the START of a loft."""
    ids = list(ring_idx)
    if reverse:
        ids = ids[::-1]
    pts = np.array([sh.verts[i] for i in ids])
    if len(ids) == 3 or polygon_is_convex(pts):
        sh.add_face(ids, tag)
        return
    c = sh.add_vert(pts.mean(0))
    m = len(ids)
    for j in range(m):
        sh.add_face((c, ids[j], ids[(j + 1) % m]), tag)


# ---------------------------------------------------------------------------
# primitives
# ---------------------------------------------------------------------------
def loft(rings, tag="body", tags=None, cap_start=True, cap_end=True, cap_tag=None, name=""):
    """Skin a list of rings. A ring is an (n,3) array; a (1,3) array is a pointed tip.
    All non-tip rings must have the same n. tags: optional per-segment tag list
    (len(rings)-1). Result is closed (caps added where needed) and oriented outward."""
    sh = Shell(name=name)
    idx = []
    n = None
    for r in rings:
        r = np.asarray(r, float).reshape(-1, 3)
        if len(r) > 1:
            if n is None:
                n = len(r)
            assert len(r) == n, "all rings must share the point count"
        idx.append([sh.add_vert(p) for p in r])
    assert n is not None, "loft needs at least one real ring"
    for k in range(len(idx) - 1):
        a, b = idx[k], idx[k + 1]
        t = tag if tags is None else tags[k]
        if len(a) == 1 and len(b) == 1:
            raise ValueError("two consecutive tips")
        for j in range(n):
            j1 = (j + 1) % n
            if len(a) == 1:
                sh.add_face((a[0], b[j1], b[j]), t)
            elif len(b) == 1:
                sh.add_face((a[j], a[j1], b[0]), t)
            else:
                sh.add_face((a[j], a[j1], b[j1], b[j]), t)
    ct = cap_tag or (tag if tags is None else None)
    if len(idx[0]) > 1:
        if not cap_start:
            raise ValueError("open loft start - every shell must be closed")
        add_cap(sh, idx[0], ct or tags[0], reverse=True)
    if len(idx[-1]) > 1:
        if not cap_end:
            raise ValueError("open loft end - every shell must be closed")
        add_cap(sh, idx[-1], ct or tags[-1], reverse=False)
    return sh.orient_outward()


def ring_superellipse(cx, cy, z, hw, ht, hb=None, n=12, p=2.0, phase=0.0, tilt_x=0.0, fit=False):
    """Cross-section ring in the plane z=const (points ordered by angle).
    hw = half width (X), ht/hb = half height above/below centre. p=2 ellipse, >2 boxier.
    n should be a multiple of 4 for exact left/right/top/bottom points; with fit=True the
    upper / lower halves are stretched so the polygon's flat top / bottom still reach ht / hb."""
    hb = ht if hb is None else hb
    pts = []
    for k in range(n):
        th = 2 * math.pi * k / n + phase
        c, s = math.cos(th), math.sin(th)
        x = hw * math.copysign(abs(c) ** (2.0 / p), c)
        y = (ht if s >= 0 else hb) * math.copysign(abs(s) ** (2.0 / p), s)
        pts.append((cx + x, cy + y, z))
    P = np.array(pts)
    if fit:
        dy = P[:, 1] - cy
        up, dn = dy.max(), -dy.min()
        if up > 1e-9:
            dy[dy > 0] *= ht / up
        if dn > 1e-9:
            dy[dy < 0] *= hb / dn
        P[:, 1] = cy + dy
    if tilt_x:
        P = (rot_x(tilt_x) @ (P - [cx, cy, z]).T).T + [cx, cy, z]
    return P


def ring_on_axis(center, axis, radius, n=8, phase=0.0, ry=None):
    """Circle (or ellipse with ry) of n points around `axis` through `center`."""
    u, v, w = frame_from_axis(axis)
    ry = radius if ry is None else ry
    c = np.asarray(center, float)
    return np.array([c + radius * math.cos(2 * math.pi * k / n + phase) * u
                     + ry * math.sin(2 * math.pi * k / n + phase) * v for k in range(n)])


def box(cx, cy, cz, sx, sy, sz, tag="body", M=None, pivot=None, name=""):
    hx, hy, hz = sx / 2, sy / 2, sz / 2
    r0 = np.array([[cx - hx, cy - hy, cz - hz], [cx + hx, cy - hy, cz - hz], [cx + hx, cy + hy, cz - hz], [cx - hx, cy + hy, cz - hz]])
    r1 = r0 + [0, 0, sz]
    sh = loft([r0, r1], tag, name=name)
    if M is not None:
        sh.rotate(M, pivot if pivot is not None else (cx, cy, cz))
    return sh


def prism(bottom, top, tag="body", name=""):
    """Closed solid between two polygons with the same vertex count (any orientation)."""
    return loft([np.asarray(bottom, float), np.asarray(top, float)], tag, name=name)


def extrude_xz(poly_xz, y0, y1, tag="body"):
    """Vertical extrusion of a polygon given in plan view (x,z)."""
    b = [(x, y0, z) for x, z in poly_xz]; t = [(x, y1, z) for x, z in poly_xz]
    return prism(b, t, tag)


def extrude_zy(poly_zy, x0, x1, tag="body"):
    """Extrusion along X of a polygon given in side view (z,y) - fins, rudders, plates."""
    a = [(x0, y, z) for z, y in poly_zy]; b = [(x1, y, z) for z, y in poly_zy]
    return prism(a, b, tag)


def extrude_xy(poly_xy, z0, z1, tag="body"):
    """Extrusion along Z of a polygon given in front view (x,y)."""
    a = [(x, y, z0) for x, y in poly_xy]; b = [(x, y, z1) for x, y in poly_xy]
    return prism(a, b, tag)


def cylinder(p0, p1, r0, r1=None, n=8, tag="body", phase=None, name=""):
    r1 = r0 if r1 is None else r1
    p0 = np.asarray(p0, float); p1 = np.asarray(p1, float)
    ph = (math.pi / n) if phase is None else phase
    a = ring_on_axis(p0, p1 - p0, r0, n, ph); b = ring_on_axis(p1, p1 - p0, r1, n, ph)
    return loft([a, b], tag, name=name)


def lathe(axis_p0, axis_dir, profile, n=8, tag="body", tags=None, phase=None, name=""):
    """Revolve a profile [(t, r), ...] (t = distance along axis from p0, r = radius).
    r == 0 at an end makes a pointed/closed tip; otherwise the end is capped."""
    d = np.asarray(axis_dir, float); d = d / np.linalg.norm(d)
    p0 = np.asarray(axis_p0, float)
    ph = (math.pi / n) if phase is None else phase
    rings = []
    for t, r in profile:
        c = p0 + d * t
        if r <= 1e-6:
            rings.append(c.reshape(1, 3))
        else:
            rings.append(ring_on_axis(c, d, r, n, ph))
    return loft(rings, tag, tags=tags, name=name)


def ellipsoid(center, rx, ry, rz, nu=8, nv=6, tag="body"):
    """Ellipsoid lofted along Z (poles at +/-Z)."""
    cx, cy, cz = center
    rings = [np.array([[cx, cy, cz - rz]])]
    for i in range(1, nv):
        th = math.pi * i / nv
        z = cz - rz * math.cos(th); s = math.sin(th)
        rings.append(ring_superellipse(cx, cy, z, rx * s, ry * s, n=nu))
    rings.append(np.array([[cx, cy, cz + rz]]))
    return loft(rings, tag)


# ---------------------------------------------------------------------------
# aircraft helpers
# ---------------------------------------------------------------------------
def naca_thickness(x, t):
    return 5 * t * (0.2969 * math.sqrt(max(x, 0)) - 0.1260 * x - 0.3516 * x * x + 0.2843 * x ** 3 - 0.1036 * x ** 4)


def airfoil_2d(t=0.12, camber=0.0, xs=(0.0, 0.06, 0.25, 0.55, 1.0)):
    """Closed airfoil loop (u=chord fraction 0..1 from the LE, w = up), convex for
    small camber. Returns points LE -> upper -> TE -> lower (TE is a single point).
    Minimum TE thickness keeps faces non-degenerate."""
    upper = []; lower = []
    for x in xs:
        yt = naca_thickness(x, t)
        yc = camber * 4 * x * (1 - x)
        if x >= 1.0:
            yt = max(yt, 0.004)
        upper.append((x, yc + yt)); lower.append((x, yc - yt))
    le = (0.0, 0.0)
    pts = [le] + upper[1:] + lower[1:][::-1]
    # TE: merge last upper/lower into two close points (blunt TE) - keep both for convexity
    return pts


def wing_section(le_x, le_y, le_z, chord, t=0.12, camber=0.0, twist=0.0, xs=None, span_axis=0):
    """3D ring for a wing station whose chord runs from le_z (front) backwards (-Z).
    The ring lies in the plane x = le_x (span_axis=0) - for fins use span_axis=1."""
    pts2 = airfoil_2d(t, camber) if xs is None else airfoil_2d(t, camber, xs)
    out = []
    for u, w in pts2:
        dz = -u * chord; dw = w * chord
        # twist about the quarter chord, nose-up positive
        if twist:
            a = math.radians(twist)
            zc = -0.25 * chord
            rz = dz - zc
            dz, dw = zc + rz * math.cos(a) - dw * math.sin(a), rz * math.sin(a) + dw * math.cos(a)
        if span_axis == 0:
            out.append((le_x, le_y + dw, le_z + dz))
        else:  # vertical fin: thickness along X, chord along Z, span along Y
            out.append((le_x + dw, le_y, le_z + dz))
    return np.array(out)


def wing(stations, tag="body", tip_tag=None, round_tip=True):
    """stations: list of dicts {x, y, z (LE), c (chord), t, camber, twist}.
    Spans along X (either direction). The last station can be closed with a rounded
    tip by lofting to a shrunken ring."""
    rings = [wing_section(s['x'], s['y'], s['z'], s['c'], s.get('t', 0.12), s.get('camber', 0.0), s.get('twist', 0.0)) for s in stations]
    if round_tip and len(stations) >= 2:
        last = stations[-1]; prev = stations[-2]
        dx = math.copysign(min(0.25 * last['c'], 0.35), last['x'] - prev['x'])
        tip = wing_section(last['x'] + dx, last['y'], last['z'] - 0.1 * last['c'], last['c'] * 0.7,
                           last.get('t', 0.12) * 0.6, 0.0, last.get('twist', 0.0))
        rings.append(tip)
    tags = [tag] * (len(rings) - 1)
    if tip_tag:
        tags[-1] = tip_tag
    return loft(rings, tag, tags=tags)


def fin(stations, tag="body"):
    """Vertical surface: stations {x, y (root->tip), z (LE), c, t} spanning along +Y."""
    rings = [wing_section(s['x'], s['y'], s['z'], s['c'], s.get('t', 0.10), 0.0, 0.0, span_axis=1) for s in stations]
    return loft(rings, tag)


def blade(root, tip, chord_root, chord_tip, thick, axis_spin, tag="prop", tip_tag=None, twist_root=30.0, twist_tip=10.0, n_st=3):
    """Propeller blade from root to tip (both 3D), as lofted flat lenses.
    axis_spin = propeller axis (thrust direction); the blade chord lies in the
    plane of rotation, twisted towards the axis."""
    root = np.asarray(root, float); tip = np.asarray(tip, float)
    r = tip - root; L = np.linalg.norm(r); rd = r / L
    ax = np.asarray(axis_spin, float); ax = ax / np.linalg.norm(ax)
    tang = np.cross(ax, rd); tang /= np.linalg.norm(tang)
    rings = []
    for i in range(n_st):
        f = i / (n_st - 1)
        c = chord_root + (chord_tip - chord_root) * f
        c *= (1.0 - 0.35 * f * f) if i == n_st - 1 else 1.0
        tw = math.radians(twist_root + (twist_tip - twist_root) * f)
        cdir = tang * math.cos(tw) + ax * math.sin(tw)
        tdir = np.cross(rd, cdir); tdir /= np.linalg.norm(tdir)
        p = root + r * f
        th = thick * (1.0 - 0.5 * f)
        rings.append(np.array([p + cdir * c * 0.5, p + tdir * th * 0.5, p - cdir * c * 0.5, p - tdir * th * 0.5]))
    tags = [tag] * (n_st - 1)
    if tip_tag:
        tags[-1] = tip_tag
    return loft(rings, tag, tags=tags)


# ---------------------------------------------------------------------------
# Part / Model containers
# ---------------------------------------------------------------------------
class Part:
    def __init__(self, name):
        self.name = name
        self.shells = []
        self.meta = {}

    def add(self, *shells):
        for s in shells:
            if isinstance(s, (list, tuple)):
                self.add(*s)
            else:
                self.shells.append(s)
        return self

    def tri_count(self):
        return sum(s.tri_count() for s in self.shells)


class Model:
    """Named parts, each a list of shells.

    Optional data that the I/O and preview stages use:
      pivots  {part: (x,y,z)}   origin of the part (hinge / turret axis); exported as the object origin
      props   {part: {k: v}}    free metadata (glTF extras)
      refs    [(name, (x,y,z), kind)]  reference points (seats, muzzles, cameras) - drawn in previews
      uvs     {(part, shell, face): [(u,v), ...]}   per-corner UVs (set by Atlas.uvs() or by import)
      texture HxWx3/4 uint8 array or None
    """
    def __init__(self, name):
        self.name = name
        self.parts = {}
        self.order = []
        self.pivots = {}
        self.props = {}
        self.refs = []
        self.uvs = None
        self.texture = None

    def part(self, name):
        if name not in self.parts:
            self.parts[name] = Part(name); self.order.append(name)
        return self.parts[name]

    def add(self, part_name, *shells):
        self.part(part_name).add(*shells)
        return self

    def tri_count(self):
        return sum(p.tri_count() for p in self.parts.values())

    def all_points(self, parts=None):
        pts = []
        for n in self.order:
            if parts is not None and n not in parts:
                continue
            for s in self.parts[n].shells:
                pts.extend(s.verts)
        return np.array(pts)

    def bbox(self, parts=None):
        P = self.all_points(parts); return P.min(0), P.max(0)


    def remove(self, part_name):
        if part_name in self.parts:
            del self.parts[part_name]; self.order.remove(part_name)
        return self

    def face_count(self):
        return sum(len(s.faces) for p in self.parts.values() for s in p.shells)

    def summary(self):
        lo, hi = self.bbox()
        return {"name": self.name, "parts": len(self.order), "shells": sum(len(self.parts[p].shells) for p in self.order),
                "faces": self.face_count(), "tris": self.tri_count(),
                "bbox_min": [round(float(v), 4) for v in lo], "bbox_max": [round(float(v), 4) for v in hi],
                "size": [round(float(b - a), 4) for a, b in zip(lo, hi)],
                "per_part": {p: {"shells": len(self.parts[p].shells), "tris": self.parts[p].tri_count(),
                                 "pivot": [round(float(v), 4) for v in self.pivots[p]] if p in self.pivots else None}
                             for p in self.order}}

    def transformed(self, M=None, t=(0, 0, 0)):
        """Copy with every vertex mapped p -> M @ p + t (pivots and refs too); winding is fixed when det(M) < 0."""
        import copy
        out = copy.copy(self)
        out.parts = {}; out.order = list(self.order)
        for n in self.order:
            p = Part(n); p.meta = dict(self.parts[n].meta)
            p.shells = [s.copy().apply(M, t) for s in self.parts[n].shells]
            out.parts[n] = p
        f = (lambda q: tuple(((np.asarray(M) @ np.asarray(q, float)) if M is not None else np.asarray(q, float)) + np.asarray(t, float)))
        out.pivots = {k: f(v) for k, v in self.pivots.items()}
        out.refs = [(a, f(b), c) for a, b, c in self.refs]
        if M is not None and np.linalg.det(M) < 0 and self.uvs is not None:
            # faces were reversed: reverse the per-corner UV lists the same way
            out.uvs = {k: list(reversed(v)) for k, v in self.uvs.items()}
        return out
