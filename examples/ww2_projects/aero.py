"""Small aircraft-building helpers on top of meshkit (examples only - not part of the library).

Native axes: +X left, +Y up, +Z forward, metres. Every helper returns closed meshkit shells.
"""
import math
import numpy as np
import meshkit as mk
from meshkit.geom import wing_section, Shell, loft


# ------------------------------------------------------------------------------------------ bodies
def body(stations, n=12, tag="skin", tags=None):
    """Lofted body along Z. stations: (z, cy, hw, ht, hb, p) - centre height cy, half width hw,
    height above / below the centre ht / hb, superellipse exponent p (2 = ellipse).
    hw <= 0 makes a pointed tip at (0, cy, z). Optional cx via a 7th value."""
    rings = []
    for s in stations:
        z, cy, hw, ht, hb, p = s[:6]
        cx = s[6] if len(s) > 6 else 0.0
        if hw <= 1e-6:
            rings.append(np.array([[cx, cy, z]]))
        else:
            rings.append(mk.ring_superellipse(cx, cy, z, hw, ht, hb, n=n, p=p, fit=True))
    rings.sort(key=lambda r: r[0, 2])
    return mk.loft(rings, tag, tags=tags)


def body_x(cx, stations, n=10, tag="skin"):
    """Same as body() but offset sideways (nacelles, pods): stations (z, cy, r_w, r_t, r_b, p)."""
    return body([tuple(s[:6]) + (cx,) for s in stations], n, tag)


# ------------------------------------------------------------------------------------------- wings
def _tip_ring(st, sign, k=0.25):
    dx = sign * _tip_dx(st, k)
    return wing_section(st["x"] + dx, st["y"], st["z"] - 0.08 * st["c"], st["c"] * 0.72, st.get("t", 0.1) * 0.55, 0.0,
                        st.get("twist", 0.0))


def _tip_dx(st, k=0.25):
    return min(k * st["c"], 0.3)


def _true_tip(stations):
    """Pull the last station in so that the rounded tip ends at the given x (the published span)."""
    st = [dict(s) for s in stations]
    last = st[-1]; sign = 1 if last["x"] >= st[0]["x"] else -1
    last["x"] -= sign * _tip_dx(last)
    return st


def wing_span(half, tag="skin", round_tips=True):
    """One closed wing across both sides. half: stations {x>=0, y, z (leading edge), c, t[, twist, camber]}
    from the root outwards; the left side (+X) is mirrored to the right (-X)."""
    half = _true_tip(half) if round_tips else half
    mirrored = [dict(s, x=-s["x"]) for s in reversed(half)]
    full = mirrored + (list(half[1:]) if abs(half[0]["x"]) < 1e-9 else list(half))
    rings = [wing_section(s["x"], s["y"], s["z"], s["c"], s.get("t", 0.12), s.get("camber", 0.0), s.get("twist", 0.0))
             for s in full]
    if round_tips:
        rings = [_tip_ring(full[0], -1)] + rings + [_tip_ring(full[-1], 1)]
    return mk.loft(rings, tag)


def wing_panel(stations, tag="skin", round_tip=True):
    """One panel spanning from stations[0] to stations[-1] along X (either direction), tip rounded."""
    if round_tip:
        stations = _true_tip(stations)
    rings = [wing_section(s["x"], s["y"], s["z"], s["c"], s.get("t", 0.12), s.get("camber", 0.0), s.get("twist", 0.0))
             for s in stations]
    if round_tip:
        sign = 1 if stations[-1]["x"] > stations[0]["x"] else -1
        rings.append(_tip_ring(stations[-1], sign))
    return mk.loft(rings, tag)


def fin(root, tip, tag="skin", x=0.0, t=0.1):
    """Vertical surface: root/tip = (y, z_le, chord). Thickness along X, centred on x."""
    rings = [wing_section(x, y, z, c, t, 0.0, 0.0, span_axis=1) for (y, z, c) in (root, tip)]
    # rounded top
    y, z, c = tip
    rings.append(wing_section(x, y + min(0.2 * c, 0.15), z - 0.1 * c, c * 0.7, t * 0.6, 0.0, 0.0, span_axis=1))
    return mk.loft(rings, tag)


def plate_panel(pts_xz, y, t, tag="skin"):
    """Thin flat plate given in plan view (x, z) - used for small flat surfaces."""
    return mk.extrude_xz(pts_xz, y - t / 2, y + t / 2, tag)


# ------------------------------------------------------------------------------------ propellers
def propeller(hub_c, axis, radius, blades=3, chord=0.25, hub_r=0.2, hub_len=0.5, spinner=True, phase=0.0,
              tag="prop", tip_tag="prop_tip", spinner_tag="spinner", twist=(30.0, 10.0)):
    """Spinner + blades. axis = thrust direction (unit). Blades start inside the spinner."""
    c = np.asarray(hub_c, float); a = np.asarray(axis, float); a /= np.linalg.norm(a)
    shells = []
    if spinner:
        prof = [(-hub_len * 0.5, hub_r * 0.9), (0.0, hub_r), (hub_len * 0.35, hub_r * 0.75), (hub_len * 0.5, 0.0)]
        shells.append(mk.lathe(c, a, prof, n=10, tag=spinner_tag))
    u, v, _ = mk.geom.frame_from_axis(a)
    for k in range(blades):
        ang = phase + 2 * math.pi * k / blades
        d = math.cos(ang) * u + math.sin(ang) * v
        root = c + d * hub_r * 0.55
        tip = c + d * radius
        shells.append(mk.blade(root, tip, chord, chord * 0.6, chord * 0.18, a, tag=tag, tip_tag=tip_tag,
                               twist_root=twist[0], twist_tip=twist[1], n_st=3))
    return shells


# ----------------------------------------------------------------------------------------- gear
def wheel(c, r, w, n=10, tag="tyre"):
    c = np.asarray(c, float)
    return mk.cylinder(c - [w / 2, 0, 0], c + [w / 2, 0, 0], r, n=n, tag=tag)


def strut(p0, p1, r=0.05, n=6, tag="gear"):
    return mk.cylinder(p0, p1, r, n=n, tag=tag)


def leg(top, axle, wheel_r, wheel_w, strut_r=0.06, tag="gear", side=0.0):
    """Gear leg from `top` (inside the structure) to the axle, wheel centred on the axle
    (side = sideways offset of the wheel from the strut, e.g. half a tyre width for single-sided forks)."""
    axle = np.asarray(axle, float)
    out = [strut(top, axle + [0, wheel_r * 0.15, 0], strut_r, tag=tag),
           mk.cylinder(axle - [abs(side) + wheel_w * 0.3, 0, 0], axle + [abs(side) + wheel_w * 0.3, 0, 0], strut_r * 0.8, n=6, tag=tag)]
    out.append(wheel(axle + [side, 0, 0], wheel_r, wheel_w))
    return out


# -------------------------------------------------------------------------------------- canopy
def canopy(z0, z1, y_base, y_top, hw, n=10, k_front=0.35, k_rear=0.35, tag="glass", p=2.2, cx=0.0):
    """Bubble from z0 (rear) to z1 (front); the base sits y_base (put it below the skin so it sinks in)."""
    L = z1 - z0; h = y_top - y_base
    zs = [z0, z0 + L * k_rear * 0.45, z0 + L * k_rear, z1 - L * k_front, z1 - L * k_front * 0.45, z1]
    sc = [0.35, 0.85, 1.0, 1.0, 0.8, 0.3]
    st = []
    for z, s in zip(zs, sc):
        if st and z <= st[-1][0] + 1e-3:        # k_front + k_rear >= 1: merge the coinciding stations
            continue
        st.append((z, y_base, hw * s, h * (0.55 + 0.45 * s), 0.02 + 0.02 * s, p, cx))
    return body(st, n=n, tag=tag)


# ----------------------------------------------------------------------------------- painting
def disc(X, Z, cx, cz, r):
    return (X - cx) ** 2 + (Z - cz) ** 2 <= r * r


def hinomaru(X, Z, cx, cz, r, white=0.0):
    """Masks (red, white ring) for a hinomaru centred at (cx, cz) in the (X, Z) or (Z, Y) plane."""
    red = disc(X, Z, cx, cz, r)
    ring = disc(X, Z, cx, cz, r + white) & ~red if white > 0 else np.zeros_like(red)
    return red, ring


def us_star(A, B, ca, cb, r, bars=True, rot=90.0):
    """Masks (blue, white) for the 1943-47 US insignia (blue disc, white star, white bars with blue outline).
    A/B = the two in-plane coordinates of the painted surface; the bars run along A."""
    from meshkit.texture import star_polygon, point_in_poly
    blue = disc(A, B, ca, cb, r)
    star = point_in_poly(A, B, star_polygon(ca, cb, r * 0.97, rot=rot))
    white = star.copy()
    if bars:
        w = r * 2.0; h = r * 0.5
        outer = (np.abs(B - cb) <= h / 2 + r * 0.08) & (np.abs(A - ca) <= r + w + r * 0.08)
        inner = (np.abs(B - cb) <= h / 2) & (np.abs(A - ca) <= r + w) & (np.abs(A - ca) >= r * 0.6)
        blue = blue | outer
        white = white | (inner & ~disc(A, B, ca, cb, r * 0.98)) | star
    return blue & ~white, white


def paint_masks(atlas, tags, kinds, fn):
    """fn(X, Y, Z) -> list of (mask, rgb); applied in order on the 3D point of every texel."""
    def f(X, Y, Z, cur):
        out = cur.copy()
        any_m = np.zeros(len(X), bool)
        for m, rgb in fn(X, Y, Z):
            out[m] = np.asarray(rgb, np.float32); any_m |= m
        return out, any_m
    atlas.paint3d(f, tags=tags, kinds=kinds)


def panel_lines(atlas, tags, zs=(), xs=(), w=0.012, k=0.82, kinds=("top", "bottom", "side")):
    """Darken thin bands at given Z (frames) and |X| (spanwise joints) positions."""
    def f(X, Y, Z, cur):
        m = np.zeros(len(X), bool)
        for z in zs:
            m |= np.abs(Z - z) < w
        for x in xs:
            m |= np.abs(np.abs(X) - x) < w
        return cur * k, m
    atlas.paint3d(f, tags=tags, kinds=kinds)
