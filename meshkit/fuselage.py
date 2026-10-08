"""Fuselages, nacelles, canopies, cockpit recesses and engine exhausts for jet aircraft and helicopters.

A station is (z, cy, hw, ht, hb, p[, cx]): centre height cy, half width hw, height above / below the
centre ht / hb, superellipse exponent p (2 = ellipse, 3-4 = boxy / chined).  hw <= 0 is a pointed tip.

Lessons from the modern pack
  * Twin-engine aft bodies must be wide enough to contain both nozzles - otherwise the nozzles look
    "stuck on".  `cover_nozzles` widens the aft stations automatically; `check.penetration` finds tails
    that cut through nozzles.
  * Glass needs something behind it: `body_with_tub` drops the crown under the canopy into a simple
    cockpit recess (floor, seat, panel are separate boxes), so riders sit inside, not on top.
"""
import math
import numpy as np
from .geom import loft, ring_superellipse, cylinder, box, Shell


def body(stations, n=14, tag="skin", tags=None):
    """Lofted body along Z (stations in any z order)."""
    rings = []
    for s in stations:
        z, cy, hw, ht, hb, p = s[:6]
        cx = s[6] if len(s) > 6 else 0.0
        if hw <= 1e-6:
            rings.append(np.array([[cx, cy, z]]))
        else:
            rings.append(ring_superellipse(cx, cy, z, hw, ht, hb, n=n, p=p, fit=True))
    rings.sort(key=lambda r: r[0, 2])
    return loft(rings, tag, tags=tags)


def pod(cx, stations, n=10, tag="skin"):
    """Nacelle / boom / sponson offset sideways by cx (stations as for body)."""
    return body([tuple(s[:6]) + (cx,) for s in stations], n, tag)


def canopy(z_rear, z_front, y_base, y_top, hw, n=12, k_front=0.4, k_rear=0.35, tag="glass", p=2.2, cx=0.0):
    """Bubble from z_rear to z_front.  Put y_base a little below the skin so the bubble sinks in."""
    L = z_front - z_rear; h = y_top - y_base
    zs = [z_rear, z_rear + L * k_rear * 0.45, z_rear + L * k_rear, z_front - L * k_front, z_front - L * k_front * 0.45, z_front]
    sc = [0.35, 0.85, 1.0, 1.0, 0.8, 0.3]
    st = []
    for z, s in zip(zs, sc):
        if st and z <= st[-1][0] + 1e-3:
            continue
        st.append((z, y_base, hw * s, h * (0.55 + 0.45 * s), 0.02 + 0.02 * s, p, cx))
    return body(st, n=n, tag=tag)


# ------------------------------------------------------------------------------- cockpit recess
def _arc_y(x, hw, ht, p):
    f = min(abs(x) / max(hw, 1e-6), 1.0)
    return ht * (1.0 - f ** p) ** (1.0 / p)


def tub_ring(z, cy, hw, ht, hb, p, n, cw, floor, k, cx=0.0):
    """Body ring whose crown between x = +-cw drops to `floor` by blend k (0..1).  The vertex count does
    not depend on k, so rings with and without the notch loft together."""
    hw = max(hw, 0.02); ht = max(ht, 0.02); hb = max(hb, 0.02)
    cw = min(cw, hw * 0.9)
    tc = math.acos(min((cw / hw) ** (p / 2), 1.0))
    m = max(n - 4, 6)
    pts = []
    for i in range(m + 1):
        a = tc - (math.pi + 2 * tc) * i / m
        c, s = math.cos(a), math.sin(a)
        x = hw * np.sign(c) * abs(c) ** (2 / p)
        y = cy + (ht if s > 0 else hb) * np.sign(s) * abs(s) ** (2 / p)
        pts.append((x, y))
    for xx in (-0.97 * cw, 0.0, 0.97 * cw):
        ya = cy + _arc_y(xx, hw, ht, p)
        pts.append((xx, ya * (1 - k) + floor * k))
    pts = [(x + cx, y, z) for (x, y) in pts]
    return np.array(pts[:-1] if np.allclose(pts[0], pts[-1]) else pts)


def body_with_tub(stations, tubs, n=16, tag="skin", ramp=0.35):
    """Body with cockpit recesses.  tubs = [(z_front, z_rear, half_width, floor_y)] (z_front > z_rear).
    Returns one closed, triangulated shell (the ramps make some quads non-planar)."""
    S = sorted(stations, key=lambda t: t[0])
    zs = [t[0] for t in S]

    def at(z):
        return tuple(float(np.interp(z, zs, [t[i] for t in S])) for i in range(6))
    extra = []
    for (zf, zr, cw, fl) in tubs:
        for z in (zf + 0.04, zf - ramp, zr + ramp, zr - 0.04):
            if zs[0] < z < zs[-1]:
                extra.append(z)
    rings = []
    for z in sorted(set(round(z, 4) for z in zs + extra)):
        st = at(z)
        k, cw, fl = 0.0, 0.3, st[1]
        for (zf, zr, cw_, fl_) in tubs:
            if zr - 0.02 <= z <= zf + 0.02:
                kk = min(1.0, max(0.0, min(zf - z, z - zr) / ramp))
                if kk >= k:
                    k, cw, fl = kk, cw_, fl_
        if st[2] < 0.03:
            rings.append(np.array([(0.0, st[1], z)]))
        else:
            rings.append(tub_ring(z, st[1], st[2], st[3], st[4], st[5], n, cw, fl, k))
    return loft(rings, tag).triangulated()


def cockpit_fittings(z_seat, floor, y_rim, width, tag="interior"):
    """Seat (pan + back) and instrument panel for one crew station in a recess.  Returns shells."""
    return [box(0, floor + 0.5, z_seat - 0.32, 0.5, 1.0, 0.12, tag=tag),
            box(0, floor + 0.12, z_seat - 0.05, 0.5, 0.24, 0.5, tag=tag),
            box(0, (floor + y_rim + 0.1) / 2, z_seat + 0.8, width, y_rim + 0.1 - floor, 0.12, tag=tag)]


def rider_feet_y(floor, hip_above_feet=0.49):
    """Seat offset for engines that put the rider's feet at the seat point (tudursvehiclemod draws riders at
    0.7 scale: hips about 0.49 m above the feet).  Hips land on a seat pan 0.22 m above the floor."""
    return floor + 0.22 - hip_above_feet


# ------------------------------------------------------------------------------- engines
def cover_nozzles(stations, nozzles, margin=0.06, length=2.5, p=None):
    """Widen / deepen the aft stations so every nozzle (x, y, z_exit, r) lies inside the body over the
    last `length` metres before its exit.  Returns a new station list (adds a station at each nozzle exit)."""
    S = sorted([list(s) for s in stations], key=lambda t: t[0])
    zs = [s[0] for s in S]
    for (x, y, ze, r) in nozzles:
        if all(abs(z - ze) > 1e-3 for z in zs) and zs[0] < ze < zs[-1]:
            row = [ze] + [float(np.interp(ze, zs, [s[i] for s in S])) for i in range(1, 6)]
            S.append(row); S.sort(key=lambda t: t[0]); zs = [s[0] for s in S]
    for s in S:
        for (x, y, ze, r) in nozzles:
            if ze - 0.01 <= s[0] <= ze + length and s[2] > 1e-6:
                need_w = abs(x) + r + margin
                need_t = (y + r + margin) - s[1]
                need_b = s[1] - (y - r - margin)
                s[2] = max(s[2], need_w)
                s[3] = max(s[3], need_t)
                s[4] = max(s[4], need_b)
                if p is not None:
                    s[5] = p
    return [tuple(s) for s in S]


def nozzle(x, y, z_exit, r, length=0.8, tag="nozzle", inner_tag="exhaust", petals=12):
    """Exhaust nozzle ending at z_exit (pointing -Z) with a dark inner disc.  Returns shells."""
    return [cylinder((x, y, z_exit + length), (x, y, z_exit), r * 1.02, r * 0.92, n=petals, tag=tag),
            cylinder((x, y, z_exit + 0.02), (x, y, z_exit - 0.01), r * 0.8, n=petals - 2, tag=inner_tag)]


def intake_face(x, y, z_lip, w, h, depth=0.1, tag="intake"):
    """Dark plate just inside an intake lip (the body behind it is solid)."""
    return box(x, y, z_lip - depth / 2 + 0.03, w, h, depth, tag=tag)
