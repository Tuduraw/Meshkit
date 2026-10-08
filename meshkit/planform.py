"""Straight-edged lifting surfaces for jet aircraft (wings, tailplanes, canards, fins).

Lessons from the modern pack: jet planforms are trapezoids with straight leading / trailing edges and
clipped tips.  The WW2-style helpers (rounded tips, extra root stations) made modern wings look wrong.
Derive the chords from published data (area, span, sweep, taper) instead of guessing them, and keep a
crank only where the real aircraft has one.

Coordinates: +X left, +Y up, +Z forward.  A station is a dict {x, y, z (leading edge), c (chord), t}
as used by `geom.wing_section`; chords run from the leading edge backwards (-Z).
"""
import math
import numpy as np
from .geom import wing_section, loft, rot_z


# ---------------------------------------------------------------------------------- published data
def chords_from_area(area, span, taper):
    """Root (centre-line) and tip chord of a trapezoid wing from the reference area, total span and taper
    ratio (tip / root).  area = span * (c_root + c_tip) / 2."""
    c_root = 2.0 * area / (span * (1.0 + taper))
    return c_root, c_root * taper


def taper_from_area(area, span, c_tip):
    """Taper ratio when the tip chord is known (from a drawing) together with area and span."""
    c_root = 2.0 * area / span - c_tip
    return c_tip / c_root


def te_sweep(le_sweep, c_root, c_tip, half_span):
    """Trailing-edge sweep (deg, + = swept back) of a trapezoid."""
    dz = half_span * math.tan(math.radians(le_sweep)) + c_tip - c_root
    return math.degrees(math.atan2(dz, half_span))


def mac(c_root, c_tip):
    """Mean aerodynamic chord and its spanwise position as a fraction of the half span."""
    lam = c_tip / c_root
    m = 2.0 / 3.0 * c_root * (1 + lam + lam * lam) / (1 + lam)
    y = (1 + 2 * lam) / (3 * (1 + lam))
    return m, y


# ---------------------------------------------------------------------------------- stations
def trapezoid(half_span, c_root, c_tip, le_sweep, z_le_root, y=0.0, dihedral=0.0, x_root=0.0,
              t_root=0.05, t_tip=0.04, crank=None):
    """Stations of one half of a trapezoid surface defined on the centre line (x = 0).

    x_root   : first exposed station (fuselage side / boom).  The chord there is interpolated on the
               straight edges, so the planform stays a true trapezoid.
    crank    : optional (x, le_sweep_outboard, dihedral_outboard) for real cranked wings only
               (F-4 outer panels, A-10 outer panels...)."""
    def at(x, sweep, z0, c0, x0, cslope, y0, dih):
        return {"x": x, "y": y0 + (x - x0) * math.tan(math.radians(dih)),
                "z": z0 - (x - x0) * math.tan(math.radians(sweep)), "c": c0 + cslope * (x - x0)}
    cs = (c_tip - c_root) / half_span
    sts = [at(x_root, le_sweep, z_le_root, c_root, 0.0, cs, y, dihedral)]
    if crank:
        xc, sw2, dih2 = crank
        k = at(xc, le_sweep, z_le_root, c_root, 0.0, cs, y, dihedral)
        sts.append(k)
        tip = at(half_span, sw2, k["z"], k["c"], xc, cs, k["y"], dih2)
        sts.append(tip)
    else:
        sts.append(at(half_span, le_sweep, z_le_root, c_root, 0.0, cs, y, dihedral))
    n = len(sts)
    for i, s in enumerate(sts):
        f = i / (n - 1)
        s["t"] = t_root + (t_tip - t_root) * f
    return sts


def from_published(area, span, le_sweep, taper, z_le_root, y=0.0, dihedral=0.0, x_root=0.0, t_root=0.05, t_tip=0.04):
    """Trapezoid half-surface straight from published area / span / LE sweep / taper."""
    cr, ct = chords_from_area(area, span, taper)
    return trapezoid(span / 2.0, cr, ct, le_sweep, z_le_root, y, dihedral, x_root, t_root, t_tip)


def station_at(stations, x):
    """Interpolated station at span position |x| (for pylons, markings, fold lines)."""
    xs = [abs(s["x"]) for s in stations]
    x = abs(x)
    for a, b, xa, xb in zip(stations, stations[1:], xs, xs[1:]):
        if xa <= x <= xb:
            f = (x - xa) / max(xb - xa, 1e-9)
            return {k: a[k] + (b[k] - a[k]) * f for k in ("x", "y", "z", "c", "t")}
    return dict(stations[-1] if x > xs[-1] else stations[0])


def mirror(stations):
    return [dict(s, x=-s["x"]) for s in stations]


def split(stations, x_cut, overlap=0.04):
    """(inner, outer) station lists at |x| = x_cut - for folding or swinging outer panels.  The outer panel
    starts `overlap` inboard so the two pieces stay attached in every state."""
    inner = [s for s in stations if abs(s["x"]) < x_cut - 1e-6] + [station_at(stations, x_cut)]
    o0 = station_at(stations, x_cut - overlap)
    o0["x"] = math.copysign(abs(o0["x"]), stations[-1]["x"] or 1.0)
    outer = [o0] + [s for s in stations if abs(s["x"]) > x_cut + 1e-6]
    return inner, outer


# ---------------------------------------------------------------------------------- shells
def panel(stations, tag="skin", xs=None):
    """One closed panel through the given stations (clipped tip - the tip face is the flat aerofoil
    section, as on jets).  Works for either span direction."""
    rings = [wing_section(s["x"], s["y"], s["z"], s["c"], s.get("t", 0.05), 0.0, s.get("twist", 0.0), xs=xs)
             for s in stations]
    return loft(rings, tag)


def surface(half, tag="skin", xs=None):
    """Both halves.  half[0]['x'] == 0 -> one continuous shell across the centre line; > 0 -> two separate
    panels (boom-mounted tailplanes, wings on a wide body).  Returns a list of shells."""
    if abs(half[0]["x"]) < 1e-6:
        full = mirror(half[::-1]) + [dict(s) for s in half[1:]]
        full[len(half) - 1]["x"] = 0.0
        return [panel(full, tag, xs)]
    return [panel(half, tag, xs), panel(mirror(half), tag, xs)]


def fin(root, tip, x=0.0, cant=0.0, t=0.05, tag="skin", mirror_x=False):
    """Clipped vertical / canted fin.  root, tip = (y, z_le, chord).  cant = outward lean in degrees (twin
    fins).  mirror_x adds the opposite fin.  Returns a list of shells."""
    (ry, rz, rc), (ty, tz, tc) = root, tip
    rings = [wing_section(0.0, y, z, c, t, 0.0, 0.0, span_axis=1) for (y, z, c) in ((ry, rz, rc), (ty, tz, tc))]
    out = []
    for sx in ((1, -1) if mirror_x and x else (1,)):
        sh = loft([r.copy() for r in rings], tag)
        if cant:
            sh.rotate(rot_z(-cant * sx), pivot=(0, ry, 0))
        sh.translate(sx * x, 0, 0)
        out.append(sh)
    return out


def plate(outline_xz, y, t=0.06, tag="skin", mirror_x=True):
    """Flat strake / LERX / chine plate from a plan outline [(x, z)] (convex).  mirror_x adds the other side."""
    from .geom import extrude_xz
    out = [extrude_xz(list(outline_xz), y - t / 2, y + t / 2, tag=tag)]
    if mirror_x:
        out.append(extrude_xz([(-x, z) for (x, z) in reversed(outline_xz)], y - t / 2, y + t / 2, tag=tag))
    return out


# ---------------------------------------------------------------------------------- checks
def describe(half):
    """Planform numbers of a half-station list (for comparing with published data)."""
    a, b = half[0], half[-1]
    span = 2 * abs(b["x"])
    area = 0.0
    for p, q in zip(half, half[1:]):
        area += abs(q["x"] - p["x"]) * (p["c"] + q["c"]) / 2
    le = math.degrees(math.atan2(a["z"] - b["z"], abs(b["x"] - a["x"])))
    te = math.degrees(math.atan2((a["z"] - a["c"]) - (b["z"] - b["c"]), abs(b["x"] - a["x"])))
    return {"span": round(span, 3), "exposed_area": round(2 * area, 2), "le_sweep": round(le, 1), "te_sweep": round(te, 1),
            "c_root": round(a["c"], 3), "c_tip": round(b["c"], 3), "taper": round(b["c"] / max(a["c"], 1e-9), 3)}
