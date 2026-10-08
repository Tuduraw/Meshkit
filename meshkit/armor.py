"""Tracked armoured vehicles (main battle tanks, IFVs): hull side profiles, running gear, track paths,
faceted / cast turrets with bustles, and bolt-on modules.

Model space: origin = centre of the ground contact patch, +X left, +Y up, +Z forward.

Lessons from the modern pack
  * Hull fronts: a LONG, SHALLOW upper glacis (western MBTs about 8-12 deg from horizontal) over a short,
    steep lower glacis.  `glacis()` builds that from angles instead of guessed points.
  * Hull tops of western MBTs are not flat: shallow glacis -> flat turret-ring deck -> the engine deck
    steps up and rises towards the rear (`rear_deck`).  Raise the turret bustle floor so it clears it.
  * Turret shape carries most of the identity (needle-nose cast M60, bulged T-72, flat T-80, boxy welded
    T-90 / Leopard 2, wedge Type 10...).  Keep the overall width at the published value: integrate wedge
    armour into the turret outline rather than bolting plates that stick out.
  * Separate vehicles with real modules (sights, smoke dischargers, baskets, ERA, drums) - and put them on
    the surface with `turret_half_width` so nothing floats or hangs in the air.
  * Narrow upper hulls with fenders over the tracks are a T-54/55-era feature; T-72/80/90 and M60 have
    full-width upper hulls.
"""
import math
import numpy as np
from .geom import box, cylinder, prism, loft, extrude_zy, rot_x, rot_y


# ------------------------------------------------------------------------------- hull profile
def glacis(z_nose, nose_y, roof_y, upper_deg, lower_deg, belly_y, z_tail, tail_y=None):
    """Side profiles (z, y) of a hull with an upper and a lower front plate given by their angles from the
    horizontal.  Returns {"upper": poly, "lower": poly, "z_glacis_top": z, "z_lower_start": z}.
    The 'upper' polygon spans the full width above `nose_y`; the 'lower' one sits between the tracks."""
    zg = z_nose - (roof_y - nose_y) / math.tan(math.radians(upper_deg))
    zl = z_nose - (nose_y - belly_y) / math.tan(math.radians(lower_deg))
    ty = roof_y if tail_y is None else tail_y
    upper = [(z_tail, nose_y - 0.05), (z_nose - 0.02, nose_y - 0.05), (z_nose, nose_y), (zg, roof_y), (z_tail, ty)]
    lower = [(z_tail + 0.1, belly_y), (zl, belly_y), (z_nose, nose_y), (z_tail, nose_y)]
    return {"upper": upper, "lower": lower, "z_glacis_top": zg, "z_lower_start": zl}


def angle_of(p, q):
    """Plate angle (deg from horizontal) between two (z, y) profile points."""
    return math.degrees(math.atan2(abs(q[1] - p[1]), abs(q[0] - p[0])))


def hull(model, W, track_w, profile, upper_w=None, rear_deck=None, skirt=None, fender_y=None, part="hull", tag="hull"):
    """Lower hull between the tracks, upper hull (full width or `upper_w` half width with fenders over the
    tracks), optional stepped engine deck rear_deck=(z_front, z_rear, y_front, y_rear) and side skirts
    skirt=(z_front, z_rear, y_bottom)."""
    xin = W / 2 - track_w
    model.add(part, extrude_zy(profile["lower"], -xin - 0.02, xin + 0.02, tag=tag))
    uw = W / 2 - 0.02 if upper_w is None else upper_w
    model.add(part, extrude_zy(profile["upper"], -uw, uw, tag=tag))
    zs = [z for z, y in profile["upper"]]
    if upper_w is not None and fender_y is not None:
        for s in (1, -1):
            model.add(part, box(s * (uw + W / 2) / 2 - s * 0.02, fender_y, (max(zs) + min(zs)) / 2 + 0.1,
                                W / 2 - uw + 0.04, 0.06, max(zs) - min(zs) + 0.2, tag=tag))
    if rear_deck:
        zf, zr, yf, yr = rear_deck
        hw = uw - 0.05
        y0 = min(y for z, y in profile["upper"]) + 0.3
        bot = np.array([(hw, y0, zf), (-hw, y0, zf), (-hw, y0, zr), (hw, y0, zr)])
        top = np.array([(hw, yf, zf), (-hw, yf, zf), (-hw, yr, zr), (hw, yr, zr)])
        model.add(part, prism(bot, top, tag=tag))
    if skirt:
        zf, zr, yb = skirt
        ytop = min(y for z, y in profile["upper"]) + 0.05
        for s in (1, -1):
            model.add(part, box(s * (W / 2 - 0.02), (yb + ytop) / 2, (zf + zr) / 2, 0.06, ytop - yb, zf - zr, tag=tag))


# ------------------------------------------------------------------------------- running gear
def track_path(wheels_z, wheel_r, sprocket, idler, top_y, sprocket_front=False, t=0.04):
    """Closed (y, z) loop around road wheels / sprocket / idler: front-bottom, front-top, rear-top, rear-bottom
    (the order tudursvehiclemod's crawler_tracks expects)."""
    zf, zr = max(wheels_z), min(wheels_z)
    front, rear = (idler, sprocket) if not sprocket_front else (sprocket, idler)
    (fz, fy, fr), (rz, ry, rr) = front, rear
    return [(t, zf + wheel_r * 0.35), (fy - fr * 0.4, fz + fr + t), (fy + fr * 0.5, fz + fr * 0.85 + t),
            (top_y + t, fz - fr * 0.3), (top_y + t, rz + rr * 0.3), (ry + rr * 0.5, rz - rr * 0.85 - t),
            (ry - rr * 0.4, rz - rr - t), (t, zr - wheel_r * 0.35)]


def path_length(path):
    return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:] + path[:1]))


def link_spacing(path, link_len=0.15):
    L = path_length(path)
    return L / round(L / link_len)


def lay_links(path, x, width, spacing, thick=0.07, tag="track"):
    """Track links placed along the path at side x (for previews and contact checks - an engine that draws
    tracks itself only needs one link)."""
    cum = [0.0]
    for a, b in zip(path, path[1:] + path[:1]):
        cum.append(cum[-1] + math.hypot(b[0] - a[0], b[1] - a[1]))
    out = []
    for i in range(int(round(cum[-1] / spacing))):
        d = i * spacing
        k = max(j for j in range(len(path)) if cum[j] <= d)
        a, b = path[k], path[(k + 1) % len(path)]
        f = (d - cum[k]) / max(cum[k + 1] - cum[k], 1e-9)
        y, z = a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f
        ang = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])) + 90.0
        sh = box(x, 0, 0, width, thick, spacing * 0.9, tag=tag)
        sh.rotate(rot_x(ang), pivot=(x, 0, 0)); sh.translate(0, y, z)
        out.append(sh)
    return out


def running_gear(model, W, track_w, wheel_r, wheels_z, sprocket, idler, rollers_z=(), top_y=0.95, belly_y=0.47,
                 part="hull", prefix="$"):
    """Road wheels, sprocket, idler and return rollers as separate rotating parts on both sides, with axle
    stubs (and short arms when a wheel hangs below the belly) in `part`.  Returns [(name, pivot)]."""
    xin = W / 2 - track_w; xc = W / 2 - track_w / 2
    items = [("rw", z, wheel_r, wheel_r) for z in wheels_z]
    items += [("sp", sprocket[0], sprocket[1], sprocket[2]), ("id", idler[0], idler[1], idler[2])]
    items += [("rr", z, top_y - 0.1, 0.1) for z in rollers_z]
    out = []
    for i, (kind, z, y, r) in enumerate(items):
        for s, sd in ((1, "l"), (-1, "r")):
            name = f"{prefix}{kind}{i}_{sd}"
            n = 12 if r > 0.2 else 8
            model.add(name, cylinder((s * (xin + 0.03), y, z), (s * (W / 2 - 0.04), y, z), r, n=n,
                                     tag="wheel" if kind != "sp" else "sprocket"))
            model.pivots[name] = (s * xc, y, z)
            out.append((name, (s * xc, y, z)))
            model.add(part, cylinder((s * (xin - 0.05), y, z), (s * (xin + 0.08), y, z), min(0.08, r * 0.6), n=6, tag="hull"))
            if y < belly_y + 0.05:
                model.add(part, box(s * (xin - 0.06), (y + belly_y + 0.12) / 2, z, 0.1, belly_y + 0.12 - y, 0.12, tag="hull"))
    return out


# ------------------------------------------------------------------------------- turrets
def ring_xz(y, cz, hw, front, rear, n=16, p=2.5, pf=None):
    """Horizontal superellipse ring; pf = exponent of the front half (< 2 gives a pointed nose)."""
    pts = []
    for k in range(n):
        a = 2 * math.pi * k / n
        c, s = math.cos(a), math.sin(a)
        pp = (pf or p) if s >= 0 else p
        x = hw * np.sign(c) * abs(c) ** (2 / pp)
        e = front if s >= 0 else rear
        pts.append((x, y, cz + e * np.sign(s) * abs(s) ** (2 / pp)))
    return np.array(pts)


def cast_turret(cz, y, rings, tag="turret"):
    """Cast / rounded turret.  rings = [(dy, half_width, front, rear, p[, p_front])] bottom -> top."""
    return loft([ring_xz(y + r[0], cz, r[1], r[2], r[3], p=r[4], pf=(r[5] if len(r) > 5 else None)) for r in rings], tag)


def _full(half, y, cz):
    pts = [(x, y, cz + z) for (x, z) in half] + [(-x, y, cz + z) for (x, z) in reversed(half)]
    out = []
    for p in pts:
        if not out or not np.allclose(out[-1], p):
            out.append(p)
    if np.allclose(out[0], out[-1]):
        out.pop()
    return np.array(out, float)


def split_half(half, zs):
    """Split a half outline (front -> rear, x >= 0) at z = zs into (front part, rear part)."""
    front, rear = [], []
    for a, b in zip(half, half[1:]):
        if a[1] >= zs:
            front.append(a)
        if (a[1] - zs) * (b[1] - zs) < 0:
            f = (zs - a[1]) / (b[1] - a[1])
            c = (a[0] + (b[0] - a[0]) * f, zs)
            front.append(c)
            if not rear:
                rear.append((0.0, zs))
            rear.append(c)
        if b[1] < zs and (not rear or rear[-1] != b):
            rear.append(b)
    if front and front[-1][0] > 0.01:
        front.append((0.0, zs))
    if rear and rear[0][0] > 0.01:
        rear.insert(0, (0.0, zs))
    return front, rear


def faceted_turret(cz, y, h, plan, top, bustle=None, tag="turret"):
    """Welded / composite turret from a base half outline `plan` and a roof half outline `top`
    ([(x, z_rel)] front -> rear, same count).  bustle=(z_split_rel, floor_rise): the part behind the
    split gets its floor raised (clears a stepped engine deck when the turret turns).  Returns shells."""
    if not bustle:
        return [prism(_full(plan, y - 0.05, cz), _full(top, y + h, cz), tag=tag)]
    zs, dy = bustle
    pf, pr = split_half(plan, zs); tf, tr = split_half(top, zs)
    out = [prism(_full(pf, y - 0.05, cz), _full(tf, y + h, cz), tag=tag)]
    frac = (dy + 0.05) / (h + 0.05)
    pr2 = [(a[0] + (b[0] - a[0]) * frac, a[1] + (b[1] - a[1]) * frac) for a, b in zip(pr, tr)] if len(pr) == len(tr) else pr
    out.append(prism(_full(pr2, y + dy, cz), _full(tr, y + h - 0.002, cz), tag=tag))
    return out


def turret_half_width(turret, z_rel, frac=0.0):
    """Half width of a turret description at z (relative to the ring) and height fraction frac
    (0 base, 1 roof).  turret = {"plan", "top"} or {"rings"} as passed to the builders above."""
    def at(poly, z):
        pts = sorted([p for p in poly if p[0] > 0.01], key=lambda p: p[1])
        return float(np.interp(z, [p[1] for p in pts], [p[0] for p in pts]))
    if "plan" in turret:
        a = at(turret["plan"], z_rel); b = at(turret["top"], z_rel)
        return a + (b - a) * frac
    rings = turret["rings"]; dys = [r[0] for r in rings]; yq = frac * dys[-1]
    k = int(np.clip(np.searchsorted(dys, yq) - 1, 0, len(rings) - 2))
    f = (yq - dys[k]) / max(dys[k + 1] - dys[k], 1e-6)
    lerp = lambda i: rings[k][i] + (rings[k + 1][i] - rings[k][i]) * f
    e = lerp(2) if z_rel >= 0 else lerp(3)
    q = min(abs(z_rel) / max(e, 1e-6), 1.0)
    return lerp(1) * (1 - q ** 2.4) ** (1 / 2.4)


def gun(trunnion, length, r, mantlet=(0.6, 0.45, 1.2), evac=None, sleeve=0.0, tag="gun", mantlet_tag="turret"):
    """Gun with mantlet; trunnion=(x, y, z).  evac=(z_from_trunnion, len, r) fume extractor.
    Returns (shells, muzzle point)."""
    x, y, z = trunnion
    mw, mh, md = mantlet
    out = [box(x, y, z + md / 2 - 0.1, mw, mh, md, tag=mantlet_tag)]
    muzzle = z + md - 0.1 + length
    out.append(cylinder((x, y, z + md - 0.15), (x, y, muzzle), r, r * 0.9, n=10, tag=tag))
    if evac:
        ez, el, er = evac
        out.append(cylinder((x, y, z + ez), (x, y, z + ez + el), er, n=10, tag=tag))
    if sleeve:
        out.append(cylinder((x, y, z + md - 0.15), (x, y, z + md + sleeve), r * 1.5, n=10, tag=tag))
    return out, (x, y, muzzle)


# ------------------------------------------------------------------------------- modules
def sight(x, y, z, w, h, d, face=1, tag="turret"):
    """Armoured sight head on a roof: box + dark window on the +Z (face=1) or -Z face."""
    return [box(x, y + h / 2 - 0.05, z, w, h + 0.1, d, tag=tag),
            box(x, y + h * 0.55, z + face * (d / 2 + 0.005), w * 0.7, h * 0.45, 0.03, tag="glass")]


def periscope(x, y, z, r, h, head=(0.35, 0.3, 0.4), tag="turret"):
    """Panoramic sight: column + head with a window."""
    hw, hh, hd = head
    return [cylinder((x, y - 0.05, z), (x, y + h, z), r, n=10, tag=tag),
            box(x, y + h + hh / 2 - 0.03, z, hw, hh, hd, tag=tag),
            box(x, y + h + hh * 0.55, z + hd / 2 + 0.005, hw * 0.7, hh * 0.45, 0.03, tag="glass")]


def smoke_bank(x, y, z, n=4, yaw=0.0, pitch=35.0, tag="gun"):
    """Grenade dischargers on a bracket, pointing forward-up; yaw turns the bank (deg)."""
    out = [box(0, 0.0, 0, 0.16, 0.12, 0.12 * n + 0.05, tag=tag)]
    for i in range(n):
        o = (i - (n - 1) / 2) * 0.12
        out.append(cylinder((0, 0.02, o), (0, 0.02 + 0.28 * math.sin(math.radians(pitch)), o + 0.28 * math.cos(math.radians(pitch))),
                            0.045, n=6, tag=tag))
    for sh in out:
        sh.rotate(rot_y(-yaw)); sh.translate(x, y, z)
    return out


def basket(cx, y, z0, z1, w, h=0.45, tag="rack", kit_tag="kit"):
    """Open stowage basket (floor, rails, posts) with some kit inside."""
    L = z1 - z0; zc = (z0 + z1) / 2
    out = [box(cx, y + 0.03, zc, w, 0.06, L, tag=tag)]
    for (dx, dz, ww, ll) in ((w / 2, 0, 0.05, L), (-w / 2, 0, 0.05, L), (0, -L / 2, w, 0.05)):
        out.append(box(cx + dx, y + h, zc + dz, ww, 0.05, ll, tag=tag))
    for (dx, dz) in ((w / 2, L / 2), (-w / 2, L / 2), (w / 2, -L / 2), (-w / 2, -L / 2), (w / 2, 0), (-w / 2, 0)):
        out.append(box(cx + dx, y + h / 2, zc + dz, 0.05, h, 0.05, tag=tag))
    out.append(box(cx + w * 0.15, y + h * 0.35, zc + L * 0.15, w * 0.45, h * 0.6, L * 0.4, tag=kit_tag))
    out.append(cylinder((cx - w * 0.25, y + 0.2, z0 + 0.15), (cx - w * 0.25, y + 0.2, z1 - 0.15), 0.17, n=8, tag=kit_tag))
    return out


def drums(z, y, xs, r=0.3, L=0.9, tag="kit", rack_tag="hull"):
    """Horizontal fuel drums across the hull rear on a rack."""
    out = [cylinder((x - L / 2, y + r, z), (x + L / 2, y + r, z), r, n=10, tag=tag) for x in xs]
    out.append(box(sum(xs) / len(xs), y + 0.05, z, abs(xs[-1] - xs[0]) + L, 0.12, 0.3, tag=rack_tag))
    return out


def searchlight(x, y, z, r=0.18, L=0.35, tag="turret"):
    return [cylinder((x, y, z - L / 2), (x, y, z + L / 2), r, n=10, tag=tag),
            cylinder((x, y, z + L / 2 - 0.02), (x, y, z + L / 2 + 0.01), r * 0.85, n=10, tag="glass")]


def rws(x, y, z, scale=1.0, tag="turret"):
    """Remote weapon station: base ring, cradle, sensor box with window."""
    s = scale
    return [cylinder((x, y - 0.05, z), (x, y + 0.18 * s, z), 0.32 * s, n=10, tag=tag),
            box(x, y + 0.38 * s, z, 0.5 * s, 0.4 * s, 0.6 * s, tag=tag),
            box(x + 0.32 * s, y + 0.42 * s, z + 0.12 * s, 0.18 * s, 0.26 * s, 0.3 * s, tag=tag),
            box(x + 0.32 * s, y + 0.44 * s, z + 0.28 * s, 0.13 * s, 0.13 * s, 0.03, tag="glass")]


def bricks_on_plate(p0, p1, x_half, cols, rows, depth=0.12, f0=0.08, f1=0.92, tag="era"):
    """ERA bricks / armour tiles laid on a sloped plate between two (z, y) profile points (e.g. the upper
    glacis), spanning x = -x_half..x_half.  Returns shells."""
    (za, ya), (zb, yb) = p0, p1
    L = math.hypot(za - zb, yb - ya)
    ang = math.degrees(math.atan2(yb - ya, za - zb))
    nz, ny = math.sin(math.radians(ang)), math.cos(math.radians(ang))
    w = 2 * x_half / cols; l = L * (f1 - f0) / rows
    out = []
    for i in range(cols):
        for j in range(rows):
            f = f0 + (f1 - f0) * (j + 0.5) / rows
            z = za + (zb - za) * f + nz * depth * 0.3; y = ya + (yb - ya) * f + ny * depth * 0.3
            b = box(0, 0, 0, w * 0.92, depth, l * 0.9, tag=tag)
            b.rotate(rot_x(ang)); b.translate(-x_half + w * (i + 0.5), y, z)
            out.append(b)
    return out


def bricks_grid(x0, x1, y0, y1, z, cols, rows, tilt=0.0, yaw=0.0, depth=0.12, tag="era", pivot=None):
    """ERA brick grid on a vertical plane at z, then tilted (about X) and turned (about Y) round pivot."""
    w = (x1 - x0) / cols; h = (y1 - y0) / rows
    pv = pivot or ((x0 + x1) / 2, (y0 + y1) / 2, z)
    out = []
    for i in range(cols):
        for j in range(rows):
            sh = box(x0 + w * (i + 0.5), y0 + h * (j + 0.5), z, w * 0.92, h * 0.9, depth, tag=tag)
            if tilt:
                sh.rotate(rot_x(tilt), pivot=pv)
            if yaw:
                sh.rotate(rot_y(yaw), pivot=pv)
            out.append(sh)
    return out
