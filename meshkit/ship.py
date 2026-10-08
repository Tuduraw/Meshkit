"""Modern warships: hulls with knuckle and flare, polygonal sloped deckhouses (stealthy / Aegis
superstructures), masts, radars, and deck fittings.

Model space: waterline y = 0, midships z = 0, +Z bow, +X port.  A hull station is
(z, deck_y, half_beam_at_deck, half_beam_at_waterline, draft).

Lessons from the modern pack
  * Character comes from the superstructure: polygonal plans with sloped walls (`house`), array faces
    placed on those walls (`wall_point`), bridges with outward-leaning windows and wings.
  * Below the waterline only a bilge and a keel point are needed - spend the triangles above it.
  * Targets: destroyers / cruisers 5,000-7,000 triangles, battleships / carriers 5,000-10,000.
  * Small items (rails, boats, launchers) must touch something - every helper here sinks a few cm in.
"""
import math
import numpy as np
from .geom import loft, prism, box, cylinder, ellipsoid, ring_superellipse, rot_y, rot_x


# ------------------------------------------------------------------------------- hull
def hull_rings(stations, n_out=40, flare=0.0):
    """Rings for `loft`: deck edge, flared topside, knuckle, waterline, bilge, keel (12 points)."""
    S = np.array(stations, float)
    zs = np.unique(np.concatenate([np.linspace(S[0, 0], S[-1, 0], n_out), S[:, 0]]))
    rings = []
    for z in zs:
        deck, hwd, hww, dr = (np.interp(z, S[:, 0], S[:, k]) for k in (1, 2, 3, 4))
        hwd = max(hwd, 0.03); hww = max(hww, 0.03)
        yk = deck * 0.55
        hwk = hww + (hwd - hww) * (0.35 - flare)
        pts = [(hwd, deck), (hwd * 0.98, deck * 0.85), (hwk, yk), (hww, 0.0), (hww * 0.9, -dr * 0.6), (hww * 0.3, -dr),
               (-hww * 0.3, -dr), (-hww * 0.9, -dr * 0.6), (-hww, 0.0), (-hwk, yk), (-hwd * 0.98, deck * 0.85), (-hwd, deck)]
        rings.append(np.array([(x, y, z) for (x, y) in pts]))
    return rings


def hull(stations, n_out=40, flare=0.0, tag="hull"):
    return loft(hull_rings(stations, n_out, flare), tag)


def deck_at(stations, z):
    """(deck height, half beam at the deck) at z."""
    S = np.array(stations, float)
    return float(np.interp(z, S[:, 0], S[:, 1])), float(np.interp(z, S[:, 0], S[:, 2]))


# ------------------------------------------------------------------------------- superstructure
def sym(half):
    """Half outline (x >= 0, bow -> stern) -> full convex loop."""
    out = list(half) + [(-x, z) for (x, z) in reversed(half)]
    clean = []
    for p in out:
        if clean and abs(clean[-1][0] - p[0]) < 1e-6 and abs(clean[-1][1] - p[1]) < 1e-6:
            continue
        clean.append(p)
    if abs(clean[0][0] - clean[-1][0]) < 1e-6 and abs(clean[0][1] - clean[-1][1]) < 1e-6:
        clean.pop()
    return clean


def house(plan, y0, y1, plan_top=None, inset=0.0, tag="super"):
    """Convex deckhouse.  plan / plan_top = [(x, z)].  inset shrinks the roof towards the centroid
    (sloped walls).  For a bridge whose windows lean OUT, pass a larger plan_top."""
    P = np.array(plan, float)
    if plan_top is None:
        c = P.mean(0); T = c + (P - c) * (1.0 - inset)
    else:
        T = np.array(plan_top, float)
    return prism(np.array([(x, y0, z) for x, z in P]), np.array([(x, y1, z) for x, z in T]), tag=tag)


def wall_point(bot_a, bot_b, top_a, top_b, y0, y1, y):
    """Centre (x, z) of a sloped wall at height y - for placing radar arrays flush on a wall.
    bot_a/bot_b = wall ends at y0, top_a/top_b = same ends at y1."""
    f = (y - y0) / (y1 - y0)
    return [((bot_a[i] + (top_a[i] - bot_a[i]) * f) + (bot_b[i] + (top_b[i] - bot_b[i]) * f)) / 2 for i in range(2)]


def array_face(c, normal, r=1.95, depth=0.25, tag="radar"):
    """Octagonal flat array (SPY-1 / SAMPSON faces...) centred at c facing `normal`."""
    c = np.asarray(c, float); n = np.asarray(normal, float); n /= np.linalg.norm(n)
    return cylinder(c - n * depth / 2, c + n * depth / 2, r, r * 0.9, n=8, tag=tag)


def dish(c, normal, r, depth=0.3, tag="radar"):
    c = np.asarray(c, float); n = np.asarray(normal, float); n /= np.linalg.norm(n)
    return cylinder(c - n * depth / 2, c + n * depth / 2, r, r * 0.85, n=10, tag=tag)


def radome(c, r, tag="dome"):
    return ellipsoid(c, r, r, r, nu=12, nv=7, tag=tag)


def bridge(plan_half, y0, y1, lean=0.45, roof=0.35, wings=(3.6, 0.3, 3.0), tag="super"):
    """Pilot house with outward-leaning windows, a roof slab and bridge wings.  plan_half = [(x, z)] half
    outline bow -> stern.  Returns shells; the window band is at y1 - 1.6 .. y1 - 0.6 for painting."""
    P = sym(plan_half)
    c = np.array(P).mean(0)
    top = [(c[0] + (x - c[0]) * 1.0 + np.sign(x) * lean * 0.6, z + (lean if z > c[1] else 0.0)) for x, z in P]
    out = [house(P, y0, y1, top, tag=tag)]
    big = [(x * 1.02, z) for (x, z) in top]
    out.append(house(big, y1 - 0.05, y1 + roof, tag=tag))
    if wings:
        L, t, d = wings
        hw = max(abs(x) for x, z in plan_half)
        zf = max(z for x, z in plan_half)
        for s in (1, -1):
            out.append(box(s * (hw + L / 2 - 0.3), y0 + 0.15, zf - d, L + 0.6, t, d, tag=tag))
            out.append(box(s * (hw + L - 0.3), y0 + 0.75, zf - d, 0.15, 1.0, d, tag=tag))
    return out


def funnel(cz, y0, y1, w, l, taper=0.85, rake=0.0, pipes=2, tag="funnel", cap_tag="black"):
    bot = sym([(w / 2, cz + l / 2), (w / 2, cz - l / 2)])
    top = [(x * taper, cz + (zz - cz) * taper - rake) for (x, zz) in bot]
    out = [house(bot, y0, y1, top, tag=tag)]
    for i in range(pipes):
        o = (i - (pipes - 1) / 2) * (l * taper / max(pipes, 1))
        out.append(cylinder((0, y1 - 0.2, cz - rake + o), (0, y1 + 1.2, cz - rake + o), min(w, l) * 0.18, n=8, tag=cap_tag))
    out.append(box(0, y1 + 0.05, cz - rake, w * taper * 0.95, 0.2, l * taper * 0.95, tag=cap_tag))
    return out


# ------------------------------------------------------------------------------- masts
def whip(x, y, z, h, r=0.05, tag="mast"):
    return cylinder((x, y, z), (x, y + h, z), r, r * 0.4, n=4, tag=tag)


def yard(c, span, depth=0.35, tag="mast"):
    """Yard arm.  depth = its fore-aft size: make it as deep as the mast at that height so it touches."""
    x, y, z = c
    return [box(x, y, z, span, 0.25, depth, tag=tag)] + [whip(x + s * span / 2, y, z, 1.6) for s in (1, -1)]


def lattice_mast(base, top, w0, w1, levels=4, tag="mast"):
    """Tapering four-legged lattice mast with frames and diagonal braces."""
    base = np.asarray(base, float); top = np.asarray(top, float)
    corners = [(1, 1), (1, -1), (-1, -1), (-1, 1)]
    out = [cylinder(base + [sx * w0 / 2, 0, sz * w0 / 2], top + [sx * w1 / 2, 0, sz * w1 / 2], 0.14, 0.09, n=4, tag=tag)
           for (sx, sz) in corners]
    for i in range(1, levels + 1):
        f = i / (levels + 1); c = base + (top - base) * f; w = w0 + (w1 - w0) * f
        for a, b in zip(corners, corners[1:] + corners[:1]):
            out.append(cylinder(c + [a[0] * w / 2, 0, a[1] * w / 2], c + [b[0] * w / 2, 0, b[1] * w / 2], 0.07, n=4, tag=tag))
        g = (i + 1) / (levels + 1); c2 = base + (top - base) * g; w2 = w0 + (w1 - w0) * g
        out.append(cylinder(c + [w / 2, 0, w / 2], c2 + [-w2 / 2, 0, w2 / 2], 0.06, n=4, tag=tag))
        out.append(cylinder(c + [w / 2, 0, -w / 2], c2 + [w2 / 2, 0, w2 / 2], 0.06, n=4, tag=tag))
    out.append(box(top[0], top[1] + 0.15, top[2], w1 + 0.4, 0.3, w1 + 0.4, tag=tag))
    return out


def pyramid_mast(cz, y0, y1, base_w, base_l, top_w, top_l, tag="super"):
    """Enclosed pyramid mast (Type 45, Kirov foremast...)."""
    return house(sym([(base_w / 2, cz + base_l / 2), (base_w / 2, cz - base_l / 2)]), y0, y1,
                 sym([(top_w / 2, cz + top_l / 2), (top_w / 2, cz - top_l / 2)]), tag=tag)


# ------------------------------------------------------------------------------- deck fittings
def rail_line(points, y, h=1.0, post=3.0, r=0.05, tag="rail"):
    """Guard rail along [(x, z)] at height y: top rail, mid rail, posts."""
    out = []
    for (x0, z0), (x1, z1) in zip(points, points[1:]):
        L = math.hypot(x1 - x0, z1 - z0)
        if L < 0.05:
            continue
        for yy in (y + h, y + h * 0.5):
            out.append(cylinder((x0, yy, z0), (x1, yy, z1), r, n=4, tag=tag))
        n = max(1, int(L / post))
        for i in range(n + 1):
            f = i / n
            out.append(cylinder((x0 + (x1 - x0) * f, y - 0.02, z0 + (z1 - z0) * f), (x0 + (x1 - x0) * f, y + h, z0 + (z1 - z0) * f),
                                0.03, n=4, tag=tag))
    return out


def deck_edge_rails(stations, z0, z1, step=6.0, inset=0.15, h=1.0, post=3.2, r=0.05, tag="rail"):
    """Rails following the deck edge (and sheer) on both sides between z0 and z1."""
    S = np.array(stations, float)
    out = []
    for s in (1, -1):
        pts = []
        for z in np.arange(z0, z1 + 0.01, step):
            d, hw = deck_at(stations, z)
            pts.append((s * (hw - inset), d, z))
        for (xa, ya, za), (xb, yb, zb) in zip(pts, pts[1:]):
            for yy in (h, h * 0.5):
                out.append(cylinder((xa, ya + yy, za), (xb, yb + yy, zb), r, n=4, tag=tag))
            n = max(1, int(math.hypot(xb - xa, zb - za) / post))
            for i in range(n):
                f = i / n
                out.append(cylinder((xa + (xb - xa) * f, ya + (yb - ya) * f - 0.05, za + (zb - za) * f),
                                    (xa + (xb - xa) * f, ya + (yb - ya) * f + h, za + (zb - za) * f), 0.03, n=4, tag=tag))
    return out


def bollards(stations, zs, inset=1.2, tag="dark"):
    out = []
    for z in zs:
        d, hw = deck_at(stations, z)
        for s in (1, -1):
            for dz in (-0.35, 0.35):
                out.append(cylinder((s * (hw - inset), d - 0.05, z + dz), (s * (hw - inset), d + 0.55, z + dz), 0.16, n=6, tag=tag))
    return out


def anchors(stations, z, y_off=-2.5, tag="dark"):
    d, hw = deck_at(stations, z)
    return [box(s * hw * 0.97, d + y_off, z, 0.25, 1.6, 1.2, tag=tag) for s in (1, -1)]


def boat(c, L=7.0, B=2.6, yaw=0.0, tag="boat"):
    """RHIB on chocks."""
    x, y, z = c
    st = [(-L / 2, 0.35, B * 0.42, 0.5, 0.2, 2.2), (-L / 4, 0.35, B / 2, 0.55, 0.3, 2.2), (L / 4, 0.35, B / 2, 0.55, 0.3, 2.2),
          (L / 2, 0.5, 0.1, 0.3, 0.1, 2)]
    out = [loft([ring_superellipse(0, cy, zz, hw, ht, hb, p=p, n=8) for (zz, cy, hw, ht, hb, p) in st], tag),
           box(0, 1.1, -L * 0.1, B * 0.5, 0.6, 1.2, tag=tag)]
    for s in out:
        if yaw:
            s.rotate(rot_y(yaw))
        s.translate(x, y, z)
    return out


def davit(base, h, reach, side, tag="mast"):
    """Davit post + arm reaching towards -side * x."""
    x, y, z = base
    return [box(x, y + h / 2, z, 0.35, h, 0.35, tag=tag), box(x - side * reach / 2, y + h, z, reach, 0.3, 0.3, tag=tag)]


def raft_rack(x, y, z, n=4, along="z", tag="white", rack_tag="mast"):
    out = []
    for i in range(n):
        o = (i - (n - 1) / 2) * 1.0
        if along == "z":
            out.append(cylinder((x, y, z + o - 0.4), (x, y, z + o + 0.4), 0.33, n=8, tag=tag))
        else:
            out.append(cylinder((x + o - 0.4, y, z), (x + o + 0.4, y, z), 0.33, n=8, tag=tag))
    out.append(box(x, y - 0.34, z, 0.8 if along == "z" else n * 1.0, 0.1, n * 1.0 if along == "z" else 0.8, tag=rack_tag))
    return out


def canisters(c, yaw, n=4, L=4.6, r=0.35, tilt=-15, tag="launcher"):
    """Quad Harpoon-style canister launcher."""
    x, y, z = c
    out = []
    for i in range(n):
        o = (i - (n - 1) / 2) * (r * 2.1)
        t = cylinder((o, 0.9, -L / 2), (o, 0.9, L / 2), r, n=6, tag=tag)
        t.rotate(rot_x(tilt), pivot=(0, 0.9, 0))
        out.append(t)
    out.append(box(0, 0.45, 0, n * r * 2.1, 0.9, 1.6, tag=tag))
    for s in out:
        s.rotate(rot_y(yaw)); s.translate(x, y, z)
    return out


def triple_tubes(c, yaw, tag="launcher"):
    """Triple lightweight torpedo tubes (Mk 32 style)."""
    x, y, z = c
    out = [cylinder((0, 0.5, 0), (0, 1.0, 0), 0.4, n=8, tag=tag)]
    for i in range(3):
        o = (i - 1) * 0.42
        out.append(cylinder((o, 1.2 + abs(o) * 0.2, -1.6), (o, 1.2 + abs(o) * 0.2, 1.6), 0.2, n=6, tag=tag))
    out.append(box(0, 1.15, 0, 1.4, 0.3, 0.8, tag=tag))
    for s in out:
        s.rotate(rot_y(yaw)); s.translate(x, y - 0.5, z)
    return out


def vls(cx, cz, w, l, y, h=0.6, tag="vls"):
    """VLS block (the cell lids are painted - see detail.grid_fn)."""
    return box(cx, y + h / 2, cz, w, h, l, tag=tag)


def ciws(x, y, z, kind="phalanx", tag="gunh"):
    """Close-in weapon mount: (static base, moving head shells, head pivot)."""
    base = [cylinder((x, y - 0.05, z), (x, y + 1.2, z), 0.7, 0.55, n=10, tag=tag)]
    if kind == "phalanx":
        head = [cylinder((x, y + 1.8, z - 0.2), (x, y + 3.2, z - 0.2), 0.55, 0.45, n=10, tag="dome"),
                box(x, y + 1.6, z, 1.0, 0.8, 1.2, tag=tag), cylinder((x, y + 1.6, z + 0.4), (x, y + 1.6, z + 2.0), 0.13, n=6, tag="gun")]
    else:
        head = [ellipsoid((x, y + 1.6, z), 0.75, 0.6, 0.8, nu=10, nv=6, tag=tag),
                cylinder((x, y + 1.6, z + 0.5), (x, y + 1.6, z + 2.1), 0.12, n=6, tag="gun")]
    return base, head, (x, y + 1.6, z)
