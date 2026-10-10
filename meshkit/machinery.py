"""Construction machinery kit (v0.6): crawlers, hydraulic rams, outriggers, clamp jaws and a joint
solver for posing / checking moving parts.

Lessons of the Construction Machinery Addon sample vehicles and the Construction Detail Pack
(tudursvehiclemod, 2026-10): backhoes, a bulldozer, a crane, drop-hammer / rotary pile drivers with
widening crawlers, trucks and a trailer.  See AI_GUIDE.md chapter 9.

Conventions are meshkit's: +X left, +Y up, +Z forward, 1 unit = 1 m.  A crawler is described in its
side view as (y, z) points.  Joints are plain dicts in the Construction Machinery Addon format
(the format tudursvehiclemod construction add-ons read), so what is checked here is what the game runs:

    {"part": "$arm", "parent": "$boom", "pivot_x": .., "pivot_y": .., "pivot_z": ..,
     "axis_x": .., "axis_y": .., "axis_z": .., "mode": "rotate", "channel": "arm",
     "factor": 1.0, "offset": 0.0, "min": .., "max": ..}

modes: rotate (degrees = channel * factor + offset), slide (units along the axis), scale (along the
axis about the pivot), aim (turns about the axis so that the part points at "target", a point on
another part - the barrel of a ram), reach (slides along the axis by the growth of the distance
"anchor" -> "target" times factor - the rod of a ram), belt (turns by belt travel * factor - rollers).
"""
import math

import numpy as np

from .geom import box, cylinder, loft, rot_axis

# ------------------------------------------------------------------------------- small helpers


def unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def rot(v, axis, deg):
    """Right-handed rotation of vector v about the unit axis (Rodrigues)."""
    v = np.asarray(v, float)
    k = unit(axis)
    a = math.radians(deg)
    return v * math.cos(a) + np.cross(k, v) * math.sin(a) + k * np.dot(k, v) * (1 - math.cos(a))


def turn(p, pivot, axis, deg):
    """Point p turned about the line (pivot, axis)."""
    pivot = np.asarray(pivot, float)
    return pivot + rot(np.asarray(p, float) - pivot, axis, deg)


def bar(a, b, w, h, tag="frame", up=(0, 1, 0)):
    """Rectangular bar from a to b: w across (perpendicular to `up`), h along `up`."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    d = unit(b - a)
    side = np.cross(up, d)
    if np.linalg.norm(side) < 1e-6:
        side = np.array([1.0, 0, 0])
    side = unit(side)
    upv = np.cross(d, side)

    def ring(p):
        return np.array([p + side * w / 2 + upv * h / 2, p - side * w / 2 + upv * h / 2,
                         p - side * w / 2 - upv * h / 2, p + side * w / 2 - upv * h / 2])
    return loft([ring(a), ring(b)], tag)


def joint(part, pivot, axis, mode="rotate", channel=None, parent=None, **extra):
    """A joint dict (Construction Machinery Addon format)."""
    j = {"part": part, "pivot_x": float(pivot[0]), "pivot_y": float(pivot[1]), "pivot_z": float(pivot[2]),
         "axis_x": float(axis[0]), "axis_y": float(axis[1]), "axis_z": float(axis[2]), "mode": mode}
    if channel:
        j["channel"] = channel
    if parent:
        j["parent"] = parent
    j.update(extra)
    return j


def point(part, p):
    """A point on a part (for aim / reach targets)."""
    return {"part": part, "x": float(p[0]), "y": float(p[1]), "z": float(p[2])}


# ------------------------------------------------------------------------------- crawlers

def crawler_layout(z_half, height, r_end, wheel_r, n_wheels, carrier_r=0.0, n_carrier=0, shoe_t=0.07, gap=0.04):
    """Running gear of one crawler in its side view (tank practice - the shoes run round the idler,
    the sprocket and under the road wheels, and nothing overlaps):

    - idler (front) and sprocket (rear): radius r_end, centred z = +-zc, tops touching the inside of the
      top run (height - shoe_t);
    - road wheels on the bottom run (centre y = shoe_t + wheel_r), at least `gap` clear of the end wheels
      and of each other - wheels are dropped (n < n_wheels) rather than squeezed;
    - carrier rollers under the top run, clear of the end wheels;
    - path: the centre line of the shoes, a convex closed loop in the order front-bottom, up the front,
      back along the top, down the rear (the order tudursvehiclemod's crawler tracks use).

    z_half is half the overall crawler length, height its overall height.  Returns a dict
    {t, zc, yc, r_end, wheels [(y, z)], wheel_r, carriers [(y, z)], carrier_r, path [(y, z)]}."""
    t = shoe_t
    R = r_end + t / 2
    zc = z_half - r_end - t
    yc = height - t - r_end
    dy = yc - (t + wheel_r)
    need = r_end + wheel_r + gap
    dz = math.sqrt(max(need * need - dy * dy, 0.0)) if need > abs(dy) else 0.0
    z_lo, z_hi = -zc + dz, zc - dz
    n = max(2, n_wheels)
    while n > 2 and (z_hi - z_lo) / (n - 1) < 2 * wheel_r + gap:
        n -= 1
    wheels = [(t + wheel_r, float(z)) for z in np.linspace(z_lo, z_hi, n)]
    carriers = []
    if n_carrier:
        lim = zc - (r_end + carrier_r + 2 * gap)
        zs = [0.0] if n_carrier == 1 else list(np.linspace(-lim * 0.55, lim * 0.55, n_carrier))
        carriers = [(height - t - carrier_r, float(z)) for z in zs]
    touching = (yc - r_end) - t < 0.03          # end wheels reach down to the ground run
    a0 = -90.0 if touching else -45.0
    front = [(yc + R * math.sin(math.radians(a)), zc + R * math.cos(math.radians(a))) for a in np.linspace(a0, 90.0, 7)]
    rear = [(yc + R * math.sin(math.radians(a)), -zc + R * math.cos(math.radians(a)))
            for a in np.linspace(90.0, 180.0 - a0, 7)]
    path = []
    if not touching:
        path.append((t / 2, z_hi + wheel_r * 0.5))
    path += front + rear
    if not touching:
        path.append((t / 2, z_lo - wheel_r * 0.5))
    return {"t": t, "zc": zc, "yc": yc, "r_end": r_end, "wheels": wheels, "wheel_r": wheel_r,
            "carriers": carriers, "carrier_r": carrier_r, "path": [(float(y), float(z)) for y, z in path]}


def layout_overlaps(layout, gap=0.0):
    """Pairs of wheels (end wheels, road wheels, carrier rollers) closer than `gap` in the side view.
    Empty when the running gear is clean.  Returns [(name_a, name_b, clearance)]."""
    L = layout
    circles = [("idler", L["yc"], L["zc"], L["r_end"]), ("sprocket", L["yc"], -L["zc"], L["r_end"])]
    circles += [("wheel%d" % i, y, z, L["wheel_r"]) for i, (y, z) in enumerate(L["wheels"])]
    circles += [("carrier%d" % i, y, z, L["carrier_r"]) for i, (y, z) in enumerate(L["carriers"])]
    out = []
    for i in range(len(circles)):
        for k in range(i + 1, len(circles)):
            a, b = circles[i], circles[k]
            c = math.hypot(a[1] - b[1], a[2] - b[2]) - a[3] - b[3]
            if c < gap - 1e-9:
                out.append((a[0], b[0], round(c, 4)))
    return out


def path_length(path):
    n = len(path)
    return sum(math.hypot(path[(i + 1) % n][0] - path[i][0], path[(i + 1) % n][1] - path[i][1]) for i in range(n))


def link_spacing(path, link_len=0.2):
    """Link pitch that divides the closed path evenly (close to link_len)."""
    L = path_length(path)
    return L / max(1, round(L / link_len))


def place_on_path(path, d):
    """(y, z, angle degrees of the run) at distance d along the closed path."""
    n = len(path)
    cum = [0.0]
    for i in range(n):
        a, b = path[i], path[(i + 1) % n]
        cum.append(cum[-1] + math.hypot(b[0] - a[0], b[1] - a[1]))
    d = d % cum[-1]
    k = 0
    while k < n - 1 and cum[k + 1] < d:
        k += 1
    a, b = path[k], path[(k + 1) % n]
    f = (d - cum[k]) / max(cum[k + 1] - cum[k], 1e-9)
    return a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))


def link_shells(x_in, x_out, length, t, place=None, tag="track"):
    """One shoe (plate + grouser) centred at y = z = 0, local +Y outward, local Z along the run - the
    frame an engine instancing one link model uses - or moved to place = (y, z, angle)."""
    x0, x1 = sorted((x_in, x_out))
    shells = [box((x0 + x1) / 2, 0, 0, x1 - x0, t, length * 0.9, tag=tag),
              box((x0 + x1) / 2, t / 2 + 0.015, 0, (x1 - x0) * 0.96, 0.03, 0.05, tag=tag)]
    if place is not None:
        y, z, ang = place
        for sh in shells:
            sh.rotate(rot_axis((1, 0, 0), ang + 90.0), pivot=(0, 0, 0))
            sh.translate(0, y, z)
    return shells


def crawler(model, side, x_in, x_out, z_half, height, r_end, wheel_r, n_wheels, carrier_r=0.0, n_carrier=0,
            frame_part="body", link_len=0.19, shoe_t=0.07, sprocket_rear=True, links="one", prefix="$",
            frame_tag="frame", wheel_tag="wheel", track_tag="track", travel=0.0):
    """A complete crawler for one side ("l" / "r") built from crawler_layout: side frame (between the road
    wheels and the carrier rollers), yokes for the end wheels, a toothed sprocket, idler, road wheels and
    carrier rollers (each its own part with a pivot and a mark on the face so turning shows) and shoes.

    links: "one"  - a single shoe in part <prefix>crawler_<side> at the link frame (engine instances it);
           "laid" - all shoes laid round the path in that part (preview / contact checks);
           "each" - one part per shoe, <prefix>link_<side>_<i> at its travel-0 place (an add-on moves them).
    frame_part: the part the frame and every roller ride on - give the moving frame of a widening crawler
    so that the whole crawler (shoes too) moves with it.

    Returns {"layout", "rollers": [(part, (x, y, z) pivot, radius)], "path", "spacing", "count", "x",
    "link_part" or "link_prefix"}.  Roller turn rate: 1 / (2 pi r) turns per unit of belt travel."""
    L = crawler_layout(z_half, height, r_end, wheel_r, n_wheels, carrier_r, n_carrier, shoe_t)
    s = 1 if x_in > 0 else -1
    xc = (x_in + x_out) / 2
    t, zc, yc, re = L["t"], L["zc"], L["yc"], L["r_end"]
    path = L["path"]
    spacing = link_spacing(path, link_len)
    count = int(round(path_length(path) / spacing))
    rollers = []
    name = lambda k: "%s%s_%s" % (prefix, k, side)
    fy0 = t + 2 * wheel_r + 0.02
    fy1 = (height - t - 2 * carrier_r - 0.02) if n_carrier else (yc + re * 0.35)
    fz = zc - re - 0.03
    fi, fo = x_in + s * 0.06, x_out - s * 0.06
    model.add(frame_part, box((fi + fo) / 2, (fy0 + fy1) / 2, 0, abs(fo - fi), fy1 - fy0, 2 * fz, tag=frame_tag))
    xa = x_in + s * 0.07                                     # yokes / brackets: overlap frame and wheel hubs
    for zz in (zc, -zc):                                     # yokes carrying the end wheels
        model.add(frame_part, bar((xa, yc, zz), (xa, (fy0 + fy1) / 2, zz - np.sign(zz) * (re + 0.1)), 0.06, 0.12, frame_tag))
    spr_z, idl_z = (-zc, zc) if sprocket_rear else (zc, -zc)
    p = name("sprocket")
    model.add(p, cylinder((x_in + s * 0.03, yc, spr_z), (x_out - s * 0.03, yc, spr_z), re * 0.97, n=12, tag=wheel_tag))
    for k in range(10):                                      # teeth
        a = 2 * math.pi * k / 10
        model.add(p, box(xc - s * 0.08, yc + math.sin(a) * re, spr_z + math.cos(a) * re, 0.06, 0.07, 0.07, tag=wheel_tag))
    rollers.append((p, (xc, yc, spr_z), re))
    p = name("idler")
    model.add(p, cylinder((x_in + s * 0.03, yc, idl_z), (x_out - s * 0.03, yc, idl_z), re * 0.97, n=14, tag=wheel_tag))
    model.add(p, box(x_out - s * 0.025, yc + re * 0.6, idl_z, 0.03, 0.08, 0.08, tag=frame_tag))
    rollers.append((p, (xc, yc, idl_z), re))
    for i, (y, z) in enumerate(L["wheels"]):
        p = name("wheel%d" % i)
        model.add(p, cylinder((x_in + s * 0.07, y, z), (x_out - s * 0.07, y, z), wheel_r, n=10, tag=wheel_tag))
        model.add(p, box(x_out - s * 0.0625, y + wheel_r * 0.575, z, 0.025, wheel_r * 0.55, 0.04, tag=frame_tag))
        # axle stub into the wheel and a bracket up to the frame (a wheel must not float under the frame)
        model.add(frame_part, cylinder((x_in, y, z), (x_in + s * 0.1, y, z), wheel_r * 0.3, n=6, tag=frame_tag))
        model.add(frame_part, bar((x_in + s * 0.03, y, z), (x_in + s * 0.03, fy0 + 0.04, z), 0.06, 0.08, frame_tag,
                                  up=(0, 0, 1)))
        rollers.append((p, (xc, y, z), wheel_r))
    for i, (y, z) in enumerate(L["carriers"]):
        p = name("carrier%d" % i)
        model.add(p, cylinder((xc - s * 0.12, y, z), (xc + s * 0.12, y, z), carrier_r, n=8, tag=wheel_tag))
        model.add(frame_part, cylinder((x_in + s * 0.04, y, z), (xc - s * 0.1, y, z), carrier_r * 0.4, n=6, tag=frame_tag))
        model.add(frame_part, bar((xa, y, z), (xa, fy1 - 0.04, z), 0.06, 0.08, frame_tag, up=(0, 0, 1)))
        rollers.append((p, (xc, y, z), carrier_r))
    for p, piv, _ in rollers:
        model.pivots[p] = tuple(float(v) for v in piv)
    out = {"layout": L, "rollers": rollers, "path": path, "spacing": spacing, "count": count, "x": xc}
    if links == "each":
        pre = "%slink_%s_" % (prefix, side)
        for i in range(count):
            model.add(pre + str(i), *link_shells(x_in, x_out, spacing, t, place_on_path(path, i * spacing), track_tag))
        out["link_prefix"] = pre
    else:
        part = name("crawler")
        if links == "laid":
            for i in range(count):
                model.add(part, *link_shells(x_in, x_out, spacing, t, place_on_path(path, i * spacing + travel), track_tag))
        else:
            model.add(part, *link_shells(x_in, x_out, spacing, t, None, track_tag))
        out["link_part"] = part
    return out


# ------------------------------------------------------------------------------- hydraulic rams

def ram_lengths(base, attach, pivot, axis, angles):
    """Shortest and longest base -> attach distance while the child (which carries `attach`) turns about
    (pivot, axis) through `angles` (degrees; the model is the pose at angle 0)."""
    base = np.asarray(base, float)
    lens = [float(np.linalg.norm(turn(attach, pivot, axis, a) - base)) for a in angles]
    lens.append(float(np.linalg.norm(np.asarray(attach, float) - base)))
    return min(lens), max(lens)


def ram_stages(lmin, lmax, overlap=0.92, max_ratio=1.75):
    """How many nested parts a ram needs: (stages, simple).  simple = a classic barrel and one rod (the
    rod can stay well inside the barrel: lmax <= max_ratio * lmin).  More than 2 stages is a telescopic
    ram - fine for a 3-stage dump body hoist, wrong-looking for a boom or leader (move the mounts)."""
    S = lmin * overlap
    n = max(2, int(math.ceil((lmax - S) / (S * 0.82))) + 1)
    return n, (n == 2 and lmax <= max_ratio * lmin)


def ram_leverage(base, attach, pivot, axis, angles):
    """Smallest moment arm of the ram about the child's hinge through the stroke: the distance from the
    hinge line to the ram's line (in the plane across the axis).  A ram mounted almost through the hinge
    hardly changes length and could not lift the part - keep this well above zero (about 1/10 of the
    distance hinge -> attachment or more)."""
    k = unit(axis)
    base = np.asarray(base, float)
    pivot = np.asarray(pivot, float)
    arm = float("inf")
    for a in list(angles) + [0.0]:
        u = unit(turn(attach, pivot, axis, a) - base)
        arm = min(arm, abs(float(np.dot(np.cross(base - pivot, u), k))))
    return arm


def ram_mount_search(pivot, axis, angles, bases, attaches, min_arm=0.3, max_stages=2, simple=True):
    """Try every base (on the parent) x attachment (on the child, rest pose) pair; keep those with at most
    max_stages stages (and a simple barrel + rod if `simple`) and at least min_arm leverage.
    Returns [(base, attach, stages, lmin, lmax, arm)] best leverage first."""
    out = []
    for b in bases:
        for a in attaches:
            lmin, lmax = ram_lengths(b, a, pivot, axis, angles)
            if lmin < 1e-3:
                continue
            n, simp = ram_stages(lmin, lmax)
            if n > max_stages or (simple and not simp):
                continue
            arm = ram_leverage(b, a, pivot, axis, angles)
            if arm >= min_arm:
                out.append((tuple(map(float, b)), tuple(map(float, a)), n, round(lmin, 4), round(lmax, 4), round(arm, 4)))
    out.sort(key=lambda r: -r[5])
    return out


def ram(model, name, base, attach, r, lmin, lmax, tag="paint", rod_tag="chrome", barrel_frac=0.62, stages=None):
    """Ram geometry in the rest pose from `base` to `attach`: part <name> (barrel) and <name>_s1 ... (rod
    or nested stages).  lmin / lmax from ram_lengths.  Returns the part names in order."""
    base = np.asarray(base, float)
    attach = np.asarray(attach, float)
    L0 = float(np.linalg.norm(attach - base))
    u = (attach - base) / L0
    n, simple = ram_stages(lmin, lmax)
    n = stages or n
    parts = [name] + ["%s_s%d" % (name, i) for i in range(1, n)]
    if n == 2 and simple:
        barrel = min(L0 * barrel_frac, lmin * 0.9)
        model.add(parts[0], cylinder(base - u * r * 0.6, base + u * barrel, r, n=8, tag=tag))
        model.add(parts[0], cylinder(base + u * (barrel - 0.04), base + u * (barrel + 0.02), r * 1.12, n=8, tag=tag))
        model.add(parts[1], cylinder(base + u * max(L0 - lmin * 0.95, r), attach, r * 0.55, n=6, tag=rod_tag))
        model.add(parts[1], cylinder(attach - u * 0.06, attach + u * 0.06, r * 0.9, n=6, tag=rod_tag))
    else:
        S = lmin * 0.92
        step = (L0 - S) / (n - 1)
        for i, part in enumerate(parts):
            rr = r * (1.0 - 0.17 * i)
            a0 = base + u * (step * i)
            model.add(part, cylinder(a0 - (u * r * 0.6 if i == 0 else 0), a0 + u * S, rr, n=8, tag=tag if i == 0 else rod_tag))
        model.add(parts[-1], cylinder(attach - u * 0.08, attach + u * 0.08, r * 0.9, n=6, tag=rod_tag))
    for p in parts:
        model.pivots[p] = tuple(float(v) for v in base)
    return parts


def ram_joints(parts, parent, base, attach, child, axis):
    """Exact joints for a ram made by ram(): the barrel aims at the attachment point on `child`, every
    further stage takes an equal share of the growth in length.  The ram stays on both pins in every
    pose of the child (a linear channel -> length approximation drifts off at mid stroke)."""
    base = np.asarray(base, float)
    attach = np.asarray(attach, float)
    u = unit(attach - base)
    tgt = point(child, attach)
    anc = point(parts[0], base)
    out = [joint(parts[0], base, unit(axis), mode="aim", parent=parent, target=tgt)]
    for i in range(1, len(parts)):
        out.append(joint(parts[i], (0, 0, 0), u, mode="reach", parent=parts[i - 1], target=tgt, anchor=anc,
                         factor=round(1.0 / (len(parts) - 1), 6)))
    return out


# ------------------------------------------------------------------------------- outriggers, jaws

def corner_mounts(front, rear, x, offset):
    """Four outrigger mounts (name, (x, z), (sx, sz) outward diagonal) at the corners (+-x, front) and
    (+-x, rear) of a frame, each pushed out by `offset` along its diagonal."""
    k = offset / math.sqrt(2)
    out = []
    for side, s in (("l", 1), ("r", -1)):
        out.append(("outrigger_f" + side, (s * (x + k), front + k), (s, 1)))
        out.append(("outrigger_r" + side, (s * (x + k), rear - k), (s, -1)))
    return out


def outriggers(model, mounts, y_beam, beam_h, stroke, pad_y, pad_r, rod_top, housing_top, parent, channel="outrigger",
               prefix="$", beam_tag="frame", jack_tag="paint", rod_tag="chrome"):
    """Box beams that slide out diagonally, then jack cylinders that push pads down - both on ONE channel
    (0 -> 0.5 beam out, 0.5 -> 1 jack down) using factor / min / max, so the pads never come down before
    the beams are clear of the crawlers.  Mount them on the swinging upper structure when they must turn
    with it (give `parent`).  Returns the joints."""
    joints = []
    w = beam_h * 1.25
    hr = w * 0.62
    for nm, (x, z), (sx, sz) in mounts:
        d = np.array([sx, 0.0, sz]) / math.hypot(sx, sz)
        outer = np.array([x, y_beam, z])
        inner = outer - d * (stroke + 0.5)
        g, leg = prefix + nm, "%s%s_leg" % (prefix, nm)
        model.add(g, bar(inner, outer + d * hr, w, beam_h, beam_tag))
        model.add(g, cylinder((x, y_beam - beam_h / 2 - 0.03, z), (x, housing_top, z), hr, n=10, tag=jack_tag))
        model.add(leg, cylinder((x, pad_y + 0.07, z), (x, rod_top, z), hr * 0.55, n=8, tag=rod_tag))
        model.add(leg, cylinder((x, pad_y, z), (x, pad_y + 0.07, z), pad_r, n=14, tag=beam_tag))
        joints.append(joint(g, (0, 0, 0), d, mode="slide", channel=channel, parent=parent,
                            factor=round(2 * stroke, 4), min=0.0, max=round(stroke, 4)))
        joints.append(joint(leg, (0, 0, 0), (0, -1, 0), mode="slide", channel=channel, parent=g,
                            factor=round(2 * pad_y, 4), offset=round(-pad_y, 4), min=0.0, max=round(pad_y, 4)))
    return joints


def clamp_jaws(model, parent, cx, cz, r_mid, band, y0, y1, channel="steady", open_deg=100.0, prefix="$steady",
               tag="paint", n=8, lug_to=None):
    """Two half rings round a vertical member at (cx, cz) - hollow in the middle where the member (a pile)
    passes - each swinging outward about a vertical hinge pin behind its rear corner ("channel" 0 =
    closed round the member, 1 = open).  A pile steady (振れ止め) at a leader foot; also clamps and
    guides.  Never model such a guide as a solid block over the member: the pile, cap and auger pass
    through it.  lug_to: (x_half, z) of the parent's face the hinge lugs reach back to (default: the
    member axis side, r_mid behind the hinge) - check with attach.groups that the lugs touch it.
    Returns the joints."""
    joints = []
    ym = (y0 + y1) / 2
    for s, side in ((1, "l"), (-1, "r")):
        g = "%s_%s" % (prefix, side)
        pts = [(cx + s * r_mid * math.cos(math.radians(a)), cz + r_mid * math.sin(math.radians(a)))
               for a in np.linspace(-90.0, 90.0, n + 1)]
        for (x0, z0), (x1, z1) in zip(pts, pts[1:]):
            model.add(g, bar((x0, ym, z0), (x1, ym, z1), band, y1 - y0, tag))
        hinge = np.array([cx + s * r_mid, ym, cz - r_mid])
        a45 = np.array([cx + s * r_mid * 0.7071, ym, cz - r_mid * 0.7071])
        model.add(g, bar(hinge, a45, band * 0.8, (y1 - y0) * 0.8, tag))
        model.add(g, cylinder(hinge + [0, -(y1 - y0) / 2 - 0.04, 0], hinge + [0, (y1 - y0) / 2 + 0.04, 0], band * 0.45, n=8, tag=tag))
        lx, lz = lug_to if lug_to is not None else (r_mid * 0.4, hinge[2] - r_mid)
        for yy in (y1 + 0.08, y0 - 0.08):                                      # hinge lugs on the parent
            model.add(parent, bar((hinge[0], yy, hinge[2] + 0.08), (cx + s * lx, yy, lz - 0.04), 0.16, 0.08, tag))
        model.pivots[g] = tuple(float(v) for v in hinge)
        joints.append(joint(g, hinge, (0, s, 0), channel=channel, factor=open_deg, parent=parent))
    return joints


# ------------------------------------------------------------------------------- posing

def _rot4(axis, deg, pivot):
    R = np.array([rot(e, axis, deg) for e in np.eye(3)]).T
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = pivot - R @ pivot
    return M


def solve(joints, values, travel=0.0, crawlers=()):
    """One 4x4 model matrix per jointed part for channel `values` ({channel: value}) and belt `travel`.
    Parents are resolved first; aim / reach read the current position of their target, so rams follow.
    crawlers: [{"parent", "link_prefix", "link_spacing", "path": [(y, z)]}] moves links laid with
    crawler(links="each") round their path by `travel`."""
    by_part = {j["part"]: j for j in joints}
    done, visiting = {}, set()

    def vec(j, k):
        return np.array([j.get(k + "_x", 0.0), j.get(k + "_y", 0.0), j.get(k + "_z", 0.0)], float)

    def matrix_of(part):
        if part not in by_part or part in visiting:
            return done.get(part, np.eye(4))
        return resolve(by_part[part])

    def point_now(p):
        v = np.array([p.get("x", 0.0), p.get("y", 0.0), p.get("z", 0.0), 1.0])
        return (matrix_of(p["part"]) @ v)[:3] if p.get("part") else v[:3]

    def resolve(j):
        if j["part"] in done:
            return done[j["part"]]
        visiting.add(j["part"])
        P = np.eye(4)
        if j.get("parent") in by_part and j["parent"] not in visiting:
            P = resolve(by_part[j["parent"]])
            if j.get("inherit_rotation", True) is False:
                piv = vec(j, "pivot")
                Q = np.eye(4)
                Q[:3, 3] = (P @ np.append(piv, 1))[:3] - piv
                P = Q
        piv = vec(j, "pivot")
        ax = unit(vec(j, "axis")) if np.linalg.norm(vec(j, "axis")) > 1e-9 else np.array([1.0, 0, 0])
        mode = j.get("mode", "rotate")
        f = j.get("factor", 1.0)
        off = j.get("offset", 1.0 if mode == "scale" else 0.0)
        v = values.get(j.get("channel"), 0.0) * f + off if j.get("channel") else off
        lo, hi = j.get("min", -1e9), j.get("max", 1e9)
        v = min(max(v, min(lo, hi)), max(lo, hi))
        M = np.eye(4)
        if mode == "rotate" and j.get("channel"):
            M = _rot4(ax, v, piv)
        elif mode == "slide":
            M[:3, 3] = ax * v
        elif mode == "scale" and j.get("channel"):
            S = np.eye(4)
            S[:3, :3] = np.eye(3) + (max(0.02, v) - 1.0) * np.outer(ax, ax)
            T = np.eye(4)
            T[:3, 3] = piv
            Ti = np.eye(4)
            Ti[:3, 3] = -piv
            M = T @ S @ Ti
        elif mode == "aim":
            t = j["target"]
            now = (np.linalg.inv(P) @ np.append(point_now(t), 1))[:3] - piv
            rest = np.array([t["x"], t["y"], t["z"]]) - piv
            now -= ax * now.dot(ax)
            rest -= ax * rest.dot(ax)
            ang = math.degrees(math.atan2(np.cross(rest, now).dot(ax), rest.dot(now)))
            M = _rot4(ax, ang + j.get("offset", 0.0), piv)
        elif mode == "reach":
            t, a = j["target"], j["anchor"]
            d_now = np.linalg.norm(point_now(t) - point_now(a))
            d_rest = np.linalg.norm(np.array([t["x"], t["y"], t["z"]]) - np.array([a["x"], a["y"], a["z"]]))
            M[:3, 3] = ax * ((d_now - d_rest) * f + j.get("offset", 0.0))
        elif mode == "belt":
            M = _rot4(ax, f * travel + j.get("offset", 0.0), piv)
        R = P @ M
        done[j["part"]] = R
        visiting.discard(j["part"])
        return R

    for j in joints:
        resolve(j)
    for c in crawlers:
        path = [tuple(p) if not isinstance(p, dict) else (p["y"], p["z"]) for p in c["path"]]
        sp = c["link_spacing"]
        count = int(round(path_length(path) / sp))
        Pm = matrix_of(c["parent"]) if c.get("parent") else np.eye(4)
        for i in range(count):
            y0, z0, a0 = place_on_path(path, i * sp)
            y1, z1, a1 = place_on_path(path, i * sp + travel)
            T0 = np.eye(4)
            T0[:3, 3] = [0, -y0, -z0]
            T1 = np.eye(4)
            T1[:3, 3] = [0, y1, z1]
            done[c["link_prefix"] + str(i)] = Pm @ T1 @ _rot4(np.array([1.0, 0, 0]), a1 - a0, np.zeros(3)) @ T0
    return done


def pose_transforms(joints, values, travel=0.0, crawlers=()):
    """solve() as point-array functions for view.render_view(..., transforms=) and the attach / check
    functions."""
    out = {}
    for part, M in solve(joints, values, travel, crawlers).items():
        out[part] = (lambda MM: (lambda P: (np.c_[P, np.ones(len(P))] @ MM.T)[:, :3]))(M)
    return out


def sweep_poses(channels, steps=(0.0, 0.25, 0.5, 0.75, 1.0), base=None):
    """Poses to check: every channel at every step, the others at `base` ({channel: value}, default 0),
    plus all channels together at each step.  channels: {channel: (lo, hi)}."""
    base = dict(base or {})
    poses = []
    for ch, (lo, hi) in channels.items():
        for s in steps:
            p = dict(base)
            p[ch] = lo + (hi - lo) * s
            poses.append(p)
    for s in steps:
        poses.append({**base, **{ch: lo + (hi - lo) * s for ch, (lo, hi) in channels.items()}})
    return poses


def ram_drift(joints, values_list, rams):
    """How far each ram's rod end is from its attachment point on the child, over the poses (the largest
    over all poses).  rams: [(rod part, attach point (rest), child part)].  Should be ~0 with ram_joints;
    use it to catch rams driven by factor approximations or mounted on the wrong parent."""
    worst = {}
    for vals in values_list:
        mats = solve(joints, vals)
        for rod, attach, child in rams:
            p = np.append(np.asarray(attach, float), 1)
            a = (mats.get(rod, np.eye(4)) @ p)[:3]
            b = (mats.get(child, np.eye(4)) @ p)[:3]
            worst[rod] = max(worst.get(rod, 0.0), float(np.linalg.norm(a - b)))
    return {k: round(v, 4) for k, v in worst.items()}


def posed(model, joints, values, travel=0.0, crawlers=()):
    """A copy of the model with every jointed part moved to the pose (for checks that compare two moving
    parts, and for exporting a posed preview)."""
    import copy
    pm = copy.deepcopy(model)
    for part, M in solve(joints, values, travel, crawlers).items():
        if part not in pm.parts:
            continue
        for sh in pm.parts[part].shells:
            sh.verts = [M[:3, :3] @ v + M[:3, 3] for v in sh.verts]
    return pm


def sweep_check(model, joints, poses, pairs=(), skip=(), travel=0.0, min_pen=0.02):
    """Floating groups and penetrations at every pose.  pairs: [(parts_a, parts_b)] that must not cut
    into each other (e.g. outrigger pads vs crawlers, a steady vs the rotary head) - both sides posed.
    Returns [{"pose", "groups", "penetration": {pair index: {part: fraction}}}] for poses with a problem
    (fraction = share of the surface of a part inside the other side; above min_pen counts)."""
    from . import attach, check
    bad = []
    for vals in poses:
        pm = posed(model, joints, vals, travel)
        g = attach.groups(pm, skip=skip)
        pen = {}
        for i, (a, b) in enumerate(pairs):
            r = check.penetration(pm, list(a), list(b))
            r = {k: v for k, v in r.items() if v > min_pen}
            if r:
                pen[i] = r
        if len(g) > 1 or pen:
            bad.append({"pose": vals, "groups": len(g), "penetration": pen})
    return bad
