"""Retractable undercarriage: legs, retraction transforms, automatic stowage fitting, bay fairings.

Lessons from the modern pack
  * Thin modern wings / bodies rarely swallow a leg with a pure rotation.  Search the angle (and the fold
    direction for nose legs), then add a small shift after the rotation (engines like tudursvehiclemod
    offer "rotate then slide"), and only then cover what still pokes out with a bay fairing (blister).
  * The target is 0 % of the leg outside the structure (`attach.outside`), not "small".
  * Fairings on a left / right pair are always mirrored so the aircraft stays symmetric.

Retraction model: P' = R(axis, angle) (P - pivot) + pivot + shift      (shift in model space)
"""
import itertools
import numpy as np
from .geom import cylinder, rot_axis
from . import attach
from .fuselage import pod


# ----------------------------------------------------------------------------------- shells
def wheel(c, r, w, n=12, tag="tyre"):
    c = np.asarray(c, float)
    return cylinder(c - [w / 2, 0, 0], c + [w / 2, 0, 0], r, n=n, tag=tag)


def leg(top, axle, r, w, strut_r=0.07, twin=False, tag="gear"):
    """Strut from `top` (the hinge, inside the structure) to the axle, single or twin wheels."""
    t = np.asarray(top, float); a = np.asarray(axle, float)
    out = [cylinder(t, a + [0, r * 0.05, 0], strut_r, n=6, tag=tag)]
    if twin:
        out += [wheel(a + [w * 1.05, 0, 0], r, w), wheel(a - [w * 1.05, 0, 0], r, w),
                cylinder(a - [w * 1.6, 0, 0], a + [w * 1.6, 0, 0], strut_r * 0.8, n=6, tag=tag)]
    else:
        out += [cylinder(a - [w * 0.35, 0, 0], a + [w * 0.35, 0, 0], strut_r * 0.8, n=6, tag=tag), wheel(a, r, w)]
    return out


# ----------------------------------------------------------------------------------- transforms
class Retraction:
    def __init__(self, pivot, axis, angle, shift=(0.0, 0.0, 0.0)):
        self.pivot = np.asarray(pivot, float); self.axis = tuple(axis)
        self.angle = float(angle); self.shift = np.asarray(shift, float)

    def __call__(self, P, frac=1.0):
        R = rot_axis(self.axis, self.angle * frac)
        return (np.asarray(P) - self.pivot) @ R.T + self.pivot + self.shift * frac

    def rotate_then_slide_offset(self):
        """Offset for engines that rotate about the pivot first and then translate in the part's own
        (rotated) frame, i.e. P' = R (P - pivot + o) + pivot.  Returns o = R^T shift."""
        return rot_axis(self.axis, self.angle).T @ self.shift

    def as_dict(self):
        return {"pivot": self.pivot.round(4).tolist(), "axis": list(self.axis), "angle": self.angle,
                "shift": self.shift.round(4).tolist(), "offset_rotate_then_slide": self.rotate_then_slide_offset().round(4).tolist()}


def transforms(retractions, frac=1.0):
    """{part: Retraction} -> {part: fn} for attach / view / check."""
    return {p: (lambda P, r=r, f=frac: r(P, f)) for p, r in retractions.items()}


# ----------------------------------------------------------------------------------- fitting
def outside_points(model, part, fn, samples=3, also_static=()):
    """Surface points of `part` (moved by fn) that are outside every static shell."""
    static = []
    for pname in model.order:
        if pname == part or (pname.startswith("$") and pname not in also_static):
            continue
        for sh in model.parts[pname].shells:
            static.append(attach._tris(sh)[1])
    P = np.vstack([attach._surface_points(sh, samples) for sh in model.parts[part].shells])
    P = fn(P)
    ins = np.zeros(len(P), bool)
    for T in static:
        lo, hi = T.reshape(-1, 3).min(0), T.reshape(-1, 3).max(0)
        sel = ~ins & np.all((P >= lo) & (P <= hi), 1)
        if sel.any():
            ins[np.where(sel)[0][attach._inside(P[sel], T)]] = True
    return P[~ins]


def fit(model, part, pivot, axis, angles, shifts=((0, 0, 0),), samples=1, cost_per_m=0.01):
    """Try every angle x shift; return (Retraction, outside_fraction) with the least leg outside
    (ties broken towards the smallest shift).  angles may mix signs to test both fold directions."""
    best = None
    for a, s in itertools.product(angles, shifts):
        r = Retraction(pivot, axis, a, s)
        o = attach.outside(model, {part: r}, [part], samples=samples)[part]
        cost = o + cost_per_m * float(np.abs(s).sum())
        if best is None or cost < best[0] - 1e-9:
            best = (cost, r, o)
    return best[1], best[2]


def shift_grid(side, dx=(0, 0.08, 0.16, 0.24), dy=(0, 0.06, 0.12, 0.18, 0.24, 0.3), dz=(0, -0.15, 0.15)):
    """Candidate shifts: up (into the structure), inboard (side = +1 left leg, -1 right, 0 nose), fore/aft."""
    out = []
    for x, y, z in itertools.product(dx, dy, dz):
        if side == 0 and x:
            continue
        out.append((-side * x, y, z))
    return out


def fairings(model, retractions, part="fuselage", pairs=(("$gear_l", "$gear_r"),), margin=0.05, tag="skin", max_iter=4):
    """Cover whatever still pokes out of the skin with a smooth blister (gear-bay bulge).  Pairs get
    mirrored blisters.  Returns a list of (leg part, centre, half size) that were added."""
    mate = {}
    for a, b in pairs:
        mate[a] = b; mate[b] = a
    added = []
    for leg_part, r in retractions.items():
        for _ in range(max_iter):
            Q = outside_points(model, leg_part, r)
            if len(Q) == 0:
                break
            lo, hi = Q.min(0) - margin, Q.max(0) + margin
            lo[1] = min(lo[1], r.pivot[1] - 0.25); hi[1] = max(hi[1], r.pivot[1] + 0.05)   # reach into the structure
            c = (lo + hi) / 2; h = (hi - lo) / 2 * 1.2 + 0.02
            zs = [c[2] - h[2] * 1.15, c[2] - h[2] * 0.7, c[2] + h[2] * 0.7, c[2] + h[2] * 1.15]
            st = [(zs[0], c[1], h[0] * 0.35, h[1] * 0.35, h[1] * 0.35, 2.0), (zs[1], c[1], h[0], h[1], h[1], 4.0),
                  (zs[2], c[1], h[0], h[1], h[1], 4.0), (zs[3], c[1], h[0] * 0.35, h[1] * 0.35, h[1] * 0.35, 2.0)]
            model.add(part, pod(c[0], st, n=10, tag=tag))
            if leg_part in mate and abs(c[0]) > 1e-3:
                model.add(part, pod(-c[0], st, n=10, tag=tag))
            added.append((leg_part, c.round(3).tolist(), h.round(3).tolist()))
    return added


def stow(model, legs, samples=1, nose_angles=None, main_angles=None, shifts=True, fairing_part="fuselage"):
    """One call for a whole undercarriage.  legs = {part: (pivot, axis, default_angle, side)} with side
    +1 / -1 for main legs, 0 for the nose leg.  Fits angle (+ shift), adds fairings, returns
    ({part: Retraction}, report) where report holds the outside fractions before / after."""
    rets, rep = {}, {}
    for part, (pv, ax, a0, side) in legs.items():
        sg = 1 if a0 >= 0 else -1
        if side == 0:
            angs = nose_angles or [sg * d for d in range(80, 120, 4)] + [-sg * d for d in range(80, 120, 4)]
        else:
            angs = main_angles or [sg * d for d in range(abs(int(a0)) - 8, abs(int(a0)) + 28, 4)]
        r, o0 = fit(model, part, pv, ax, angs, samples=samples)
        if shifts and o0 > 0:
            r, o0 = fit(model, part, pv, ax, [r.angle], shift_grid(side), samples=samples)
        rets[part] = r; rep[part] = {"fit_outside": round(o0, 3)}
    added = fairings(model, rets, part=fairing_part)
    out = attach.outside(model, rets, list(rets), samples=3)
    for p in rets:
        rep[p]["outside"] = out[p]
    rep["fairings"] = added
    return rets, rep
