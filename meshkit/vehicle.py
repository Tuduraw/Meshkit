"""tudursvehiclemod vehicle JSON helpers for previews and checks (0.7.0).

The mod draws some parts from the vehicle JSON rather than from where they sit in the OBJ. The most important case
is the crawler track: the OBJ holds ONE track link per side ("$track_l" / "$track_r"), and the mod repeats it around
the closed path of `crawler_tracks[].path`, `link_spacing` apart, turning each copy with the path
(VehicleEntityRenderer.getCrawlerTrackPlacements: count = round(length / spacing), angle = atan2(dz, dy) + 90 deg about
X, translate (0, y, z); flip=true reverses the path and mirrors the link in X). Without this, a preview shows bare
wheels and a check cannot see a track cutting into a mudguard or the lower hull.

    from meshkit import vehicle
    js = vehicle.load_json("panther_g.json")
    m2 = vehicle.bake_tracks(model, js)              # copy of the model with every track link placed
    vehicle.select(m2.order, "$wheel_*,$track_*")    # glob patterns for --parts / --skip
    rep = vehicle.clash(m2, ["$track_l", "$track_r"], ["hull"])   # where the tracks run inside the hull

Command line: render / seethrough / clash take --json VEHICLE.json (tracks placed as in game), and
--parts / --skip accept glob patterns (`$wheel_*`). render --states draws several visibility states in rows.
"""
import fnmatch
import json
import math

import numpy as np

from .geom import Model, Part, rot_x

TRACK_ANGLE_CORRECTION = 90.0       # VehicleEntityRenderer.CRAWLER_TRACK_ANGLE_CORRECTION_DEGREES


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _patterns(spec):
    if spec is None:
        return []
    if isinstance(spec, str):
        spec = spec.split(",")
    return [p.strip() for p in spec if p and p.strip()]


def select(names, spec):
    """Names matching any of the comma separated glob patterns in spec (exact names work too)."""
    pats = _patterns(spec)
    return [n for n in names if any(fnmatch.fnmatchcase(n, p) for p in pats)]


def track_placements(track, phase=0.0):
    """[(y, z, angle_deg)] for one crawler_tracks entry, exactly as the mod places the link copies."""
    path = [(float(p["y"]), float(p["z"])) for p in track["path"]]
    if track.get("flip"):
        path = path[::-1]
    n = len(path)
    if n < 2:
        return []
    cum = [0.0]
    for i in range(n):
        a, b = path[i], path[(i + 1) % n]
        cum.append(cum[-1] + math.hypot(b[0] - a[0], b[1] - a[1]))
    total = cum[-1]
    if total < 1e-6:
        return []
    spacing = max(0.05, float(track.get("link_spacing", 0.5)))
    count = max(1, int(math.floor(total / spacing + 0.5)))
    out = []
    for i in range(count):
        d = ((i * spacing + phase) % total + total) % total
        k = 0
        while k < n - 1 and cum[k + 1] < d:
            k += 1
        a, b = path[k], path[(k + 1) % n]
        seg = cum[k + 1] - cum[k]
        t = (d - cum[k]) / seg if seg > 1e-9 else 0.0
        out.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t,
                    math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))))
    return out


def bake_tracks(model, js, phase=0.0):
    """Copy of `model` in which every crawler-track part holds all of its placed link copies (UVs repeated)."""
    tracks = [t for t in js.get("crawler_tracks", []) if t.get("part") in model.parts]
    if not tracks:
        return model
    import copy
    out = copy.copy(model)
    out.parts = dict(model.parts)
    out.uvs = dict(model.uvs) if model.uvs is not None else None
    for t in tracks:
        name = t["part"]
        src = model.parts[name]
        new = Part(name); new.meta = dict(src.meta)
        mirror = np.diag([-1.0, 1.0, 1.0]) if t.get("flip") else np.eye(3)
        for (y, z, ang) in track_placements(t, phase):
            M = rot_x(ang + TRACK_ANGLE_CORRECTION) @ mirror
            for si, sh in enumerate(src.shells):
                c = sh.copy().apply(M, (0.0, y, z))
                ni = len(new.shells)
                new.shells.append(c)
                if out.uvs is not None:
                    for fi in range(len(sh.faces)):
                        uv = model.uvs.get((name, si, fi))
                        if uv is not None:
                            out.uvs[(name, ni, fi)] = list(reversed(uv)) if np.linalg.det(M) < 0 else uv
        out.parts[name] = new
    return out


def clash(model, parts_a, parts_b, samples=2, min_points=3):
    """Where parts_a run INSIDE closed shells of parts_b (a track through a mudguard, a wheel in the hull...).
    Returns {part_a: {"inside": fraction, "hits": [{"part", "shell", "tags", "points", "lo", "hi"}]}}, the hits
    sorted by the number of surface sample points inside; lo/hi bound the offending points."""
    from . import attach
    shells_b = []
    for pb in parts_b:
        for si, sh in enumerate(model.parts[pb].shells):
            tri = attach._tris(sh)[1]
            shells_b.append((pb, si, sorted(set(sh.tags)), tri, tri.reshape(-1, 3).min(0), tri.reshape(-1, 3).max(0)))
    out = {}
    for pa in parts_a:
        P = np.vstack([attach._surface_points(sh, samples) for sh in model.parts[pa].shells])
        ins = np.zeros(len(P), bool)
        hits = []
        for (pb, si, tags, tri, lo, hi) in shells_b:
            sel = np.all((P >= lo - 1e-6) & (P <= hi + 1e-6), 1)
            if not sel.any():
                continue
            idx = np.where(sel)[0]
            inside = idx[attach._inside(P[idx], tri)]
            if len(inside) >= min_points:
                Q = P[inside]
                hits.append({"part": pb, "shell": si, "tags": tags, "points": int(len(inside)),
                             "lo": [round(float(v), 3) for v in Q.min(0)], "hi": [round(float(v), 3) for v in Q.max(0)]})
            ins[inside] = True
        hits.sort(key=lambda h: -h["points"])
        out[pa] = {"inside": round(float(ins.mean()), 4), "hits": hits}
    return out


def parse_states(spec, names):
    """'all;-$track_*;=hull,$wheel_*' -> [(label, skip_set, only_set_or_None)]. '-' hides parts, '=' shows only
    those, 'all' (or empty) shows everything."""
    out = []
    for raw in [s.strip() for s in spec.split(";") if s.strip()]:
        if raw in ("all", "*"):
            out.append((raw, set(), None))
        elif raw.startswith("-"):
            out.append((raw, set(select(names, raw[1:])), None))
        elif raw.startswith("="):
            out.append((raw, set(), set(select(names, raw[1:]))))
        else:
            out.append((raw, set(), set(select(names, raw))))
    return out
