"""Nakajima Kikka (橘花) - twin-jet special attack / attack aircraft, 1945 prototype. Built with meshkit only.

Sources (dimensions / layout only):
  - en.wikipedia "Nakajima Kikka": span 10.0 m, length 9.25 m, height 3.05 m, wing area 13.2 m2,
    two Ne-20 turbojets in pods slung under the wings, straight wings (no Me 262 sweep), no triangular
    fuselage section, folding wings for storage in tunnels, tricycle gear, single seat.
Compromises: root/tip chord 1.95/0.85 m (from the wing area), nacelle position (1.65 m off the centre
line, 2.8 m long, 0.68 m diameter - the Ne-20 is about 0.62 m), fold line just outboard of the nacelles
(outer panels are the moving parts $wing_outer_l/r, hinged about the Z axis), tail sizes, gear positions.
Built level; origin = main-wheel contact, z = 0 at the main axle.
"""
import math
import numpy as np
import meshkit as mk
from meshkit.texture import texture_model
import aero

SPAN, LENGTH = 10.0, 9.25
GREEN, GREY, YELLOW = (60, 74, 52), (168, 166, 152), (214, 160, 40)
RED, WHITE = (176, 30, 34), (236, 236, 230)
ROOT_LE, ROOT_C, TIP_LE, TIP_C = 6.0, 1.95, 5.72, 0.85
WING_Y, DIH = -0.32, math.radians(5)
FOLD_X = 2.62
NAC_X, NAC_Y = 1.65, -0.56
GROUND_Y, MAIN_Z = -1.62, 4.9


def station(x):
    f = abs(x) / (SPAN / 2)
    return {"x": x, "y": WING_Y + abs(x) * math.tan(DIH), "z": ROOT_LE + (TIP_LE - ROOT_LE) * f,
            "c": ROOT_C + (TIP_C - ROOT_C) * f, "t": 0.13 - 0.04 * f}


def build():
    m = mk.Model("kikka")
    st = [(0.0, 0.12, 0, 0, 0, 2), (0.5, 0.1, .17, .22, .18, 2.0), (2.0, 0.04, .36, .42, .38, 2.2),
          (4.0, 0.0, .48, .55, .50, 2.3), (5.3, 0.0, .50, .58, .52, 2.3), (6.6, -.02, .48, .52, .50, 2.2),
          (7.9, -.05, .39, .42, .42, 2.1), (8.85, -.08, .23, .24, .24, 2.0), (LENGTH, -.1, 0, 0, 0, 2)]
    m.add("fuselage", aero.body(st, n=12, tag="skin"))
    m.add("fuselage", mk.box(0, 0.62, 5.0, 0.4, 0.3, 0.1, tag="interior"))     # seat back / head armour
    m.add("fuselage", mk.box(0, 0.55, 5.95, 0.5, 0.12, 0.1, tag="interior"))   # instrument coaming
    # inner wing (one shell between the fold lines)
    m.add("wing", aero.wing_span([station(0.0), station(1.2), station(FOLD_X)], round_tips=False))
    # outer folding panels, overlapping the inner wing by 4 cm
    for s, name in ((1, "$wing_outer_l"), (-1, "$wing_outer_r")):
        sts = [station(s * (FOLD_X - 0.04)), station(s * 3.8), station(s * SPAN / 2)]
        m.add(name, aero.wing_panel(sts))
        p = station(s * FOLD_X)
        m.pivots[name] = (s * FOLD_X, p["y"] + 0.06, p["z"] - 0.35 * p["c"])
    # Ne-20 pods under the wing
    for s in (1, -1):
        cx = s * NAC_X
        ns = [(4.25, NAC_Y, 0.10, 0.10, 0.10, 2), (4.5, NAC_Y, 0.24, 0.24, 0.24, 2), (5.2, NAC_Y, 0.33, 0.35, 0.33, 2),
              (6.3, NAC_Y, 0.34, 0.36, 0.34, 2), (7.0, NAC_Y, 0.31, 0.32, 0.31, 2), (7.15, NAC_Y, 0.29, 0.30, 0.29, 2)]
        m.add("wing", aero.body_x(cx, ns, n=10, tag="skin"))
        m.add("wing", mk.cylinder((cx, NAC_Y, 7.0), (cx, NAC_Y, 7.18), 0.22, n=10, tag="intake"))     # intake face
        m.add("wing", mk.cylinder((cx, NAC_Y, 4.15), (cx, NAC_Y, 4.45), 0.16, 0.11, n=8, tag="exhaust"))  # nozzle cone
    # tail
    m.add("tail", aero.fin((0.25, 1.95, 1.65), (1.35, 0.95, 0.85), t=0.09))
    tp = [{"x": 0.0, "y": 0.22, "z": 1.75, "c": 1.15, "t": 0.09}, {"x": 1.8, "y": 0.3, "z": 1.35, "c": 0.6, "t": 0.07}]
    m.add("tail", aero.wing_span(tp))
    m.add("canopy", aero.canopy(4.55, 6.25, 0.32, 0.98, 0.34, n=12, k_front=0.45, k_rear=0.4))
    # gear
    ax_y = GROUND_Y + 0.30
    for s, name in ((1, "$gear_l"), (-1, "$gear_r")):
        top = (s * 0.95, WING_Y + 0.05, MAIN_Z + 0.15)
        m.add(name, aero.leg(top, (s * 0.95, ax_y, MAIN_Z), 0.30, 0.15, strut_r=0.06))
        m.pivots[name] = top
    nz = 7.85
    m.add("$gear_n", aero.leg((0, -0.25, nz + 0.12), (0, GROUND_Y + 0.22, nz), 0.22, 0.12, strut_r=0.05))
    m.pivots["$gear_n"] = (0, -0.25, nz + 0.12)
    m.refs += [("seat_pilot", (0, 0.28, 5.3), "seat"), ("cam_cockpit", (0, 0.85, 5.45), "camera")]
    m = m.transformed(None, (0, -GROUND_Y, -MAIN_Z))
    paint(m)
    return m


def paint(m):
    gy = -GROUND_Y; mz = -MAIN_Z

    def fn(atlas, model):
        def scheme(X, Y, Z, cur):
            out = np.empty((len(X), 3), np.float32); out[:] = GREY
            out[Y > gy - 0.05] = GREEN
            return out, None
        atlas.paint3d(scheme, tags=["skin"], kinds=("side", "front"))
        atlas.fill(["skin"], GREEN, kinds=("top",))
        atlas.fill(["skin"], GREY, kinds=("bottom",))

        def marks(X, Y, Z):
            Zd = Z - mz
            le = ROOT_LE + (TIP_LE - ROOT_LE) * np.abs(X) / (SPAN / 2)
            res = [((np.abs(X) > 0.55) & (np.abs(X) < 2.2) & (Zd < le + 0.05) & (Zd > le - 0.28) & (np.abs(Y - gy) < 0.6), YELLOW)]
            for s in (1, -1):
                xc = s * 3.75; zc = ROOT_LE + (TIP_LE - ROOT_LE) * 0.75 - 0.55 * (ROOT_C + (TIP_C - ROOT_C) * 0.75)
                r, _ = aero.hinomaru(X, Zd, xc, zc, 0.36)
                res.append((r, RED))
            res.append(((Zd > 4.6) & (Zd < 6.2) & (np.abs(X) < 0.33) & (Y - gy > 0.35), (62, 64, 56)))
            return res
        aero.paint_masks(atlas, ["skin"], ("top", "bottom"), marks)

        def side(X, Y, Z):
            red, white = aero.hinomaru(Z - mz, Y - gy, 2.9, 0.02, 0.28, white=0.05)
            return [(white, WHITE), (red, RED)]
        aero.paint_masks(atlas, ["skin"], ("side",), side)
        aero.panel_lines(atlas, ["skin"], zs=[mz + z for z in (2.0, 4.0, 6.6, 7.9)], xs=(FOLD_X,))
        atlas.paint_alpha(105, tags=["glass"])

    texture_model(m, (512, 512), colors={"glass": (150, 180, 190)},
                  flat={"tyre": (36, 36, 36), "gear": (128, 128, 124), "intake": (34, 34, 36), "exhaust": (70, 60, 52),
                        "interior": (70, 74, 62)},
                  glass_tags=("glass",), paint=fn)
