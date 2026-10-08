"""Curtiss XF15C-1 - mixed-power (piston + jet) carrier fighter prototype, 1945. Built with meshkit only.

Sources (dimensions / layout only):
  - en.wikipedia "Curtiss XF15C": span 14.63 m (48 ft), length 13.41 m (44 ft), height 4.65 m (15 ft 3 in),
    wing area 37.2 m2; R-2800 radial with a 4-blade 13 ft 1 in (3.99 m) Hamilton Standard propeller;
    J36 turbojet in the rear fuselage fed by ducts in the wing roots; tricycle gear, main gear retracting
    inwards; bubble canopy; both aircraft completed with the original (low) tail - the T-tail was only proposed.
Compromises: root/tip chord 3.3/1.6 m (from the area), fold line 3.1 m from the centre line (outer panels
$wing_outer_l/r fold up about Z), jet exhaust under the rear fuselage, tail sizes, gear positions.
The level model is 4.6 m high to the fin tip with 0.3 m propeller clearance. 4 x 20 mm wing guns shown
(never fitted on the prototypes). Colour: overall glossy sea blue.
Origin = main-wheel contact, z = 0 at the main axle.
"""
import math
import numpy as np
import meshkit as mk
from meshkit.texture import texture_model
import aero

SPAN, LENGTH, PROP_R = 14.63, 13.41, 1.995
SEA, BLUE, WHITE = (34, 46, 76), (24, 34, 92), (236, 236, 234)
ROOT_LE, ROOT_C, TIP_LE, TIP_C = 9.45, 3.3, 8.95, 1.6
WING_Y, DIH = -0.55, math.radians(5)
FOLD_X = 3.1
GROUND_Y, MAIN_Z = -2.3, 7.9


def station(x):
    f = abs(x) / (SPAN / 2)
    return {"x": x, "y": WING_Y + abs(x) * math.tan(DIH), "z": ROOT_LE + (TIP_LE - ROOT_LE) * f,
            "c": ROOT_C + (TIP_C - ROOT_C) * f, "t": 0.17 - 0.07 * f}


def build():
    m = mk.Model("xf15c")
    st = [(0.0, 0.55, 0, 0, 0, 2), (0.6, 0.5, .2, .25, .25, 2.0), (2.0, 0.35, .42, .5, .55, 2.1),
          (4.0, 0.2, .58, .7, .72, 2.2), (6.0, 0.1, .66, .78, .8, 2.3), (8.0, 0.05, .68, .8, .78, 2.3),
          (10.0, 0.0, .69, .75, .73, 2.2), (11.2, 0.0, .72, .72, .72, 2.0), (12.55, 0.0, .72, .72, .72, 2.0),
          (12.75, 0.0, .6, .6, .6, 2.0)]
    m.add("fuselage", aero.body(st, n=14, tag="skin"))
    m.add("fuselage", mk.cylinder((0, 0, 12.6), (0, 0, 12.8), 0.5, n=12, tag="engine"))        # engine face in the cowl
    # J36 exhaust under the tail
    m.add("fuselage", aero.body([(1.55, -0.12, .22, .2, .22, 2), (2.2, -0.12, .28, .26, .28, 2), (3.6, -0.02, .3, .3, .3, 2)],
                                n=10, tag="skin"))
    m.add("fuselage", mk.cylinder((0, -0.12, 1.45), (0, -0.12, 1.65), 0.17, n=10, tag="exhaust"))
    m.add("fuselage", mk.box(0, 0.98, 8.25, 0.42, 0.36, 0.12, tag="interior"))
    m.add("fuselage", mk.box(0, 0.9, 9.25, 0.5, 0.12, 0.12, tag="interior"))
    m.add("$propeller", aero.propeller((0, 0, 12.95), (0, 0, 1), PROP_R, blades=4, chord=0.26, hub_r=0.34, hub_len=0.9))
    m.pivots["$propeller"] = (0.0, 0.0, 12.95)
    # wing: centre section + folding outer panels
    m.add("wing", aero.wing_span([station(0.0), station(1.4), station(FOLD_X)], round_tips=False))
    for s, name in ((1, "$wing_outer_l"), (-1, "$wing_outer_r")):
        m.add(name, aero.wing_panel([station(s * (FOLD_X - 0.04)), station(s * 5.0), station(s * SPAN / 2)]))
        p = station(s * FOLD_X)
        m.pivots[name] = (s * FOLD_X, p["y"] + 0.08, p["z"] - 0.35 * p["c"])
        for k, x in enumerate((3.55, 3.9)):            # 20 mm guns (outer panels)
            q = station(s * x)
            p0 = np.array([s * x, q["y"], q["z"] - 0.5])
            m.add(name, mk.cylinder(p0, p0 + [0, 0, 1.0], 0.035, n=6, tag="gun"))
    # tail (original low tailplane)
    m.add("tail", aero.fin((0.55, 2.75, 2.25), (2.15, 1.55, 1.0), t=0.1))
    tp = [{"x": 0.0, "y": 0.42, "z": 2.95, "c": 1.75, "t": 0.1}, {"x": 2.8, "y": 0.5, "z": 2.45, "c": 0.9, "t": 0.08}]
    m.add("tail", aero.wing_span(tp))
    m.add("canopy", aero.canopy(7.6, 9.45, 0.6, 1.3, 0.38, n=12, k_front=0.45, k_rear=0.5))
    # gear
    ax_y = GROUND_Y + 0.42
    for s, name in ((1, "$gear_l"), (-1, "$gear_r")):
        top = (s * 2.1, station(2.1)["y"], MAIN_Z + 0.25)
        m.add(name, aero.leg(top, (s * 2.1, ax_y, MAIN_Z), 0.42, 0.22, strut_r=0.08))
        m.pivots[name] = top
    nz = 11.4
    m.add("$gear_n", aero.leg((0, -0.5, nz + 0.15), (0, GROUND_Y + 0.3, nz), 0.3, 0.16, strut_r=0.07))
    m.pivots["$gear_n"] = (0, -0.5, nz + 0.15)
    m.refs += [("seat_pilot", (0, 0.55, 8.55), "seat"), ("cam_cockpit", (0, 1.15, 8.7), "camera")]
    m.refs += [(f"muzzle_{i}", (s * x, station(x)["y"], station(x)["z"] + 0.5), "muzzle")
               for i, (s, x) in enumerate(((1, 3.55), (1, 3.9), (-1, 3.55), (-1, 3.9)))]
    m = m.transformed(None, (0, -GROUND_Y, -MAIN_Z))
    paint(m)
    return m


def paint(m):
    gy = -GROUND_Y; mz = -MAIN_Z

    def fn(atlas, model):
        atlas.fill(["skin"], SEA)

        def top(X, Y, Z):
            Zd = Z - mz
            res = []
            le = ROOT_LE + (TIP_LE - ROOT_LE) * np.abs(X) / (SPAN / 2)
            res.append(((np.abs(X) > 0.75) & (np.abs(X) < 1.6) & (Zd < le + 0.02) & (Zd > le - 0.14), (16, 18, 22)))  # duct inlets
            return res
        aero.paint_masks(atlas, ["skin"], ("top", "bottom"), top)
        # insignia: upper surface of the left wing (+X), lower surface of the right wing (-X)
        xs = 5.4; fs = xs / (SPAN / 2)
        zs = ROOT_LE + (TIP_LE - ROOT_LE) * fs - 0.5 * (ROOT_C + (TIP_C - ROOT_C) * fs)
        for sgn, kind in ((1, "top"), (-1, "bottom")):
            aero.paint_masks(atlas, ["skin"], (kind,),
                             lambda X, Y, Z, sgn=sgn: list(zip(aero.us_star(X, Z - mz, sgn * xs, zs, 0.55), (BLUE, WHITE))))

        def side(X, Y, Z):
            b, w = aero.us_star(Z - mz, Y - gy, 4.6, 0.25, 0.45)
            return [(b, BLUE), (w, WHITE)]
        aero.paint_masks(atlas, ["skin"], ("side",), side)
        aero.panel_lines(atlas, ["skin"], zs=[mz + z for z in (2.0, 4.0, 6.0, 10.0, 11.2)], xs=(1.4, FOLD_X, 5.0), k=0.75)
        atlas.paint_alpha(100, tags=["glass"])

    texture_model(m, (512, 512), colors={"glass": (150, 180, 196)},
                  flat={"prop": (30, 30, 30), "prop_tip": (220, 190, 30), "spinner": SEA, "tyre": (36, 36, 36),
                        "gear": (150, 152, 150), "gun": (45, 45, 48), "interior": (70, 76, 64), "engine": (40, 40, 42),
                        "exhaust": (60, 52, 46)},
                  glass_tags=("glass",), paint=fn)
