"""Mitsubishi J8M1 Shusui (秋水) - rocket interceptor (Me 163B derived), 1945 prototype. meshkit only.

Sources (dimensions / layout only):
  - ja.wikipedia 「秋水」: span 9.5 m, length 5.95-6.05 m, height 2.7 m; no wind-driven generator in the
    nose (the nose is lengthened for the radio and battery instead), wooden wing about 10 cm longer each
    side than the Me 163, tailless layout kept, take-off on a jettisonable dolly, landing on a skid.
  - en.wikipedia "Mitsubishi J8M": two 30 mm cannon (wing roots), wooden fin, Toko Ro.2 rocket motor.
Compromises: Me 163-like planform (23 deg LE sweep, root/tip chord 2.75/1.2 m, 4 deg washout), fin size,
canopy shape, skid / tailwheel / dolly geometry. Colour: the overall orange-yellow of Japanese prototypes.
Built level on the take-off dolly ($dolly is a separate part: drop it after take-off); origin = dolly
wheel contact, z = 0 at the dolly axle.
"""
import math
import numpy as np
import meshkit as mk
from meshkit.texture import texture_model
import aero

SPAN, LENGTH = 9.5, 6.05
ORANGE, RED, WHITE = (226, 128, 40), (176, 30, 34), (236, 236, 230)
ROOT_LE, ROOT_C, TIP_C = 4.05, 2.75, 1.2
T23 = math.tan(math.radians(23))
GROUND_Y, MAIN_Z = -0.95, 3.25


def le(x):
    return ROOT_LE - abs(x) * T23


def build():
    m = mk.Model("j8m1")
    st = [(0.22, 0.06, .17, .19, .17, 2.0), (0.9, 0.05, .36, .43, .40, 2.1), (2.1, 0.02, .49, .60, .55, 2.2),
          (3.3, 0.0, .52, .63, .58, 2.2), (4.3, 0.0, .49, .57, .54, 2.2), (5.2, -.01, .44, .48, .47, 2.1),
          (5.7, -.02, .35, .37, .37, 2.0), (5.93, -.03, .24, .25, .25, 2.0), (6.02, -.03, .12, .13, .13, 2.0),
          (LENGTH, -.03, 0, 0, 0, 2)]
    m.add("fuselage", aero.body(st, n=12, tag="skin"))
    m.add("fuselage", mk.cylinder((0, 0.06, 0.0), (0, 0.06, 0.3), 0.12, n=10, tag="nozzle"))      # rocket nozzle
    m.add("fuselage", mk.box(0, 0.55, 4.0, 0.38, 0.28, 0.1, tag="interior"))
    m.add("fuselage", mk.box(0, 0.5, 4.75, 0.44, 0.1, 0.1, tag="interior"))
    # landing skid (retracted against the belly) and tailwheel
    m.add("fuselage", mk.box(0, -0.6, 3.4, 0.14, 0.08, 1.9, tag="gear"))
    m.add("fuselage", aero.leg((0, -0.3, 0.9), (0, -0.58, 0.75), 0.13, 0.08, strut_r=0.035))
    # wing: one shell, washout towards the tips
    half = []
    for f in (0.0, 0.12, 0.55, 1.0):
        x = f * SPAN / 2
        half.append({"x": x, "y": -0.16 + x * math.tan(math.radians(1)), "z": le(x),
                     "c": ROOT_C + (TIP_C - ROOT_C) * f, "t": 0.14 - 0.05 * f, "twist": -4.0 * f})
    m.add("wing", aero.wing_span(half))
    # 30 mm cannon in the wing roots
    for s in (1, -1):
        p0 = np.array([s * 0.72, -0.16, le(0.72) - 0.6])
        m.add("wing", mk.cylinder(p0, p0 + [0, 0, 0.82], 0.035, n=6, tag="gun"))
    # fin (wooden), swept
    m.add("tail", aero.fin((0.35, 2.35, 1.95), (1.6, 0.95, 0.85), t=0.1))
    m.add("canopy", aero.canopy(3.85, 4.95, 0.36, 0.86, 0.33, n=12, k_front=0.5, k_rear=0.5))
    # take-off dolly: axle, two wheels, two struts up into the belly
    ay = GROUND_Y + 0.33
    m.add("$dolly", mk.cylinder((-0.62, ay, MAIN_Z), (0.62, ay, MAIN_Z), 0.05, n=6, tag="gear"))
    for s in (1, -1):
        m.add("$dolly", aero.wheel((s * 0.58, ay, MAIN_Z), 0.33, 0.16))
        m.add("$dolly", aero.strut((s * 0.22, ay, MAIN_Z), (s * 0.12, -0.45, MAIN_Z + 0.1), 0.045))
    m.pivots["$dolly"] = (0.0, ay, MAIN_Z)
    m.refs += [("seat_pilot", (0, 0.22, 4.2), "seat"), ("cam_cockpit", (0, 0.75, 4.35), "camera")]
    m.refs += [(f"muzzle_{i}", (s * 0.72, -0.16, le(0.72) + 0.22), "muzzle") for i, s in enumerate((1, -1))]
    m = m.transformed(None, (0, -GROUND_Y, -MAIN_Z))
    paint(m)
    return m


def paint(m):
    gy = -GROUND_Y; mz = -MAIN_Z

    def fn(atlas, model):
        atlas.fill(["skin"], ORANGE)

        def marks(X, Y, Z):
            Zd = Z - mz
            res = []
            for s in (1, -1):
                xc = s * 3.3; zc = le(3.3) - 0.5 * (ROOT_C + (TIP_C - ROOT_C) * 3.3 / (SPAN / 2))
                red, white = aero.hinomaru(X, Zd, xc, zc, 0.38, white=0.06)
                res += [(white, WHITE), (red, RED)]
            res.append(((Zd > 3.9) & (Zd < 4.9) & (np.abs(X) < 0.3) & (Y - gy > 0.3), (62, 64, 56)))
            # elevon hinge lines (outer 55 % of the span)
            te = le(X) - (ROOT_C + (TIP_C - ROOT_C) * np.abs(X) / (SPAN / 2))
            res.append(((np.abs(X) > 1.9) & (np.abs(Zd - (te + 0.55)) < 0.015), (170, 94, 30)))
            return res
        aero.paint_masks(atlas, ["skin"], ("top", "bottom"), marks)

        def side(X, Y, Z):
            red, white = aero.hinomaru(Z - mz, Y - gy, 2.0, 0.04, 0.3, white=0.06)
            return [(white, WHITE), (red, RED)]
        aero.paint_masks(atlas, ["skin"], ("side",), side)
        aero.panel_lines(atlas, ["skin"], zs=[mz + z for z in (0.9, 2.1, 5.25)])
        atlas.paint_alpha(105, tags=["glass"])

    texture_model(m, (512, 512), colors={"glass": (150, 180, 190)},
                  flat={"tyre": (36, 36, 36), "gear": (110, 110, 106), "nozzle": (40, 38, 36), "gun": (45, 45, 48),
                        "interior": (70, 74, 62)},
                  glass_tags=("glass",), paint=fn)
