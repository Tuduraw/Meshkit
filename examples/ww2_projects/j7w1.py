"""Kyushu J7W1 Shinden (震電) - canard pusher interceptor, 1945 prototype. Built with meshkit only.

Sources (dimensions / layout only):
  - en.wikipedia "Kyushu J7W Shinden": span 11.114 m, length 9.66 m, height 3.92 m, wing area 20.5 m2,
    6-blade pusher propeller, Ha-43 radial behind the cockpit, long narrow oblique intakes on the fuselage
    sides, 4 x 30 mm Type 5 cannon in the nose.
  - ja.wikipedia 「震電」: propeller diameter 3.4 m, 20 deg leading-edge sweep, fins on the main wing with
    small wheels under them, retractable tricycle gear, 5 deg nose-up on the ground, guns 3 deg up.
Compromises (no three-view measured): chord distribution from the wing area (root 2.6 m / tip 1.15 m),
fin position (40 % semi-span) and size, canard span 3.0 m, gear positions, intake shape, fuselage
cross-sections. The model is built level (pack convention); all wheels touch y = 0 when level, i.e. the
5 deg nose-up ground attitude is not modelled.
Axes: +X left, +Y up, +Z forward; origin = main-wheel contact point, z = 0 at the main axle.
"""
import math
import numpy as np
import meshkit as mk
from meshkit.texture import texture_model
import aero

SPAN, LENGTH, PROP_R = 11.114, 9.66, 1.7
T20 = math.tan(math.radians(20))
GREEN, GREY, YELLOW = (58, 72, 50), (166, 164, 150), (214, 160, 40)
RED, WHITE = (176, 30, 34), (236, 236, 230)
# design frame: z from the propeller (0) to the nose tip (9.66), y = 0 on the thrust line
ROOT_LE, ROOT_C, TIP_C = 3.6, 2.6, 1.15
GROUND_Y, MAIN_Z = -2.1, 2.3


def wing_le(x):
    return ROOT_LE - abs(x) * T20


def build():
    m = mk.Model("j7w1")
    # ---- fuselage
    st = [(0.55, 0, .30, .30, .30, 2.0), (1.2, 0, .44, .46, .44, 2.2), (2.2, 0, .55, .62, .58, 2.3),
          (3.4, 0, .58, .66, .60, 2.3), (4.4, 0, .55, .68, .58, 2.3), (5.6, -.02, .50, .55, .52, 2.2),
          (7.0, -.04, .42, .44, .44, 2.1), (8.4, -.06, .30, .31, .31, 2.0), (9.3, -.07, .15, .15, .15, 2.0),
          (LENGTH, -.08, 0, 0, 0, 2)]
    m.add("fuselage", aero.body(st, n=12, tag="skin"))
    # cooling intakes: long, narrow, oblique slots ahead of the wing roots (sunk 3 cm into the skin)
    for s in (1, -1):
        M = mk.rot_axis((1, 0, 0), -14) @ mk.rot_axis((0, 1, 0), s * 4)
        m.add("fuselage", mk.box(s * 0.555, -0.12, 4.1, 0.09, 0.34, 1.3, tag="intake", M=M))
    # 4 x 30 mm cannon muzzles in the nose, 3 deg up
    for sx in (1, -1):
        for sy in (1, -1):
            p0 = np.array([sx * 0.11, -0.08 + sy * 0.07, 8.9]); d = np.array([0, math.sin(math.radians(3)), 1.0])
            m.add("fuselage", mk.cylinder(p0, p0 + d * 0.84, 0.035, n=6, tag="gun"))
    # cockpit interior seen through the canopy: seat back + headrest, instrument coaming
    m.add("fuselage", mk.box(0, 0.78, 4.35, 0.42, 0.34, 0.12, tag="interior"))
    m.add("fuselage", mk.box(0, 0.72, 5.25, 0.5, 0.14, 0.12, tag="interior"))
    # ---- main wing (one shell across the span), 20 deg LE sweep, slight dihedral
    half = []
    for f in (0.0, 0.18, 0.5, 1.0):
        x = f * SPAN / 2
        half.append({"x": x, "y": -0.22 + x * math.tan(math.radians(2.5)), "z": wing_le(x),
                     "c": ROOT_C + (TIP_C - ROOT_C) * f, "t": 0.14 - 0.05 * f})
    m.add("wing", aero.wing_span(half, tag="skin"))
    # fins on the wing at 40 % semi-span, reaching below it with a small wheel at the foot
    for s in (1, -1):
        x = s * 0.40 * SPAN / 2
        le = wing_le(x)
        rings = []
        for (y, dz, c, t) in ((-0.82, -0.75, 0.95, 0.06), (-0.70, -0.55, 1.25, 0.09), (-0.1, -0.42, 1.55, 0.10),
                              (1.0, -1.05, 1.0, 0.09), (1.5, -1.4, 0.7, 0.07), (1.62, -1.5, 0.45, 0.05)):
            rings.append(mk.wing_section(x, y, le + dz, c, t, span_axis=1))
        m.add("wing", mk.loft(rings, tag="skin"))
        m.add("wing", aero.wheel((x, -0.92, le - 1.15), 0.14, 0.08, n=8))
        m.add("wing", mk.cylinder((x - 0.05, -0.92, le - 1.15), (x + 0.05, -0.92, le - 1.15), 0.035, n=6, tag="gear"))
    # ---- canard
    ch = [{"x": 0.0, "y": -0.08, "z": 8.65, "c": 0.78, "t": 0.11}, {"x": 1.5, "y": -0.06, "z": 8.45, "c": 0.48, "t": 0.09}]
    m.add("canard", aero.wing_span(ch, tag="skin"))
    # ---- canopy (bubble faired into the spine)
    m.add("canopy", aero.canopy(3.85, 5.6, 0.48, 1.12, 0.36, n=12, k_front=0.4, k_rear=0.45))
    # ---- six-blade pusher propeller (thrust towards +Z)
    m.add("$propeller", aero.propeller((0, 0, 0.42), (0, 0, 1), PROP_R, blades=6, chord=0.24, hub_r=0.3,
                                       hub_len=0.75, phase=math.radians(15)))
    m.pivots["$propeller"] = (0.0, 0.0, 0.42)
    # ---- landing gear (extended)
    ax_y = GROUND_Y + 0.33
    for s, name in ((1, "$gear_l"), (-1, "$gear_r")):
        top = (s * 1.55, -0.2, MAIN_Z + 0.25)
        m.add(name, aero.leg(top, (s * 1.55, ax_y, MAIN_Z), 0.33, 0.17, strut_r=0.07, side=s * 0.0))
        m.pivots[name] = top
    nz = 7.9; ny = GROUND_Y + 0.25
    m.add("$gear_n", aero.leg((0, -0.25, nz + 0.15), (0, ny, nz), 0.25, 0.14, strut_r=0.06))
    m.pivots["$gear_n"] = (0, -0.25, nz + 0.15)
    # ---- reference points
    m.refs += [("seat_pilot", (0, 0.45, 4.6), "seat"), ("cam_cockpit", (0, 1.0, 4.75), "camera")]
    m.refs += [(f"muzzle_{i}", (sx * 0.11, -0.08 + sy * 0.07 + 0.04, 9.74), "muzzle")
               for i, (sx, sy) in enumerate(((1, 1), (-1, 1), (1, -1), (-1, -1)))]
    # ---- move to the pack origin: main contact at y = 0, main axle at z = 0
    m = m.transformed(None, (0, -GROUND_Y, -MAIN_Z))
    paint(m)
    return m


def paint(m):
    gy = -GROUND_Y; mz = -MAIN_Z     # design -> model offsets

    def fn(atlas, model):
        def scheme(X, Y, Z, cur):
            out = np.empty((len(X), 3), np.float32)
            up = Y > gy - 0.05
            out[:] = GREY
            out[up] = GREEN
            return out, None
        atlas.paint3d(scheme, tags=["skin"], kinds=("side", "front"))
        atlas.fill(["skin"], GREEN, kinds=("top",))
        atlas.fill(["skin"], GREY, kinds=("bottom",))

        def marks(X, Y, Z):
            Zd = Z - mz; Yd = Y - gy
            res = []
            # yellow IFF band on the inner leading edges (top and bottom)
            le = ROOT_LE - np.abs(X) * T20
            res.append(((np.abs(X) > 0.62) & (np.abs(X) < 2.0) & (Zd < le + 0.05) & (Zd > le - 0.3) & (np.abs(Yd + 0.2) < 0.25), YELLOW))
            # wing hinomaru, red without border, at 70 % semi-span
            xc = 0.70 * SPAN / 2
            zc = ROOT_LE - xc * T20 - 0.55 * (ROOT_C + (TIP_C - ROOT_C) * 0.7)
            for s in (1, -1):
                r, _ = aero.hinomaru(X, Zd, s * xc, zc, 0.42)
                res.append((r & (np.abs(X) > 2.6), RED))
            # cockpit floor under the canopy
            res.append(((Zd > 3.9) & (Zd < 5.55) & (np.abs(X) < 0.34) & (Yd > 0.5), (62, 64, 56)))
            return res
        aero.paint_masks(atlas, ["skin"], ("top", "bottom"), marks)

        def side(X, Y, Z):
            Zd = Z - mz; Yd = Y - gy
            red, white = aero.hinomaru(Zd, Yd, 2.6, 0.0, 0.32, white=0.06)
            return [(white, WHITE), (red, RED)]
        aero.paint_masks(atlas, ["skin"], ("side",), side)
        aero.panel_lines(atlas, ["skin"], zs=[mz + z for z in (1.2, 2.2, 3.4, 5.9, 7.2, 8.4)])
        atlas.paint_alpha(105, tags=["glass"])

    texture_model(m, (512, 512), colors={"glass": (150, 180, 190)},
                  flat={"prop": (112, 78, 46), "prop_tip": (210, 170, 40), "spinner": (112, 78, 46), "tyre": (36, 36, 36),
                        "gear": (128, 128, 124), "gun": (45, 45, 48), "intake": (28, 30, 28), "interior": (70, 74, 62)},
                  glass_tags=("glass",), paint=fn)
