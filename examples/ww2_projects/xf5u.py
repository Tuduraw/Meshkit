"""Vought XF5U-1 "Flying Flapjack" - low-aspect-ratio disc-wing fighter prototype, 1945-47. meshkit only.

Sources (dimensions / layout only):
  - en.wikipedia "Vought XF5U": length 8.73 m, span 9.91 m (across the ailavators), height 4.50 m,
    wing area 44.2 m2, propellers on the leading edge at the wingtips, cockpit in a nose nacelle ahead
    of the leading edge, bubble canopy.
  - oldmachinepress.com "Vought XF5U Flying Flapjack": wing (disc) span 7.1 m, propeller diameter 16 ft
    (4.9 m), all-moving "ailavators" at the sides (8 ft 4 in each), two vertical tails with a stabilising
    surface between them, main gear track 4.9 m, retractable tailwheel, 18.7 deg ground angle,
    six .50 guns (three each side of the cockpit).
Compromises: disc planform as an ellipse-like chord law c(x) = 6.0 m * sqrt(1 - (x/4.3)^2) with a 16 %
section, 4-blade propellers (the real ones were two staggered 2-blade pairs), tip pods, fin and
tailplane sizes, single main wheels instead of twin, overall glossy sea blue scheme.
Built level (pack convention) - the tailwheel is 1.85 m up so the 18.7 deg ground attitude results
when the nose is raised. Origin = main-wheel contact, z = 0 at the main axle.
"""
import math
import numpy as np
import meshkit as mk
from meshkit.texture import texture_model
import aero

DISC, WIDTH, LENGTH, PROP_R = 7.1, 9.91, 8.73, 2.45
SEA, BLUE, WHITE = (34, 46, 76), (24, 34, 92), (236, 236, 234)
C0, LE0, XK = 6.0, 6.0, 4.3
PROP_X = 3.15
HUB_Z = LE0 + 0.3         # propeller plane ahead of the whole leading edge
GROUND_Y, MAIN_Z = -2.55, 4.75


def chord(x):
    return C0 * math.sqrt(max(0.0, 1 - (abs(x) / XK) ** 2))


def le(x):
    return LE0 - (C0 - chord(x)) * 0.75


def station(x):
    return {"x": x, "y": 0.0, "z": le(x), "c": chord(x), "t": 0.16}


def build():
    m = mk.Model("xf5u")
    # disc wing: one shell across the span
    m.add("wing", aero.wing_span([station(x) for x in (0.0, 0.9, 1.8, 2.5, 3.0, DISC / 2)]))
    # nose nacelle with the cockpit, running back into the wing
    st = [(3.2, -0.02, .45, .42, .4, 2.2), (5.6, 0.0, .54, .6, .5, 2.3), (6.6, 0.0, .5, .55, .45, 2.2),
          (7.4, -0.04, .36, .36, .32, 2.0), (7.82, -0.06, .14, .14, .13, 2.0), (7.9, -0.06, 0, 0, 0, 2)]
    m.add("wing", aero.body(st, n=12, tag="skin"))
    m.add("canopy", aero.canopy(5.6, 7.15, 0.42, 1.08, 0.36, n=12, k_front=0.45, k_rear=0.45))
    m.add("wing", mk.box(0, 0.62, 5.95, 0.4, 0.36, 0.12, tag="interior"))
    m.add("wing", mk.box(0, 0.58, 6.85, 0.48, 0.12, 0.1, tag="interior"))
    # .50 guns, three each side of the cockpit (muzzles in the leading edge)
    for s in (1, -1):
        for k in range(3):
            x = s * (0.75 + 0.14 * k); p0 = np.array([x, -0.05, le(x) - 0.7])
            m.add("wing", mk.cylinder(p0, p0 + [0, 0, 0.9], 0.03, n=6, tag="gun"))
    # propeller pods at the tips and the propellers (counter-rotating)
    for s, name in ((1, "$propeller_l"), (-1, "$propeller_r")):
        x = s * PROP_X
        pod = [(le(x) - 1.6, 0.0, .28, .3, .3, 2), (le(x) - 0.2, 0.0, .36, .38, .38, 2), (HUB_Z - 0.6, 0.0, .3, .3, .3, 2),
               (HUB_Z - 0.2, 0.0, .26, .26, .26, 2)]
        m.add("wing", aero.body_x(x, pod, n=10, tag="skin"))
        hub = (x, 0.0, HUB_Z)
        m.add(name, aero.propeller(hub, (0, 0, 1), PROP_R, blades=4, chord=0.42, hub_r=0.32, hub_len=0.8,
                                   phase=math.radians(45), twist=(s * 28.0, s * 10.0)))
        m.pivots[name] = hub
    # ailavators: all-moving surfaces on the trailing-edge corners, hinged spanwise
    for s, name in ((1, "$ailavator_l"), (-1, "$ailavator_r")):
        st_a = [{"x": s * 2.55, "y": 0.0, "z": 0.95, "c": 1.83, "t": 0.08}, {"x": s * WIDTH / 2, "y": 0.0, "z": 0.75, "c": 1.55, "t": 0.07}]
        m.add(name, aero.wing_panel(st_a))
        m.pivots[name] = (s * 2.55, 0.0, 0.95 - 0.3 * 1.83)
    # twin fins and the stabiliser between them
    for s in (1, -1):
        m.add("tail", aero.fin((-0.05, 1.65, 2.05), (1.95, 0.55, 1.15), x=s * 1.3, t=0.09))
    m.add("tail", aero.wing_span([{"x": 0.0, "y": 0.34, "z": 0.45, "c": 1.1, "t": 0.08},
                                  {"x": 1.42, "y": 0.34, "z": 0.45, "c": 1.0, "t": 0.08}], round_tips=False))
    # gear: long main legs under the pods, short tailwheel
    ax_y = GROUND_Y + 0.45
    for s, name in ((1, "$gear_l"), (-1, "$gear_r")):
        top = (s * 2.45, -0.05, MAIN_Z + 0.25)
        m.add(name, aero.leg(top, (s * 2.45, ax_y, MAIN_Z), 0.45, 0.24, strut_r=0.09))
        m.pivots[name] = top
    tz = 1.0
    m.add("$gear_t", aero.leg((0, 0.0, tz + 0.2), (0, GROUND_Y + 1.85 + 0.16, tz), 0.16, 0.1, strut_r=0.05))
    m.pivots["$gear_t"] = (0, 0.0, tz + 0.2)
    m.refs += [("seat_pilot", (0, 0.28, 6.25), "seat"), ("cam_cockpit", (0, 0.92, 6.4), "camera")]
    m = m.transformed(None, (0, -GROUND_Y, -MAIN_Z))
    paint(m)
    return m


def paint(m):
    gy = -GROUND_Y; mz = -MAIN_Z

    def fn(atlas, model):
        atlas.fill(["skin"], SEA)
        for sgn, kind in ((1, "top"), (-1, "bottom")):
            aero.paint_masks(atlas, ["skin"], (kind,),
                             lambda X, Y, Z, sgn=sgn: list(zip(aero.us_star(X, Z - mz, sgn * 1.9, 2.4, 0.5), (BLUE, WHITE))))

        def side(X, Y, Z):
            b, w = aero.us_star(Z - mz, Y - gy, 4.4, 0.05, 0.3, bars=False)
            return [(b, BLUE), (w, WHITE)]
        aero.paint_masks(atlas, ["skin"], ("side",), side)
        aero.panel_lines(atlas, ["skin"], zs=[mz + z for z in (1.5, 3.0, 4.5)], xs=(0.9, 1.8, 2.5), k=0.75,
                         kinds=("top", "bottom"))
        atlas.paint_alpha(100, tags=["glass"])

    texture_model(m, (512, 512), colors={"glass": (150, 180, 196)},
                  flat={"prop": (120, 92, 60), "prop_tip": (220, 190, 30), "spinner": (30, 30, 30), "tyre": (36, 36, 36),
                        "gear": (150, 152, 150), "gun": (45, 45, 48), "interior": (70, 76, 64)},
                  glass_tags=("glass",), paint=fn)
