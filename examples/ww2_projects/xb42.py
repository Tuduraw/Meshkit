"""Douglas XB-42 Mixmaster - pusher attack bomber prototype, 1944 (first prototype, twin canopies). meshkit only.

Sources (dimensions / layout only):
  - en.wikipedia "Douglas XB-42 Mixmaster": span 21.49 m, length 16.36 m, height 5.74 m, wing area 51.6 m2;
    two Allison V-1710 in the fuselage driving contra-rotating pusher propellers at the tail; full cruciform
    tail with a ventral fin and rudder carrying a tailwheel to protect the propellers; twin bubble canopies
    (pilot and co-pilot side by side) on the first prototype; bombardier behind a plexiglass nose;
    air intakes in the wing leading edges; tricycle gear; 2 fixed forward guns, remote twin-gun turrets
    in the wing trailing edges (retracted - not modelled), 8,000 lb internal bomb bay.
Compromises: propeller diameter 4.0 m (two 3-blade props, 0.45 m apart), root/tip chord 3.6/1.4 m
(from the area), fuselage sections, tail sizes, gear positions, panel / bomb-bay lines.
Colour: natural metal with an olive drab anti-glare panel. Built level; origin = main-wheel contact,
z = 0 at the main axle.
"""
import math
import numpy as np
import meshkit as mk
from meshkit.texture import texture_model
import aero

SPAN, LENGTH, PROP_R = 21.49, 16.36, 2.0
METAL, OD, BLUE, WHITE = (186, 190, 194), (82, 84, 58), (30, 44, 104), (238, 238, 236)
ROOT_LE, ROOT_C, TIP_LE, TIP_C = 9.65, 3.6, 8.95, 1.4
WING_Y, DIH = -0.42, math.radians(4)
GROUND_Y, MAIN_Z = -3.1, 7.9


def station(x):
    f = abs(x) / (SPAN / 2)
    return {"x": x, "y": WING_Y + abs(x) * math.tan(DIH), "z": ROOT_LE + (TIP_LE - ROOT_LE) * f,
            "c": ROOT_C + (TIP_C - ROOT_C) * f, "t": 0.16 - 0.06 * f}


def build():
    m = mk.Model("xb42")
    st = [(1.25, 0, .36, .38, .38, 2), (2.5, 0, .56, .62, .60, 2.1), (4.5, 0.03, .74, .86, .80, 2.2),
          (7.0, 0.05, .82, 1.0, .9, 2.3), (9.5, 0.05, .83, 1.02, .92, 2.3), (11.5, 0.04, .80, .98, .90, 2.3),
          (13.5, 0.0, .70, .82, .80, 2.2), (14.6, -0.04, .58, .64, .64, 2.1), (15.2, -0.06, .5, .54, .54, 2.0)]
    m.add("fuselage", aero.body(st, n=14, tag="skin"))
    # plexiglass nose and the bombardier's seat behind it
    gn = [(15.0, -0.06, .53, .57, .57, 2.0), (15.6, -0.08, .44, .46, .46, 2.0), (16.05, -0.1, .27, .27, .27, 2.0),
          (LENGTH, -0.11, 0, 0, 0, 2)]
    m.add("canopy", aero.body(gn, n=14, tag="glass"))
    m.add("fuselage", mk.box(0, -0.15, 15.3, 0.45, 0.4, 0.2, tag="interior"))
    # twin bubble canopies, pilot (left) and co-pilot (right)
    for s in (1, -1):
        m.add("canopy", aero.canopy(10.9, 12.55, 0.78, 1.42, 0.36, n=12, k_front=0.4, k_rear=0.4, cx=s * 0.4))
        m.add("fuselage", mk.box(s * 0.4, 1.1, 11.2, 0.42, 0.42, 0.12, tag="interior"))
    # fixed forward guns low in the nose
    for s in (1, -1):
        p0 = np.array([s * 0.36, -0.45, 14.2])
        m.add("fuselage", mk.cylinder(p0, p0 + [0, 0, 1.0], 0.04, n=6, tag="gun"))
    # wing
    half = [station(0.0), station(2.2), station(6.5), station(SPAN / 2)]
    m.add("wing", aero.wing_span(half))
    # cruciform tail (all trailing edges kept 0.3 m ahead of the front propeller)
    m.add("tail", aero.fin((0.6, 4.25, 2.6), (2.45, 3.0, 1.3), t=0.1))
    rings = [mk.wing_section(0, y, z, c, 0.1, span_axis=1) for (y, z, c) in
             ((-2.55, 3.05, 0.75), (-2.45, 3.2, 1.1), (-0.5, 3.85, 2.25))]
    m.add("tail", mk.loft(rings, tag="skin"))
    m.add("tail", aero.wheel((0, -2.6, 2.45), 0.2, 0.12, n=8))
    m.add("tail", mk.cylinder((-0.08, -2.6, 2.45), (0.08, -2.6, 2.45), 0.05, n=6, tag="gear"))
    tp = [{"x": 0.0, "y": 0.05, "z": 3.45, "c": 1.9, "t": 0.1}, {"x": 3.3, "y": 0.12, "z": 2.95, "c": 1.0, "t": 0.08}]
    m.add("tail", aero.wing_span(tp))
    # contra-rotating 3-blade pushers
    m.add("$prop_front", aero.propeller((0, 0, 1.05), (0, 0, 1), PROP_R, blades=3, chord=0.3, hub_r=0.36,
                                        hub_len=0.5, spinner=False))
    m.add("$prop_front", mk.cylinder((0, 0, 0.8), (0, 0, 1.35), 0.36, 0.37, n=12, tag="spinner"))
    m.pivots["$prop_front"] = (0.0, 0.0, 1.05)
    m.add("$prop_rear", aero.propeller((0, 0, 0.55), (0, 0, 1), PROP_R, blades=3, chord=0.3, hub_r=0.34,
                                       hub_len=0.9, phase=math.radians(60), twist=(-30.0, -10.0)))
    m.pivots["$prop_rear"] = (0.0, 0.0, 0.55)
    # gear
    ax_y = GROUND_Y + 0.5
    for s, name in ((1, "$gear_l"), (-1, "$gear_r")):
        top = (s * 2.6, station(2.6)["y"], MAIN_Z + 0.3)
        m.add(name, aero.leg(top, (s * 2.6, ax_y, MAIN_Z), 0.5, 0.26, strut_r=0.1))
        m.pivots[name] = top
    nz = 13.9
    m.add("$gear_n", aero.leg((0, -0.6, nz + 0.2), (0, GROUND_Y + 0.38, nz), 0.38, 0.2, strut_r=0.08))
    m.pivots["$gear_n"] = (0, -0.6, nz + 0.2)
    m.refs += [("seat_pilot", (0.4, 0.75, 11.45), "seat"), ("seat_copilot", (-0.4, 0.75, 11.45), "seat"),
               ("seat_bombardier", (0, -0.5, 15.0), "seat"), ("cam_cockpit", (0.4, 1.3, 11.6), "camera")]
    m.refs += [(f"muzzle_{i}", (s * 0.36, -0.45, 15.2), "muzzle") for i, s in enumerate((1, -1))]
    m = m.transformed(None, (0, -GROUND_Y, -MAIN_Z))
    paint(m)
    return m


def paint(m):
    gy = -GROUND_Y; mz = -MAIN_Z

    def fn(atlas, model):
        atlas.fill(["skin"], METAL)

        def top(X, Y, Z):
            Zd = Z - mz
            res = [((Zd > 12.3) & (Zd < 15.1) & (np.abs(X) < 0.5) & (Y - gy > 0.3), OD)]     # anti-glare panel
            le = ROOT_LE + (TIP_LE - ROOT_LE) * np.abs(X) / (SPAN / 2)
            res.append(((np.abs(X) > 0.95) & (np.abs(X) < 2.1) & (Zd < le + 0.02) & (Zd > le - 0.12), (40, 40, 42)))
            # insignia: upper left wing (+X) and lower right wing (-X)
            # bomb-bay doors under the wing centre section
            res.append(((np.abs(X) < 0.5) & (Y - gy < -0.6) & ((np.abs(Zd - 6.0) < 0.012) | (np.abs(Zd - 9.6) < 0.012) |
                                                             (np.abs(X) < 0.012)), (120, 124, 128)))
            return res
        aero.paint_masks(atlas, ["skin"], ("top", "bottom"), top)
        # insignia: upper surface of the left wing (+X), lower surface of the right wing (-X)
        xs = 7.6; fs = xs / (SPAN / 2)
        zs = ROOT_LE + (TIP_LE - ROOT_LE) * fs - 0.5 * (ROOT_C + (TIP_C - ROOT_C) * fs)
        for sgn, kind in ((1, "top"), (-1, "bottom")):
            aero.paint_masks(atlas, ["skin"], (kind,),
                             lambda X, Y, Z, sgn=sgn: list(zip(aero.us_star(X, Z - mz, sgn * xs, zs, 0.62), (BLUE, WHITE))))

        def side(X, Y, Z):
            b, w = aero.us_star(Z - mz, Y - gy, 5.6, 0.1, 0.5, rot=90)
            return [(b, BLUE), (w, WHITE)]
        aero.paint_masks(atlas, ["skin"], ("side",), side)
        aero.panel_lines(atlas, ["skin"], zs=[mz + z for z in (2.5, 4.5, 7.0, 11.5, 13.5)], xs=(2.2, 6.5), k=0.86)

        # glazing: metal frames opaque, panes translucent
        def frames(X, Y, Z):
            Zd = Z - mz
            f = (np.abs(((Zd - 15.0) / 0.42) - np.round((Zd - 15.0) / 0.42)) < 0.06) & (Zd > 15.0)
            f |= (np.abs(X) < 0.025) & (Zd > 15.0)
            return [(f, METAL)]
        aero.paint_masks(atlas, ["glass"], ("top", "bottom", "side", "front"), frames)

        def alpha(X, Y, Z):
            Zd = Z - mz
            f = (np.abs(((Zd - 15.0) / 0.42) - np.round((Zd - 15.0) / 0.42)) < 0.06) & (Zd > 15.0)
            f |= (np.abs(X) < 0.025) & (Zd > 15.0)
            return np.where(f, 255.0, 100.0)
        atlas.paint_alpha(alpha, tags=["glass"])

    texture_model(m, (1024, 1024), colors={"glass": (160, 190, 200)},
                  flat={"prop": (30, 30, 30), "prop_tip": (220, 190, 30), "spinner": (186, 190, 194), "tyre": (36, 36, 36),
                        "gear": (150, 152, 150), "gun": (45, 45, 48), "interior": (84, 88, 70)},
                  glass_tags=("glass",), paint=fn)
