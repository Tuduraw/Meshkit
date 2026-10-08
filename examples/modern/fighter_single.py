"""Modern single-engine fighter (F-16C class) built only from published numbers with meshkit 0.3.

Published data used: length 15.06 m, span 9.45 m (without tip missiles), wing area 27.87 m2, LE sweep 40 deg,
height 5.09 m, M61 in the left wing root.  Taper 0.21 and the section shapes are estimates.

Shows: planform.from_published (straight trapezoid, clipped tips), fuselage.body_with_tub (cockpit recess),
gear.stow (angle + shift fitting, fairings, 0 % outside), detail passes (panel lines, ink, weathering),
check.compare_size / marking_spot.
"""
import meshkit as mk
from meshkit import planform as pf, fuselage as fu, gear, detail as dt
from meshkit.texture import texture_model

L = 15.06
# nose at z = +L/2, tail at -L/2;  stations (z, cy, hw, ht, hb, p)
S = lambda s: L / 2 - s
FUS = [(S(0.0), 1.95, 0, 0, 0, 2), (S(0.6), 1.97, 0.2, 0.22, 0.22, 2), (S(1.6), 2.05, 0.42, 0.42, 0.40, 2),
       (S(2.6), 2.1, 0.52, 0.52, 0.55, 2.2), (S(4.0), 2.05, 0.62, 0.50, 0.72, 2.4), (S(5.5), 2.0, 0.78, 0.50, 0.82, 2.8),
       (S(7.0), 2.0, 0.92, 0.55, 0.88, 3.0), (S(9.0), 2.0, 1.00, 0.56, 0.86, 3.0), (S(11.0), 2.05, 0.86, 0.60, 0.72, 2.8),
       (S(13.0), 2.1, 0.70, 0.62, 0.62, 2.4), (S(14.6), 2.1, 0.58, 0.56, 0.56, 2.0)]
NOZZLE = (0.0, 2.1, S(15.03), 0.55)
CANOPY = (S(5.5), S(2.3), 2.38, 3.25, 0.42)            # z_rear, z_front, y_base, y_top, half width


def build():
    m = mk.Model("fighter_single")
    stations = fu.cover_nozzles(FUS, [NOZZLE], length=1.0)
    floor = CANOPY[2] - 0.55
    m.add("fuselage", fu.body_with_tub(stations, [(CANOPY[1] - 0.3, CANOPY[0] + 0.15, CANOPY[4] * 0.95, floor)]))
    m.add("fuselage", fu.pod(0.0, [(S(3.85), 1.08, 0.48, 0.38, 0.34, 3), (S(4.4), 1.05, 0.52, 0.42, 0.36, 3),
                                   (S(6.0), 1.15, 0.55, 0.45, 0.38, 3), (S(8.5), 1.45, 0.55, 0.4, 0.4, 3),
                                   (S(10.5), 1.65, 0.45, 0.3, 0.3, 2.6)]))
    m.add("fuselage", fu.intake_face(0, 1.05, S(3.88), 0.85, 0.55))
    m.add("fuselage", *fu.cockpit_fittings(S(4.1), floor, CANOPY[2], CANOPY[4] * 1.8))
    m.add("fuselage", *fu.nozzle(*NOZZLE, length=0.75))
    m.add("$canopy", fu.canopy(*CANOPY))
    m.pivots["$canopy"] = (0, CANOPY[2] + 0.1, CANOPY[0])
    # wing from published area / span / sweep;  exposed from the body side
    wing = pf.from_published(27.87, 9.45, 40.0, 0.21, z_le_root=S(6.6) + 0.0, y=1.85, x_root=0.0, t_root=0.04, t_tip=0.035)
    m.add("wing", *pf.surface(wing))
    tail = pf.trapezoid(2.79, 2.6, 0.9, 40.0, S(12.0), y=1.98, dihedral=-10.0, x_root=0.0)
    m.add("tail", *pf.surface(tail))
    m.add("tail", *pf.fin((2.5, S(10.6), 3.4), (5.15, S(13.55), 1.05)))
    m.add("wing", *pf.plate([(0.45, S(3.4)), (1.05, S(6.8)), (1.0, S(7.6)), (0.45, S(7.6))], 1.9))
    # gear: main legs fold forward into belly fairings, nose leg folds back
    for s, name in ((1, "$gear_l"), (-1, "$gear_r")):
        m.add(name, *gear.leg((s * 0.88, 1.64, S(9.05)), (s * 1.02, 0.36, S(9.05)), 0.36, 0.21))
    m.add("$gear_n", *gear.leg((0, 1.1, S(4.95)), (0, 0.24, S(5.1)), 0.24, 0.15))
    rets, rep = gear.stow(m, {"$gear_l": ((0.88, 1.64, S(9.05)), (1, 0, 0), -88, 1),
                              "$gear_r": ((-0.88, 1.64, S(9.05)), (1, 0, 0), -88, -1),
                              "$gear_n": ((0, 1.1, S(4.95)), (1, 0, 0), 92, 0)})
    for p, r in rets.items():
        m.pivots[p] = tuple(r.pivot); m.props[p] = r.as_dict()
    m.props["_gear_report"] = rep
    m.refs.append(("seat", (0, fu.rider_feet_y(floor), S(4.1)), "seat"))

    def paint(atlas, model):
        def base(X, Y, Z, cur):
            out = cur.copy(); out[:] = (118, 124, 130); out[Y < 1.5] = (150, 156, 162)
            return out, None
        atlas.paint3d(base, tags=["skin"])
        dt.tone(atlas, ["skin"], 0.035, 1.3)
        dt.panel_lines(atlas, ["skin"])
        dt.rivets(atlas, ["skin"])
        dt.access_panels(atlas, ["skin"])
        dt.ink(atlas, ["skin"], depth_m=0.03, crease_m=0.02, strength=0.4, halo=0.14)
        dt.streaks(atlas, ["skin"], color=(60, 60, 58), amount=0.08, width=0.1)
        dt.soot(atlas, ["skin"], (0, 2.1, -L / 2), 2.2)
        atlas.paint_alpha(100, tags=["glass"])
    texture_model(m, (1024, 1024), flat={"tyre": (34, 34, 34), "gear": (190, 190, 186), "intake": (30, 30, 32),
                                         "nozzle": (82, 78, 74), "exhaust": (40, 34, 30), "interior": (58, 60, 62)},
                  colors={"glass": (120, 150, 165)}, glass_tags=("glass",), paint=paint)
    return m


if __name__ == "__main__":
    import json
    from meshkit import check
    m = build()
    print(json.dumps(m.props["_gear_report"], indent=1))
    print(pf.describe(pf.from_published(27.87, 9.45, 40.0, 0.21, 0.0)))
    print(check.compare_size(m, {"length": 15.06, "height": 5.09}))
