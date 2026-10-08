"""Modern twin-engine fighter (Su-27 class) - the case the WW2 kit handled badly.

Published: length 21.94 m, span 14.7 m, wing area 62 m2, LE sweep 42 deg, twin fins on the nacelle booms,
tailplanes on booms outside the nozzles.  Section shapes, taper and boom sizes are estimates.

Shows: fuselage.cover_nozzles (aft body wide enough to contain BOTH nozzles), boom-mounted tailplanes as two
separate panels (planform stations starting at x > 0), mirrored canted fins, check.penetration (tails must
not cut through nozzles), gear.stow with fairings.
"""
import meshkit as mk
from meshkit import planform as pf, fuselage as fu, gear, check

L = 21.94
S = lambda s: L / 2 - s
FUS = [(S(0.0), 2.15, 0, 0, 0, 2), (S(1.5), 2.2, 0.45, 0.45, 0.45, 2), (S(3.5), 2.3, 0.7, 0.68, 0.68, 2.2),
       (S(5.5), 2.3, 0.82, 0.7, 0.75, 2.4), (S(8.0), 2.25, 1.2, 0.55, 0.5, 3.2), (S(11.0), 2.2, 1.6, 0.45, 0.35, 4),
       (S(15.0), 2.15, 1.5, 0.4, 0.3, 4), (S(18.0), 2.1, 1.0, 0.35, 0.3, 3), (S(20.5), 2.1, 0.4, 0.3, 0.3, 2.4),
       (S(21.9), 2.1, 0.18, 0.18, 0.18, 2)]
NOZZLES = [(1.1, 1.6, S(20.2), 0.55), (-1.1, 1.6, S(20.2), 0.55)]
CANOPY = (S(7.0), S(3.8), 2.65, 3.5, 0.45)


def build():
    m = mk.Model("fighter_twin")
    st = fu.cover_nozzles(FUS, NOZZLES, length=4.0)          # widen the aft body around both engines
    floor = CANOPY[2] - 0.55
    m.add("fuselage", fu.body_with_tub(st, [(CANOPY[1] - 0.3, CANOPY[0] + 0.15, CANOPY[4] * 0.95, floor)]))
    m.add("fuselage", *fu.cockpit_fittings(S(5.0), floor, CANOPY[2], CANOPY[4] * 1.8))
    for x in (1.1, -1.1):                                    # intake / engine nacelles under the body
        m.add("fuselage", fu.pod(x, [(S(8.2), 1.5, 0.5, 0.5, 0.55, 4), (S(10.0), 1.5, 0.6, 0.55, 0.6, 3.5),
                                     (S(15.0), 1.55, 0.62, 0.55, 0.6, 3.0), (S(19.4), 1.6, 0.55, 0.55, 0.55, 2.4)]))
        m.add("fuselage", fu.intake_face(x, 1.5, S(8.22), 0.95, 1.0))
    for nz in NOZZLES:
        m.add("fuselage", *fu.nozzle(*nz, length=1.0))
    for x in (2.05, -2.05):                                  # tail booms outside the nacelles
        m.add("fuselage", fu.pod(x, [(S(14.2), 1.85, 0.05, 0.05, 0.05, 2), (S(15.2), 1.8, 0.3, 0.28, 0.28, 2),
                                     (S(20.0), 1.75, 0.28, 0.26, 0.26, 2), (S(20.9), 1.75, 0.1, 0.1, 0.1, 2)]))
    m.add("$canopy", fu.canopy(*CANOPY))
    m.pivots["$canopy"] = (0, CANOPY[2] + 0.1, CANOPY[0])
    wing = pf.from_published(62.0, 14.7, 42.0, 0.2, z_le_root=S(8.5), y=2.15, dihedral=-2.5)
    m.add("wing", *pf.surface(wing))
    m.add("wing", *pf.plate([(0.6, S(4.5)), (1.9, S(9.3)), (2.2, S(10.5)), (0.6, S(10.5))], 2.25))
    tail = pf.trapezoid(4.94, 3.4, 1.2, 35.0, S(16.8), y=1.75, x_root=2.05)    # starts on the boom -> 2 panels
    m.add("tail", *pf.surface(tail))
    m.add("tail", *pf.fin((1.95, S(15.8), 3.6), (5.65, S(19.0), 1.3), x=2.1, mirror_x=True))
    for s, name in ((1, "$gear_l"), (-1, "$gear_r")):
        m.add(name, *gear.leg((s * 2.17, 2.0, S(12.6)), (s * 2.17, 0.45, S(12.6)), 0.45, 0.3))
    m.add("$gear_n", *gear.leg((0, 1.7, S(6.4)), (0, 0.33, S(6.5)), 0.33, 0.18))
    rets, rep = gear.stow(m, {"$gear_l": ((2.17, 2.0, S(12.6)), (0, 0, 1), -86, 1),
                              "$gear_r": ((-2.17, 2.0, S(12.6)), (0, 0, -1), -86, -1),
                              "$gear_n": ((0, 1.7, S(6.4)), (1, 0, 0), 95, 0)})
    for p, r in rets.items():
        m.pivots[p] = tuple(r.pivot); m.props[p] = r.as_dict()
    m.props["_gear_report"] = rep
    m.props["_tail_in_nozzles"] = check.penetration(m, ["tail"], ["fuselage"], tags_b=("nozzle", "exhaust"))
    return m


if __name__ == "__main__":
    import json
    m = build()
    print(json.dumps(m.props["_gear_report"], indent=1))
    print("tail inside body/nozzles:", m.props["_tail_in_nozzles"])
    print(check.compare_size(m, {"length": 21.94, "width": 14.7}))
