"""Aegis destroyer (Arleigh Burke Flight IIA class) with the meshkit 0.3 ship kit.

Published: length 155.3 m, beam 20.4 m, draft 6.3 m, 96 VLS cells, SPY-1D arrays on the forward deckhouse
(facing forward-port / forward-starboard) and the after deckhouse (facing aft).  Deckhouse outlines,
heights and fittings are estimates from photographs.

Shows: hull with flare, polygonal sloped deckhouse with arrays flush on its walls (wall_point / array_face),
bridge with outward-leaning windows and wings, raked mast, funnels, boats, rails, VLS lid grids and a hull
number that reads correctly on both sides (side_number + split_sides).
"""
import numpy as np
import meshkit as mk
from meshkit import ship as sh, detail as dt
from meshkit.texture import texture_model

HULL = [(-77.6, 6.6, 8.6, 7.6, 1.5), (-70, 6.8, 9.6, 8.7, 5.0), (-40, 7.2, 10.1, 9.7, 6.2), (0, 7.5, 10.2, 10.0, 6.3),
        (30, 8.4, 9.7, 9.0, 6.2), (50, 9.4, 8.3, 7.0, 5.8), (65, 10.4, 6.0, 3.8, 5.0), (74, 11.0, 3.2, 1.0, 3.5),
        (77.7, 11.4, 0.5, 0.05, 1.2)]


def build():
    m = mk.Model("destroyer_aegis")
    m.add("hull", sh.hull(HULL, flare=0.1))
    add = lambda *s: m.add("super", *s)
    bot = [(3.6, 37.5), (7.6, 31.0), (7.6, 3.0)]; top = [(3.1, 35.9), (6.4, 30.4), (6.4, 3.9)]
    y0, y1 = 7.5, 15.6
    add(sh.house(sh.sym(bot), y0, y1, sh.sym(top)))
    for s in (1, -1):                                   # forward arrays flush on the angled walls
        x, z = sh.wall_point(bot[0], bot[1], top[0], top[1], y0, y1, 12.0)
        n = np.array([s * 0.84, 0.15, 0.53])
        add(sh.array_face((s * x, 12.0, z) + n * 0.03, n))
    add(sh.house(sh.sym([(3.0, 35.4), (6.0, 30.5), (6.0, 17.0)]), 15.5, 18.1, inset=0.03))
    add(*sh.bridge([(2.8, 34.9), (5.8, 30.7), (5.8, 24.5)], 18.0, 20.6))
    add(mk.cylinder((0, 20.9, 24.5), (0, 44.0, 18.5), 0.75, 0.3, n=6, tag="mast"))
    for s in (1, -1):
        add(mk.cylinder((s * 4.5, 18.0, 22.0), (0, 32.0, 21.5), 0.4, 0.3, n=6, tag="mast"))
    add(*sh.yard((0, 31.0, 21.8), 11.0, depth=1.2), *sh.yard((0, 36.5, 20.2), 7.0, depth=1.0))
    add(sh.radome((0, 40.0, 19.55), 1.3), sh.whip(0, 41.1, 19.5, 4.0))
    add(sh.house(sh.sym([(7.4, 3.2), (7.4, -6.0)]), 7.3, 13.2, inset=0.04))
    add(*sh.funnel(-0.8, 13.0, 21.5, 6.6, 7.4, taper=0.82, rake=0.9, pipes=3))
    abot = [(7.4, -6.0), (7.4, -24.0), (3.8, -29.5)]; atop = [(6.5, -6.6), (6.5, -23.4), (3.3, -28.2)]
    add(sh.house(sh.sym(abot), 7.2, 16.4, sh.sym(atop)))
    for s in (1, -1):                                   # aft arrays
        x, z = sh.wall_point(abot[1], abot[2], atop[1], atop[2], 7.2, 16.4, 12.2)
        n = np.array([s * 0.84, 0.15, -0.54])
        add(sh.array_face((s * x, 12.2, z) + n * 0.03, n))
        add(*sh.boat((s * 5.3, 13.12, -10.0)), *sh.davit((s * 6.8, 13.15, -7.0), 2.2, 1.8, s))
    add(*sh.funnel(-14.5, 16.3, 21.0, 6.2, 7.0, taper=0.82, rake=0.9, pipes=3))
    add(sh.house(sh.sym([(9.6, -41.5), (9.6, -58.0)]), 7.0, 13.6, inset=0.02))
    add(sh.vls(0, 46.0, 7.4, 9.6, 8.95), sh.vls(0, -36.5, 9.0, 11.0, 7.1))
    for s in (1, -1):
        add(*sh.triple_tubes((s * 7.6, 7.25, -32.0), 90 * s))
    add(*sh.anchors(HULL, 70.0), *sh.bollards(HULL, [-70, -55, 15, 38, 55]))
    add(*sh.deck_edge_rails(HULL, -77.0, 72.0, step=6.0))
    add(sh.whip(0, 11.0, 77.0, 3.0, 0.07), sh.whip(0, 6.4, -76.8, 3.8, 0.07))
    base, head, piv = sh.ciws(0, 13.55, -50.0)
    m.add("super", *base); m.add("$ciws", *head); m.pivots["$ciws"] = piv

    def paint(atlas, model):
        S = np.array(HULL)

        def hullc(X, Y, Z, cur):
            out = cur.copy(); out[:] = (128, 132, 136); out[Y < 0.6] = (34, 34, 34); out[Y < -0.3] = (128, 48, 40)
            return out, None
        atlas.paint3d(hullc, tags=["hull"], kinds=("side", "front", "bottom"))
        atlas.paint3d(dt.lines_fn("y", 2.4, 0.05, k=0.86, lo=0.7), tags=["hull"], kinds=("side",))
        atlas.paint3d(dt.lines_fn("z", 7.2, 0.05, k=0.86, lo=0.7), tags=["hull"], kinds=("side",))
        atlas.paint3d(dt.grid_fn("x", "z", -3.6, 3.6, 41.3, 50.7, 1.2, 1.2), tags=["vls"], kinds=("top",))
        atlas.paint3d(dt.grid_fn("x", "z", -4.4, 4.4, -41.9, -31.1, 1.2, 1.2), tags=["vls"], kinds=("top",))
        dt.side_number(atlas, "81", ["hull"], 57.0, 4.0, 3.2)
        dt.streaks(atlas, ["hull"], color=(110, 70, 50), amount=0.16, width=0.35, seed=21, kinds=("side",))
        dt.ink(atlas, ["hull", "super", "funnel", "launcher", "vls", "boat"], depth_m=0.08, crease_m=0.03, strength=0.45, halo=0.15)
    texture_model(m, (2048, 2048), colors={"hull": (128, 132, 136), "super": (138, 142, 146), "funnel": (138, 142, 146),
                                           "vls": (78, 80, 82), "launcher": (112, 118, 114), "boat": (205, 120, 50)},
                  flat={"rail": (150, 154, 158), "mast": (110, 114, 118), "radar": (120, 124, 128), "dome": (230, 230, 226),
                        "dark": (40, 42, 44), "black": (30, 30, 30), "white": (226, 226, 222), "gun": (96, 100, 104),
                        "gunh": (138, 142, 146)}, paint=paint, split_sides=True)
    return m


if __name__ == "__main__":
    from meshkit import check
    m = build()
    print(check.compare_size(m, {"length": 155.3, "width": 20.4}))
