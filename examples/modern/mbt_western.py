"""Western 3rd-generation MBT (Leopard 2A6 class) with meshkit 0.3 armor + detail kits.

Published: hull length 7.72 m, length with gun 10.97 m, width 3.75 m, height 3.0 m, 7 road wheels.
Plate angles (upper glacis ~10 deg, lower ~40 deg), turret outline and module positions are estimates.

Shows: armor.glacis (angles -> profile), stepped engine deck, turret bustle with raised floor,
turret_half_width for flush modules, running gear + track path, texture detail (hubs, grime, ink),
check.compare_size.
"""
import numpy as np
import meshkit as mk
from meshkit import armor as ar, detail as dt
from meshkit.texture import texture_model, value_noise

W, TW, WR = 3.75, 0.635, 0.35
ZN, ZT = 3.86, -3.86
PROF = ar.glacis(ZN, 1.22, 1.62, upper_deg=10.0, lower_deg=40.0, belly_y=0.47, z_tail=ZT)
TURRET = {"plan": [(0.0, 3.2), (0.25, 3.15), (1.0, 2.2), (1.75, 1.7), (1.8, -1.3), (1.6, -2.4), (0.0, -2.4)],
          "top": [(0.0, 2.6), (0.25, 2.55), (0.9, 1.9), (1.6, 1.5), (1.65, -1.2), (1.45, -2.25), (0.0, -2.25)]}
TZ, TY, TH = -0.3, 1.62, 0.78


def build():
    m = mk.Model("mbt_western")
    ar.hull(m, W, TW, PROF, rear_deck=(-2.4, ZT, 1.72, 1.79), skirt=(3.8, -3.75, 0.55))
    rollers = ar.running_gear(m, W, TW, WR, list(np.linspace(2.75, -2.75, 7)), (-3.4, 0.62, 0.31), (3.35, 0.58, 0.3),
                              rollers_z=(2.1, 0.7, -0.7, -2.1))
    m.add("$turret", *ar.faceted_turret(TZ, TY, TH, TURRET["plan"], TURRET["top"], bustle=(-0.9, 0.22)))
    m.add("hull", mk.cylinder((0, TY - 0.12, TZ), (0, TY + 0.02, TZ), 1.1, n=16, tag="hull"))
    m.pivots["$turret"] = (0, TY, TZ)
    top = TY + TH
    for sh in ar.sight(-0.78, top, TZ + 1.15, 0.62, 0.42, 0.95) + ar.periscope(0.55, top, TZ - 0.55, 0.2, 0.3):
        m.add("$turret", sh)
    for s in (1, -1):
        x = s * (ar.turret_half_width(TURRET, 0.2, 0.6) - 0.03)
        m.add("$turret", *ar.smoke_bank(x, TY + 0.55, TZ + 0.2, n=4, yaw=-s * 12))
    m.add("$turret", *ar.basket(0.0, TY + 0.2, TZ - 2.95, TZ - 2.3, 2.6))
    shells, muzzle = ar.gun((0, TY + 0.4, TZ + 0.6), 5.3, 0.1, mantlet=(0.75, 0.5, 1.6), evac=(3.2, 0.6, 0.15))
    m.add("$gun", *shells); m.pivots["$gun"] = (0, TY + 0.4, TZ + 0.6)
    m.refs.append(("muzzle", muzzle, "muzzle"))
    path = ar.track_path(list(np.linspace(2.75, -2.75, 7)), WR, (-3.4, 0.62, 0.31), (3.35, 0.58, 0.3), 0.95)
    sp = ar.link_spacing(path)
    m.add("$track_link", mk.box(-(W / 2 - TW / 2), 0, 0, TW, 0.07, sp * 0.9, tag="track"))
    m.props["track"] = {"path": path, "spacing": sp, "rollers": rollers}
    wheels = [(p[1], p[2]) for n, p in rollers if n.startswith("$rw") or n.startswith("$id")]

    def paint(atlas, model):
        def camo(X, Y, Z, cur):
            out = np.empty((len(X), 3), np.float32); out[:] = (70, 80, 54)
            out[value_noise(X * 0.7 + Y * 0.5, Z, 1.1, seed=1) > 0.4] = (92, 72, 50)
            out[value_noise(X * 0.7 + Y * 0.5, Z, 1.4, seed=2) > 0.65] = (34, 34, 30)
            return out, None
        atlas.paint3d(camo, tags=["hull", "turret", "wheel", "sprocket"])
        dt.tone(atlas, ["hull", "turret"])
        atlas.paint3d(dt.grid_fn("x", "z", -1.4, 1.4, ZT + 0.2, -2.5, 2.8, 0.09, gap=0.05, color=(50, 54, 40)),
                      tags=["hull"], kinds=("top",))
        dt.wheel_hubs(atlas, ["wheel"], wheels, WR)
        dt.ink(atlas, ["hull", "turret", "wheel", "sprocket", "kit"], depth_m=0.03, crease_m=0.01, strength=0.5, halo=0.2)
        dt.grime(atlas, ["hull"], 1.55, color=(104, 90, 68), amount=0.75, scale=0.6)
        dt.streaks(atlas, ["hull", "turret"], color=(60, 56, 46), amount=0.14, width=0.12)
        dt.chips(atlas, ["hull", "turret"], density=0.03, scale=0.05)
    texture_model(m, (1024, 1024), flat={"track": (46, 44, 40), "gun": (58, 62, 52), "glass": (60, 80, 90),
                                         "rack": (46, 44, 40), "kit": (84, 82, 66)}, paint=paint)
    return m


if __name__ == "__main__":
    from meshkit import check
    m = build()
    print("upper glacis", round(ar.angle_of(PROF["upper"][2], PROF["upper"][3]), 1), "deg")
    print(check.compare_size(m, {"length": 10.97, "width": 3.75, "height": 3.0}, parts=[p for p in m.order if p != "$track_link"]))
