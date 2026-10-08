"""Example: a small generic light tank, built only with meshkit (no Blender).

    python -m meshkit run examples/light_tank.py -o examples/out

Conventions: +X left, +Y up, +Z forward, 1 unit = 1 m, origin = ground contact centre.
Moving parts are separate parts whose pivot is the rotation origin ($turret yaws about Y,
$gun pitches about X)."""
import meshkit as mk
from meshkit.texture import texture_model, value_noise
import numpy as np

L, W, H = 4.6, 2.3, 1.0           # hull length, width, height (m)
TRACK_W, CLEAR = 0.38, 0.35       # track width, ground clearance of the hull floor


def build():
    m = mk.Model("light_tank")
    # lower hull between the tracks: side profile (z, y) extruded across
    hw = W / 2 - TRACK_W
    TOP = CLEAR + H
    low = [(-L / 2 + 0.2, CLEAR), (L / 2 - 0.45, CLEAR), (L / 2 - 0.1, 0.70), (-L / 2 + 0.1, 0.70)]
    m.add("hull", mk.extrude_zy(low, -hw - 0.02, hw + 0.02, tag="hull"))
    # upper hull over the full width (overlaps the lower hull by 5 cm so the shells interpenetrate)
    up = [(-L / 2, 0.65), (L / 2, 0.65), (L / 2 - 0.55, TOP), (-L / 2 + 0.15, TOP), (-L / 2, TOP - 0.25)]
    m.add("hull", mk.extrude_zy(up, -W / 2, W / 2, tag="hull"))
    for s in (1, -1):
        # track run: a rounded side profile extruded across the track width
        prof = [(-L / 2 + 0.25, 0.02), (L / 2 - 0.25, 0.02), (L / 2 - 0.02, 0.32), (L / 2 - 0.25, 0.62),
                (-L / 2 + 0.25, 0.62), (-L / 2 + 0.02, 0.32)]
        xa, xb = s * (W / 2 - TRACK_W), s * (W / 2 - 0.01)
        m.add("tracks", mk.extrude_zy(prof, min(xa, xb), max(xa, xb), tag="track"))
        # road wheels poke out of the track's inner face so they touch it
        for z in np.linspace(-L / 2 + 0.6, L / 2 - 0.6, 5):
            xi = s * (W / 2 - TRACK_W - 0.06)
            m.add("tracks", mk.cylinder((xi, 0.33, z), (xi + s * 0.10, 0.33, z), 0.27, n=10, tag="wheel"))
    # turret: superellipse rings lofted upwards, pivot on its yaw axis
    ty = TOP
    rings = [mk.ring_superellipse(0, 0, 0, 0.85, 0.75, n=12, p=3.0), mk.ring_superellipse(0, 0, 0, 0.72, 0.62, n=12, p=3.0)]
    # ring_superellipse works in the XY plane at z; rotate so the rings lie horizontal (XZ plane)
    R = mk.rot_x(-90)
    rings = [r @ R.T + [0, ty - 0.02 + 0.55 * k, -0.2] for k, r in enumerate(rings)]
    m.add("$turret", mk.loft(rings, tag="turret"))
    m.pivots["$turret"] = (0.0, ty, -0.2)
    # gun: mantlet + barrel, pivot at the trunnions
    gz = 0.45; gy = ty + 0.28
    m.add("$gun", mk.box(0, gy, gz, 0.42, 0.32, 0.3, tag="turret"))
    m.add("$gun", mk.cylinder((0, gy, gz + 0.1), (0, gy, gz + 2.0), 0.055, n=8, tag="gun"))
    m.pivots["$gun"] = (0.0, gy, gz - 0.05)
    # hatch on the roof
    m.add("$turret", mk.cylinder((0.2, ty + 0.52, -0.45), (0.2, ty + 0.6, -0.45), 0.28, n=10, tag="turret"))
    m.refs += [("seat_commander", (0.2, ty + 0.6, -0.45), "seat"), ("muzzle", (0, gy, gz + 2.0), "muzzle")]
    # texture: projected olive with soft mottling, flat swatches for tracks / wheels / gun
    def paint(atlas, model):
        def camo(X, Y, Z, cur):
            n = value_noise(X * 1.0 + Y, Z, 1.2, seed=3)
            return np.where((n > 0.15)[:, None], [72, 82, 52], [104, 112, 70]).astype(np.float32), None
        atlas.paint3d(camo, tags=["hull", "turret"])
    texture_model(m, (256, 256), flat={"track": (52, 50, 46), "wheel": (70, 72, 64), "gun": (60, 64, 52)}, paint=paint)
    return m
