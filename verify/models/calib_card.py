"""Verification model "calib_card" - a test card for meshkit (not a vehicle).

Every element is there to make a specific fault visible in a render:
  - axis arrows   red = +X (model left), green = +Y (up), blue = +Z (forward). An axis swap or mirror
                  (e.g. a wrong Blender / glTF conversion) shows as a wrong colour on a wrong side.
  - base plate    0.5 m checkerboard on the top face with a red square in the front-left corner (+X, +Z)
                  and a yellow "F" on the right half: UV / texture-projection errors, flipped images.
  - primitives    box, cylinder, lathe (cone), ellipsoid, superellipse loft, wing, propeller blade - each its
                  own part, so 'parts' mode shows the split and 'orientation' mode shows inverted faces.
  - glass dome    translucent (alpha 110) - the mod-style dither.
  - $lid          a hinged part with its pivot on the hinge - pivots must survive OBJ / glb.
  - refs          a seat marker (yellow ring in previews with refs).

    python -m meshkit run verify/models/calib_card.py -o out
Conventions: +X left, +Y up, +Z forward, 1 unit = 1 m."""
import numpy as np
import meshkit as mk
from meshkit.texture import texture_model


def arrow(p0, d, length, r, tag):
    d = np.asarray(d, float); p0 = np.asarray(p0, float)
    shaft = mk.cylinder(p0, p0 + d * (length - 0.25), r, n=8, tag=tag)
    head = mk.lathe(p0 + d * (length - 0.25), d, [(0.0, r * 2.4), (0.25, 0.0)], n=8, tag=tag)
    return [shaft, head]


def build():
    m = mk.Model("calib_card")
    # base plate 4 x 0.2 x 4 m, top at y = 0.2
    m.add("plate", mk.box(0, 0.1, 0, 4.0, 0.2, 4.0, tag="plate"))
    # axis arrows from the plate centre
    m.add("axis_x", arrow((0, 0.35, 0), (1, 0, 0), 2.3, 0.05, "axis_x"))
    m.add("axis_y", arrow((0, 0.2, 0), (0, 1, 0), 2.3, 0.05, "axis_y"))
    m.add("axis_z", arrow((0, 0.35, 0), (0, 0, 1), 2.3, 0.05, "axis_z"))
    m.add("axis_y", mk.box(0, 0.35, 0, 0.18, 0.3, 0.18, tag="axis_y"))      # hub joining the three arrows
    # primitives (one part each), on the rear half of the plate
    m.add("p_box", mk.box(1.35, 0.5, -1.35, 0.6, 0.6, 0.6, tag="solid_a"))
    m.add("p_cylinder", mk.cylinder((0.45, 0.2, -1.35), (0.45, 1.0, -1.35), 0.3, n=12, tag="solid_b"))
    m.add("p_cone", mk.lathe((-0.45, 0.2, -1.35), (0, 1, 0), [(0.0, 0.32), (0.5, 0.2), (0.9, 0.0)], n=10, tag="solid_a"))
    m.add("p_ellipsoid", mk.ellipsoid((-1.35, 0.5, -1.35), 0.32, 0.35, 0.25, nu=10, nv=7, tag="solid_b"))
    R = mk.rot_x(-90)
    # superellipse rings lie in XY at z; turn them horizontal and stack them
    rings = [mk.ring_superellipse(0, 0, 0, 0.3 - 0.06 * k, 0.22 - 0.04 * k, n=12, p=3.0) @ R.T + [1.35, 0.2 + 0.28 * k, -0.45]
             for k in range(4)]
    m.add("p_loft", mk.loft(rings, tag="solid_a"))
    m.add("p_wing", mk.wing([{"x": -0.9, "y": 0.75, "z": -0.2, "c": 0.8, "t": 0.12},
                             {"x": -1.85, "y": 0.85, "z": -0.35, "c": 0.45, "t": 0.1}], tag="solid_b"))
    m.add("p_wing", mk.cylinder((-0.95, 0.2, -0.6), (-0.95, 0.8, -0.6), 0.05, n=6, tag="solid_b"))
    m.add("p_blade", mk.blade((-1.4, 0.95, 0.9), (-1.4, 1.75, 0.9), 0.22, 0.12, 0.04, (0, 0, 1), tag="solid_a"))
    m.add("p_blade", mk.cylinder((-1.4, 0.2, 0.9), (-1.4, 1.0, 0.9), 0.06, n=6, tag="solid_a"))
    # glass dome on the front-right
    m.add("glass", mk.lathe((-1.0, 0.15, 1.3), (0, 1, 0), [(0.0, 0.5), (0.18, 0.48), (0.38, 0.38), (0.52, 0.2), (0.58, 0.0)],
                            n=14, tag="glass"))
    # hinged lid: a box leaning back from its hinge line, pivot on the hinge
    hinge = (1.2, 0.2, 1.0)
    lid = mk.box(1.2, 0.2 + 0.35, 1.0 + 0.04, 0.9, 0.7, 0.08, tag="lid")
    lid.rotate(mk.rot_x(-20), pivot=hinge)
    m.add("$lid", lid)
    m.pivots["$lid"] = hinge
    m.refs += [("seat_test", (1.2, 0.9, 1.0), "seat")]

    def paint(atlas, model):
        def plate(X, Y, Z, cur):
            ix = np.floor((X + 2.0) / 0.5).astype(int); iz = np.floor((Z + 2.0) / 0.5).astype(int)
            out = np.where(((ix + iz) % 2 == 0)[:, None], [235, 235, 228], [60, 64, 72]).astype(np.float32)
            corner = (X > 1.5) & (Z > 1.5)                               # front-left square (+X, +Z)
            out[corner] = (210, 40, 40)
            # letter F on the right half (-X): reads correctly from above with the nose (+Z) up the image
            fx, fz = -1.0, 0.0
            stem = (np.abs(X - fx - 0.25) < 0.08) & (Z > fz - 0.5) & (Z < fz + 0.5)
            bar1 = (X < fx + 0.33) & (X > fx - 0.35) & (np.abs(Z - fz - 0.42) < 0.08)
            bar2 = (X < fx + 0.33) & (X > fx - 0.15) & (np.abs(Z - fz - 0.02) < 0.07)
            out[stem | bar1 | bar2] = (240, 200, 30)
            return out, None
        atlas.paint3d(plate, tags=["plate"])
        atlas.paint_alpha(110, tags=["glass"])

    texture_model(m, (256, 256), colors={"glass": (150, 190, 215)},
                  flat={"axis_x": (220, 40, 40), "axis_y": (40, 180, 60), "axis_z": (40, 80, 230),
                        "solid_a": (200, 150, 90), "solid_b": (120, 150, 190), "lid": (150, 110, 170)},
                  glass_tags=("glass",), paint=paint)
    return m
