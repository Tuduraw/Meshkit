"""Original compact crawler rig (not a copy of any real machine): crawler carrier, swinging upper
structure, a mast pinned at its foot and raised by one thick central ram, four diagonal outriggers on
the upper structure, and a split pile steady at the mast foot.

    python -m meshkit run examples/machinery/crawler_rig.py -o out --mode parts
    python examples/machinery/crawler_rig.py        # + the pose checks of AI_GUIDE 9.6

Shows the meshkit.machinery workflow: crawler_layout -> crawler, ram_mount_search -> ram -> ram_joints,
outriggers (two stages on one channel), clamp_jaws, then solve / sweep_check / ram_drift over poses."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import meshkit as mk  # noqa: E402
from meshkit import machinery as mc  # noqa: E402

SWING = (0.0, 0.7, 0.0)
FOOT = (0.0, 1.0, 1.6)            # mast foot pin
DOWN = 80.0                       # the mast lowers back by this angle ("mast" 1 = upright)
PILE = (0.0, 1.0, 2.35)
JOINTS = []
RAMS = []                         # (rod part, attach point, child) for ram_drift


def build():
    JOINTS.clear()
    RAMS.clear()
    m = mk.Model("crawler_rig")
    # --- carrier: two crawlers whose running gear never overlaps (layout_overlaps is checked below)
    for side, s in (("l", 1), ("r", -1)):
        mc.crawler(m, side, s * 0.72, s * 1.45, 2.05, 0.68, r_end=0.27, wheel_r=0.11, n_wheels=6, carrier_r=0.07,
                   n_carrier=1, link_len=0.2, links="laid")
    m.add("body", mk.box(0, 0.47, 0, 1.44, 0.43, 2.6, tag="frame"))
    m.add("body", mk.cylinder((0, 0.6, 0), (0, 0.75, 0), 0.6, n=16, tag="frame"))   # swing bearing
    # --- upper structure (swings); engine hood off to the right, clear of the lowered mast
    U = "$upper"
    m.add(U, mk.box(0, 0.85, -0.35, 1.9, 0.3, 3.3, tag="paint"))
    m.add(U, mk.box(-0.6, 1.35, -0.45, 0.6, 0.7, 2.7, tag="paint"))
    m.add(U, mk.box(0.6, 1.6, 0.6, 0.7, 1.2, 1.1, tag="glass"))
    m.add(U, mk.box(0, 1.19, -1.6, 0.5, 0.38, 0.3, tag="frame"))          # rest for the lowered mast
    m.add(U, mk.box(0, 0.95, 1.45, 0.7, 0.5, 0.5, tag="frame"))           # mast foot bracket
    m.pivots[U] = SWING
    JOINTS.append(mc.joint(U, SWING, (0, 1, 0), channel="swing"))
    # --- mast, pinned at its foot; "mast" 0 lays it back onto the rest
    M = "$mast"
    m.add(M, mk.box(0, 4.75, 1.75, 0.4, 7.5, 0.3, tag="mast"))
    m.add(M, mk.cylinder((-0.36, FOOT[1], FOOT[2]), (0.36, FOOT[1], FOOT[2]), 0.08, n=8, tag="frame"))
    m.pivots[M] = FOOT
    JOINTS.append(mc.joint(M, FOOT, (1, 0, 0), channel="mast", parent=U, factor=DOWN, offset=-DOWN))
    # --- one thick central ram: search mounts that need only a barrel and a rod and keep leverage
    angles = [-DOWN * i / 24 for i in range(25)]
    bases = [(0.0, 1.05, z) for z in np.arange(-1.2, 1.0, 0.1)]
    attaches = [(0.0, y, 1.55) for y in np.arange(3.0, 7.0, 0.25)]
    best = mc.ram_mount_search(FOOT, (1, 0, 0), angles, bases, attaches, min_arm=0.3)
    base, attach, stages, lmin, lmax, arm = best[0]
    m.add(U, mk.box(0, 1.05, base[2], 0.4, 0.2, 0.4, tag="frame"))         # ram foot bracket
    m.add(M, mk.box(0, attach[1], 1.55, 0.28, 0.4, 0.17, tag="frame"))     # ram lug
    parts = mc.ram(m, "$ram", base, attach, 0.2, lmin, lmax)
    JOINTS.extend(mc.ram_joints(parts, U, base, attach, M, (1, 0, 0)))
    RAMS.append((parts[-1], attach, M))
    # --- four outriggers under the corners of the upper structure (they swing with it)
    JOINTS.extend(mc.outriggers(m, mc.corner_mounts(1.3, -2.0, 0.95, 0.15), 0.9, 0.12, 1.4, 0.74, 0.2, 1.7, 1.8,
                                parent=U))
    # --- split pile steady at the mast foot, hollow where the pile passes
    JOINTS.extend(mc.clamp_jaws(m, M, PILE[0], PILE[2], 0.3, 0.1, 1.05, 1.25, lug_to=(0.12, 1.8)))
    m.add("$pile", mk.cylinder((0, 0.0, PILE[2]), (0, 3.0, PILE[2]), 0.18, n=10, tag="pile"))
    m.props["machinery"] = {"ram": {"stages": stages, "lmin": lmin, "lmax": lmax, "arm": arm}}
    return m


def checks(m):
    out = {"overlaps": mc.layout_overlaps(mc.crawler_layout(2.05, 0.68, 0.27, 0.11, 6, 0.07, 1), gap=0.03),
           "ram": m.props["machinery"]["ram"]}
    poses = mc.sweep_poses({"mast": (0.0, 1.0), "outrigger": (0.0, 1.0), "steady": (0.0, 1.0)},
                           base={"mast": 1.0})
    out["ram_drift"] = mc.ram_drift(JOINTS, poses, RAMS)
    # shoes are drawn by the engine and the pile hangs free in the steady: leave them out of the groups
    bad = mc.sweep_check(m, JOINTS, poses[::3], skip=("$crawler_l", "$crawler_r", "$pile"),
                         pairs=[(["$outrigger_fl_leg", "$outrigger_fr_leg"], ["$crawler_l", "$crawler_r"])])
    out["bad_poses"] = bad
    return out


if __name__ == "__main__":
    from PIL import Image
    from meshkit import check, view
    model = build()
    rep = check.report(model)
    print("ok", rep["ok"], rep["summary"])
    print(checks(model))
    imgs = [view.render_view(model, {"eye": (12, 4, 0), "target": (0, 2.5, 0), "fov": 42}, 420, 420, "parts",
                             transforms=mc.pose_transforms(JOINTS, vals))
            for vals in ({"mast": 1, "outrigger": 1, "steady": 1}, {"mast": 0.5}, {"mast": 0})]
    sheet = Image.new("RGB", (420 * len(imgs), 420))
    for i, im in enumerate(imgs):
        sheet.paste(Image.fromarray(im), (420 * i, 0))
    sheet.save(sys.argv[1] if len(sys.argv) > 1 else "crawler_rig_poses.png")
