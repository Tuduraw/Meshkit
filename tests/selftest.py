"""Self test (no Blender, no ww2gen needed):  python tests/selftest.py

Builds the example model, checks it, round-trips it through OBJ and GLB (geometry, UVs, pivots,
refs must survive), renders every view and display mode, and checks that a broken mesh is caught."""
import os
import sys
import tempfile
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import meshkit as mk  # noqa: E402
from meshkit import check, view, cli  # noqa: E402


def corners(m):
    out = []
    for p in m.order:
        for si, sh in enumerate(m.parts[p].shells):
            P = sh.V()
            for fi, f in enumerate(sh.faces):
                for k in range(1, len(f) - 1):
                    tri = [f[0], f[k], f[k + 1]]
                    uv = np.asarray(m.uvs[(p, si, fi)])[[0, k, k + 1]] if m.uvs else np.zeros((3, 2))
                    out.append(np.hstack([P[tri].ravel(), uv.ravel()]))
    a = np.round(np.array(out), 4)          # rounded so that sorting pairs the same triangles
    return a[np.lexsort(a.T[::-1])]


def main():
    tmp = tempfile.mkdtemp()
    model, rep = cli.run_script(os.path.join(ROOT, "examples", "light_tank.py"), tmp, render=False)
    assert rep["ok"], rep["issues"][:5]
    ref = corners(model)
    for ext in ("obj", "glb"):
        p = os.path.join(tmp, "light_tank." + ext)
        m2 = mk.load(p)
        r2 = check.report(m2)
        assert r2["issue_count"] == 0, (ext, r2["issues"][:5])
        assert m2.order == model.order, (ext, m2.order)
        c2 = corners(m2)
        assert c2.shape == ref.shape and np.abs(c2 - ref).max() < 2e-4, (ext, np.abs(c2 - ref).max())
        if ext == "glb":
            assert np.allclose(m2.pivots["$turret"], model.pivots["$turret"])
            assert [r[0] for r in m2.refs] == [r[0] for r in model.refs]
            assert m2.texture is not None
        print(ext, "round trip ok", c2.shape[0], "triangles")
    # blender axes round trip
    p = os.path.join(tmp, "zup.obj")
    mk.write_obj(model, p, axes="blender")
    m3 = mk.read_obj(p, axes="blender")
    assert np.abs(corners(m3) - ref).max() < 2e-4
    print("blender-axes round trip ok")
    # every view / mode renders
    for mode in ("texture", "parts", "solid", "tags", "orientation", "xray"):
        img = view.render_view(model, "iso", 200, 140, mode=mode, wire=True)
        assert img.shape == (140, 200, 3) and img.std() > 1, mode
    view.sheet(model, views=view.VIEW_NAMES, W=200, H=140).save(os.path.join(tmp, "all.png"))
    print("render ok")
    # robot kit (meshkit.mech): the concept example must build clean, and the fill ratios must keep
    # the four design languages apart
    mm, mrep = cli.run_script(os.path.join(ROOT, "examples", "mech", "concept_legs.py"), tmp, render=False)
    assert mrep["issue_count"] == 0 and mrep["contact_groups"] == 4, (mrep["issues"][:5], mrep["floating"])
    from meshkit import mech
    fills = {g: mech.fill_ratio(mm, [g]) for g in mm.order}
    assert fills["skeletal"] < 0.12 < fills["faceted"] < fills["fortress"], fills
    lr = mech.limb_report([{"name": "l", "hip": [0, 3, 0], "knee": [0, 1.7, 0.38], "ankle": [0, 0.42, -0.04]}])
    assert 20 < lr[0]["bends"][0] < 40, lr
    print("mech kit ok", {k: round(v, 2) for k, v in fills.items()})
    # a hole and a flipped face must be reported, and show red in orientation mode
    bad = mk.Model("bad"); sh = mk.box(0, 0, 0, 1, 1, 1)
    sh.faces[0] = tuple(reversed(sh.faces[0])); del sh.faces[1]; del sh.tags[1]
    bad.add("a", sh)
    rb = check.report(bad)
    assert not rb["ok"] and rb["issue_count"] > 0
    img = view.render_view(bad, "iso", 200, 140, mode="orientation")
    assert ((img[..., 0] > 200) & (img[..., 1] < 90)).sum() > 50
    print("defect detection ok")
    # floating part
    fl = mk.Model("fl"); fl.add("a", mk.box(0, 0, 0, 1, 1, 1)); fl.add("b", mk.box(3, 0, 0, 1, 1, 1))
    assert check.report(fl)["contact_groups"] == 2
    print("floating detection ok")
    modern_kits()
    seethrough_and_shared_texels()
    machinery_kit()
    print("ALL OK", tmp)


def modern_kits():
    """v0.3: planform / fuselage / gear / armor / ship / detail kits and the new checks."""
    import math
    from meshkit import planform as pf, gear, armor as ar
    # planform from published data reproduces area / span / sweep, tips are clipped (no extra ring)
    w = pf.from_published(27.87, 9.45, 40.0, 0.21, 0.0)
    d = pf.describe(w)
    assert abs(d["exposed_area"] - 27.87) < 0.05 and d["span"] == 9.45 and abs(d["le_sweep"] - 40) < 0.1, d
    sh = pf.surface(w)[0]
    assert abs(sh.bbox()[1][0] - 4.725) < 1e-6, sh.bbox()
    assert len(pf.surface(pf.trapezoid(3, 2, 1, 30, 0, x_root=1.0))) == 2      # boom tails -> 2 panels
    # rotate-then-slide offset reproduces the model-space shift
    r = gear.Retraction((0, 1, 0), (1, 0, 0), 90, (0.1, 0.2, -0.3))
    P = np.array([[0.3, 0.2, 0.5]])
    R = mk.rot_axis((1, 0, 0), 90)
    assert np.allclose((P - r.pivot + r.rotate_then_slide_offset()) @ R.T + r.pivot, r(P))
    # glacis angles
    pr = ar.glacis(3.8, 1.2, 1.6, 10, 40, 0.47, -3.8)
    assert abs(ar.angle_of(pr["upper"][2], pr["upper"][3]) - 10) < 1e-6
    # the four modern examples build clean
    sys.path.insert(0, os.path.join(ROOT, "examples", "modern"))
    for name, dims in (("fighter_single", {"length": 15.06, "height": 5.09}), ("fighter_twin", {"length": 21.94, "width": 14.7}),
                       ("mbt_western", {"width": 3.75, "height": 3.0}), ("destroyer_aegis", {"length": 155.3, "width": 20.4})):
        mod = __import__(name)
        m = mod.build()
        rep = check.report(m)
        assert rep["ok"], (name, rep["issues"][:3], rep.get("floating"))
        for k, v in check.compare_size(m, dims).items():
            assert v[3], (name, k, v)
        g = m.props.get("_gear_report")
        if g:
            assert all(g[p]["outside"] == 0 for p in ("$gear_l", "$gear_r", "$gear_n")), (name, g)
        if "_tail_in_nozzles" in m.props:
            assert m.props["_tail_in_nozzles"]["tail"] == 0, m.props["_tail_in_nozzles"]
        print(name, "ok", rep["summary"]["tris"], "tris")
    print("modern kits ok")


def seethrough_and_shared_texels():
    """v0.5: the see-through check finds an opening with nothing behind it, texture_model gives a face that shares
    the texels of an opening texels of its own, hanging_shells finds parts under an airframe."""
    from meshkit import seethrough
    from meshkit.texture import texture_model, hanging_shells
    hole = lambda X, Y, Z: (np.abs(X) < 0.3) & (np.abs(Z) < 0.3) & (Y > 0.99)
    cut = lambda atlas, model: atlas.cut_alpha(hole, tags=["skin"])
    m = mk.Model("box"); m.add("a", mk.box(0, 0.5, 0, 1, 1, 1, tag="skin"))
    texture_model(m, (64, 64))
    assert seethrough.check(m, 8, 120)["open"] == 0
    texture_model(m, (64, 64), paint=cut)            # a hole in the lid, nothing under it: the inside shows
    r = seethrough.check(m, 8, 120)
    assert r["open"] > 0 and r["where"][0]["part"] == "a", seethrough.summary(r)
    # a long plate whose end lies under a thin lid with a hole through it (both skins of the lid, y > 0.98): the
    # plate shares the lid's texels in the plan view (too little of it is hidden to get a chart of its own), so the
    # hole would open the plate too, and in the view from below the plate owns the texels, so the lid's lower skin
    # would stay closed - unless texture_model charts those faces apart
    hole2 = lambda X, Y, Z: (np.abs(X) < 0.3) & (np.abs(Z) < 0.3) & (Y > 0.98)
    cut = lambda atlas, model: atlas.cut_alpha(hole2, tags=["skin"])
    s = mk.Model("shared")
    s.add("lid", mk.box(0, 1.0, 0, 1.0, 0.02, 1.0, tag="skin"))
    s.add("plate", mk.box(4.55, 0.5, 0, 10.0, 0.02, 0.9, tag="skin"))
    at = texture_model(s, (256, 256), paint=cut, fix_shared=False)
    assert ("plate", 0, 2) in at.hijacked(), at.hijacked()          # the plate's top face samples the hole
    bad = seethrough.check(s, 16, 160)["open"]
    at = texture_model(s, (256, 256), paint=cut)
    good = seethrough.check(s, 16, 160)["open"]
    reg = at.regions["top#solo#plate#0"]
    assert not at.hijacked() and at.alpha[reg.y0:reg.y0 + reg.h, reg.x0:reg.x0 + reg.w].min() > 254, at.solo
    # (what is left looks into the 2 cm thick lid through the edge of its hole: a hole cut by texture only has
    # no walls - real openings need a frame, a tub or a liner behind them, see AI_GUIDE 8.2)
    assert bad > 2 * good, (bad, good)
    # hanging_shells: a float under the body is one, a fin on top is not
    h = mk.Model("h")
    h.add("fuselage", mk.box(0, 2.0, 0, 1.0, 1.0, 6.0, tag="body"))
    h.add("float", mk.box(0, 0.4, 0, 0.6, 0.5, 4.0, tag="body"))
    h.add("fin", mk.box(0, 2.9, -2.5, 0.1, 1.0, 0.8, tag="body"))
    hs = hanging_shells(h, tags=("body",))
    assert ("float", 0) in hs and ("fin", 0) not in hs, hs
    print("see-through check / shared texels / hanging shells ok", {"open before": bad, "after": good})


def machinery_kit():
    from meshkit import machinery as mc
    # running gear: nothing overlaps; wheels are dropped rather than squeezed
    L = mc.crawler_layout(2.05, 0.68, 0.27, 0.11, 14, 0.07, 1)
    assert mc.layout_overlaps(L, gap=0.03) == [] and len(L["wheels"]) < 14, L["wheels"]
    # ram: stages from the stroke, leverage, mount search, exact joints
    pivot, axis = (0.0, 1.0, 1.6), (1, 0, 0)
    angles = [-80.0 * i / 24 for i in range(25)]
    assert mc.ram_stages(1.0, 1.6) == (2, True) and mc.ram_stages(1.0, 6.0)[0] > 2
    assert mc.ram_leverage((0, 1.0, 1.6), (0, 6.0, 1.6), pivot, axis, angles) < 1e-6     # through the hinge
    best = mc.ram_mount_search(pivot, axis, angles, [(0, 1.05, z / 10) for z in range(-12, 10)],
                               [(0, y / 4, 1.55) for y in range(12, 28)], min_arm=0.3)
    assert best and best[0][2] == 2 and best[0][5] >= 0.3, best[:1]
    m = mk.Model("rig")
    m.add("$upper", mk.box(0, 0.85, 0, 1.9, 0.3, 3.3, tag="frame"))
    m.add("$mast", mk.box(0, 4.75, 1.75, 0.4, 7.5, 0.3, tag="frame"))
    b, a, n, lmin, lmax, arm = best[0]
    parts = mc.ram(m, "$ram", b, a, 0.2, lmin, lmax)
    joints = [mc.joint("$mast", pivot, axis, channel="mast", parent="$upper", factor=80.0, offset=-80.0)]
    joints += mc.ram_joints(parts, "$upper", b, a, "$mast", axis)
    poses = mc.sweep_poses({"mast": (0.0, 1.0)})
    assert max(mc.ram_drift(joints, poses, [(parts[-1], a, "$mast")]).values()) < 1e-6
    # outriggers: the pad stays up until the beam is out
    mc.outriggers(m, mc.corner_mounts(1.3, -1.6, 0.95, 0.15), 0.9, 0.12, 1.4, 0.74, 0.2, 1.7, 1.8, parent="$upper")
    oj = mc.outriggers(mk.Model("x"), mc.corner_mounts(1.3, -1.6, 0.95, 0.15), 0.9, 0.12, 1.4, 0.74, 0.2, 1.7, 1.8,
                       parent="$upper")
    mats = mc.solve(oj, {"outrigger": 0.5})
    assert abs(mats["$outrigger_fl_leg"][1, 3] - mats["$outrigger_fl"][1, 3]) < 1e-9
    # clamp jaws: hollow where the member passes, open with the channel
    jj = mc.clamp_jaws(m, "$mast", 0.0, 2.35, 0.3, 0.1, 1.05, 1.25, lug_to=(0.12, 1.8))
    closed = mc.posed(m, jj, {"steady": 0.0})
    P = np.vstack([sh.V() for sh in closed.parts["$steady_l"].shells])
    r = np.hypot(P[:, 0], P[:, 2] - 2.35)
    assert r.min() > 0.2, r.min()
    opened = mc.posed(m, jj, {"steady": 1.0})
    assert np.vstack([sh.V() for sh in opened.parts["$steady_l"].shells])[:, 0].min() > P[:, 0].min() + 0.1
    print("machinery kit ok", {"wheels": len(L["wheels"]), "ram": best[0][2:]})


if __name__ == "__main__":
    main()
