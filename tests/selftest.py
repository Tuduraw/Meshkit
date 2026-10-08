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


if __name__ == "__main__":
    main()
