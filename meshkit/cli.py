"""Command line:  python -m meshkit <command> ...

  run SCRIPT.py [-o DIR] [--formats obj,glb] [--views ...] [--mode ...] [--wire] [--no-contacts] [--strict]
        [--seethrough]
        Execute a modelling script (it defines build() returning a Model, or sets a global `model`),
        then check it, export it and render a preview sheet. Prints a JSON report. --seethrough adds the
        see-through check (report["seethrough"], <name>_seethrough.png).
  render FILE [-o PNG] [--views iso,left,top] [--mode texture|parts|solid|tags|orientation|xray] [--wire]
        [--size 520x340] [--cols 3] [--texture PNG] [--axes native|blender] [--parts a,b] [--skip a,b] [--zoom 1]
        [--json VEHICLE.json] [--states 'all;-$track_*,$wheel_*;=hull']
        --parts/--skip take glob patterns ($wheel_*). --json: a tudursvehiclemod vehicle JSON - the crawler tracks
        are placed around their path as in game. --states: one row of the views per visibility state ('all',
        '-patterns' hides, '=patterns' or 'patterns' shows only those) - for checking what lies behind wheels,
        tracks, skirts or the parts in front.
  check FILE [--no-contacts] [--texture PNG] [--axes ...]       closed-mesh / contact report (JSON)
  info FILE [--axes ...]                                         sizes, parts, triangle counts (JSON)
  convert IN OUT [--texture PNG] [--axes-in ...] [--axes-out ...]  obj <-> glb
  diff A B                                                       compare two models (sizes, parts, tris)
  silhouette FILE --view side|top|front -o PNG [--ref DRAWING.png --ref-box x0,y0,x1,y1]
        orthographic outline in model units (overlay on a reference drawing for checking proportions)
  seethrough FILE [--texture PNG] [--axes ...] [--directions 32] [--res 360] [--skip a,b] [-o PNG]
        pixels where the back of a surface is in view (a hole, an opening with nothing behind it, a face that
        shares the texels of an opening...), by part and place (JSON); -o: the worst views, magenta / cyan
        (--json VEHICLE.json places the crawler tracks first)
  texshare FILE --texture PNG [--parts ...] [--skip ...] [--json VEHICLE.json] [--tol 0.03] [--views 24] [-o PNG]
        visible surfaces that sample the same texels from different places (an interleaved wheel row painted with
        the wheel behind it, a mudguard with the track's picture...) - by part pair (JSON); -o: the texture with the
        conflicting texels in magenta
  clash FILE --json VEHICLE.json [--a '$track_*'] [--b hull] [--samples 2]
        where parts (default: the crawler tracks placed as in game) run inside closed shells of others (JSON)
"""
import argparse
import json
import os
import runpy
import sys
import time

import numpy as np


def _load(path, texture=None, axes="native"):
    from . import io_obj, io_gltf
    p = path.lower()
    if p.endswith(".obj"):
        m = io_obj.read_obj(path, axes=axes, texture=texture)
    elif p.endswith((".glb", ".gltf")):
        m = io_gltf.read_gltf(path)
        if texture:
            from PIL import Image
            m.texture = np.array(Image.open(texture).convert("RGBA"))
    else:
        raise SystemExit(f"unsupported file: {path}")
    return m


def _with_json(m, path):
    """Place the crawler tracks of a tudursvehiclemod vehicle JSON (no-op without one)."""
    if not path:
        return m
    from . import vehicle
    return vehicle.bake_tracks(m, vehicle.load_json(path))


def _size(s):
    w, h = s.lower().split("x")
    return int(w), int(h)


def _views(s):
    return [v.strip() for v in s.split(",") if v.strip()]


def _json(o):
    def conv(x):
        if isinstance(x, (np.floating,)):
            return float(x)
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, tuple):
            return list(x)
        raise TypeError(type(x))
    return json.dumps(o, ensure_ascii=False, indent=1, default=conv)


def run_script(script, out_dir=None, formats=("obj", "glb"), views=None, mode="texture", wire=False, contacts=True,
               size=(520, 340), name=None, render=True, seethrough=False):
    """Execute a modelling script and return (model, report). Used by the CLI and the MCP server."""
    from . import check, io_obj, io_gltf, view
    from .geom import Model
    t0 = time.time()
    here = os.path.dirname(os.path.abspath(script))
    sys.path.insert(0, here)
    try:
        ns = runpy.run_path(script, run_name="__meshkit__")
    finally:
        if sys.path and sys.path[0] == here:
            sys.path.pop(0)
    if "build" in ns and callable(ns["build"]):
        model = ns["build"]()
    else:
        model = ns.get("model")
    if isinstance(model, dict):          # ww2gen-style dict {"model": ..., ...}
        model = model["model"]
    if not isinstance(model, Model) and not hasattr(model, "parts"):
        raise SystemExit("the script must define build() returning a meshkit Model (or set `model`)")
    for attr, default in (("pivots", {}), ("props", {}), ("refs", []), ("uvs", None), ("texture", None)):
        if not hasattr(model, attr):
            setattr(model, attr, default)
    name = name or model.name or os.path.splitext(os.path.basename(script))[0]
    out_dir = out_dir or os.path.join(here, "out")
    os.makedirs(out_dir, exist_ok=True)
    rep = check.report(model, contacts=contacts)
    rep["outputs"] = {}
    if "obj" in formats:
        p = os.path.join(out_dir, name + ".obj")
        io_obj.write_obj(model, p, uvs=model.uvs)
        rep["outputs"]["obj"] = p
        if model.texture is not None:
            from PIL import Image
            tp = os.path.join(out_dir, name + ".png")
            Image.fromarray(model.texture).save(tp)
            rep["outputs"]["texture"] = tp
    if "glb" in formats:
        p = os.path.join(out_dir, name + ".glb")
        io_gltf.write_glb(model, p)
        rep["outputs"]["glb"] = p
    if render:
        p = os.path.join(out_dir, name + "_preview.png")
        kw = {}
        if views:
            kw["views"] = views
        view.sheet(model, W=size[0], H=size[1], mode=mode, wire=wire, **kw).save(p)
        rep["outputs"]["preview"] = p
    if seethrough:
        from . import seethrough as st
        sr = st.check(model)
        rep["seethrough"] = st.summary(sr)
        if sr["worst"]:
            p = os.path.join(out_dir, name + "_seethrough.png")
            st.sheet(model, sr).save(p)
            rep["outputs"]["seethrough"] = p
    rep["seconds"] = round(time.time() - t0, 2)
    return model, rep


def main(argv=None):
    ap = argparse.ArgumentParser(prog="meshkit", description="Blender-free modelling toolkit for AI agents")
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("run"); a.add_argument("script"); a.add_argument("-o", "--out")
    a.add_argument("--formats", default="obj,glb"); a.add_argument("--views"); a.add_argument("--mode", default="texture")
    a.add_argument("--wire", action="store_true"); a.add_argument("--no-contacts", action="store_true")
    a.add_argument("--size", default="520x340"); a.add_argument("--name"); a.add_argument("--strict", action="store_true")
    a.add_argument("--no-render", action="store_true")
    a.add_argument("--seethrough", action="store_true", help="also run the see-through check (adds ~10-30 s)")

    r = sub.add_parser("render"); r.add_argument("file"); r.add_argument("-o", "--out")
    r.add_argument("--views", default="iso,iso_rear,left,top,front,iso_below"); r.add_argument("--mode", default="texture")
    r.add_argument("--wire", action="store_true"); r.add_argument("--size", default="520x340"); r.add_argument("--cols", type=int, default=3)
    r.add_argument("--texture"); r.add_argument("--axes", default="native"); r.add_argument("--parts"); r.add_argument("--skip")
    r.add_argument("--zoom", type=float, default=1.0); r.add_argument("--json"); r.add_argument("--states")

    c = sub.add_parser("check"); c.add_argument("file"); c.add_argument("--no-contacts", action="store_true")
    c.add_argument("--texture"); c.add_argument("--axes", default="native")

    i = sub.add_parser("info"); i.add_argument("file"); i.add_argument("--axes", default="native")

    v = sub.add_parser("convert"); v.add_argument("src"); v.add_argument("dst"); v.add_argument("--texture")
    v.add_argument("--axes-in", default="native"); v.add_argument("--axes-out", default="native")

    d = sub.add_parser("diff"); d.add_argument("a"); d.add_argument("b")

    s = sub.add_parser("silhouette"); s.add_argument("file"); s.add_argument("--view", default="side")
    s.add_argument("-o", "--out", required=True); s.add_argument("--ref"); s.add_argument("--ref-box")
    s.add_argument("--axes", default="native")

    t = sub.add_parser("seethrough"); t.add_argument("file"); t.add_argument("--texture"); t.add_argument("--axes", default="native")
    t.add_argument("--directions", type=int, default=32); t.add_argument("--res", type=int, default=360)
    t.add_argument("--skip"); t.add_argument("-o", "--out"); t.add_argument("--json")

    x = sub.add_parser("texshare"); x.add_argument("file"); x.add_argument("--texture"); x.add_argument("--axes", default="native")
    x.add_argument("--parts"); x.add_argument("--skip"); x.add_argument("--json"); x.add_argument("--tol", type=float, default=0.03)
    x.add_argument("--views", type=int, default=24); x.add_argument("--res", type=int, default=300)
    x.add_argument("--no-mirror", action="store_true"); x.add_argument("-o", "--out")

    k = sub.add_parser("clash"); k.add_argument("file"); k.add_argument("--json", required=True); k.add_argument("--axes", default="native")
    k.add_argument("--a"); k.add_argument("--b"); k.add_argument("--samples", type=int, default=2)

    ns = ap.parse_args(argv)
    from . import check, view, io_obj, io_gltf

    if ns.cmd == "run":
        _, rep = run_script(ns.script, ns.out, _views(ns.formats), _views(ns.views) if ns.views else None, ns.mode,
                            ns.wire, not ns.no_contacts, _size(ns.size), ns.name, not ns.no_render, ns.seethrough)
        print(_json(rep))
        if ns.strict and not rep["ok"]:
            sys.exit(1)
    elif ns.cmd == "render":
        from . import vehicle
        m = _with_json(_load(ns.file, ns.texture, ns.axes), ns.json)
        out = ns.out or os.path.splitext(ns.file)[0] + "_preview.png"
        W, H = _size(ns.size)
        kw = {}
        if ns.parts:
            kw["only"] = set(vehicle.select(m.order, ns.parts))
        if ns.skip:
            kw["skip"] = set(vehicle.select(m.order, ns.skip))
        if ns.states:
            view.states_sheet(m, vehicle.parse_states(ns.states, m.order), _views(ns.views), W, H, ns.mode, ns.wire,
                              zoom=ns.zoom, **kw).save(out)
        else:
            view.sheet(m, _views(ns.views), W, H, ns.mode, ns.wire, cols=ns.cols, zoom=ns.zoom, **kw).save(out)
        print(out)
    elif ns.cmd == "check":
        m = _load(ns.file, ns.texture, ns.axes)
        rep = check.report(m, contacts=not ns.no_contacts)
        print(_json(rep))
        sys.exit(0 if rep["ok"] else 1)
    elif ns.cmd == "info":
        m = _load(ns.file, axes=ns.axes)
        s = m.summary()
        s["has_uvs"] = m.uvs is not None; s["texture"] = None if m.texture is None else list(m.texture.shape)
        s["refs"] = [[n, [round(float(c), 4) for c in p], k] for n, p, k in m.refs]
        print(_json(s))
    elif ns.cmd == "convert":
        m = _load(ns.src, ns.texture, ns.axes_in)
        if ns.dst.lower().endswith(".obj"):
            io_obj.write_obj(m, ns.dst, axes=ns.axes_out)
            if m.texture is not None:
                from PIL import Image
                png = os.path.splitext(ns.dst)[0] + ".png"
                Image.fromarray(m.texture).save(png)
                mtl = os.path.splitext(ns.dst)[0] + ".mtl"
                io_obj.write_mtl(mtl, "material", os.path.basename(png))
                # rewrite with mtllib so other tools find the texture
                io_obj.write_obj(m, ns.dst, axes=ns.axes_out, mtl=(os.path.basename(mtl), "material"))
        elif ns.dst.lower().endswith(".glb"):
            io_gltf.write_glb(m, ns.dst)
        else:
            raise SystemExit("destination must be .obj or .glb")
        print(ns.dst)
    elif ns.cmd == "diff":
        A, B = _load(ns.a), _load(ns.b)
        sa, sb = A.summary(), B.summary()
        out = {"size": [sa["size"], sb["size"]], "tris": [sa["tris"], sb["tris"]],
               "only_in_a": [p for p in A.order if p not in B.parts], "only_in_b": [p for p in B.order if p not in A.parts],
               "changed_parts": {}}
        for p in A.order:
            if p in B.parts:
                ta, tb = sa["per_part"][p]["tris"], sb["per_part"][p]["tris"]
                la, ha = A.bbox([p]); lb, hb = B.bbox([p])
                dv = float(max(np.abs(la - lb).max(), np.abs(ha - hb).max()))
                if ta != tb or dv > 1e-4:
                    out["changed_parts"][p] = {"tris": [ta, tb], "bbox_shift": round(dv, 4)}
        print(_json(out))
    elif ns.cmd == "seethrough":
        from . import seethrough, vehicle
        m = _with_json(_load(ns.file, ns.texture, ns.axes), ns.json)
        rep = seethrough.check(m, ns.directions, ns.res, skip=set(vehicle.select(m.order, ns.skip)) if ns.skip else ())
        out = seethrough.summary(rep)
        if ns.out and rep["worst"]:
            seethrough.sheet(m, rep).save(ns.out)
            out["image"] = ns.out
        print(_json(out))
    elif ns.cmd == "texshare":
        from . import texshare, vehicle
        m = _with_json(_load(ns.file, ns.texture, ns.axes), ns.json)
        skip = set(vehicle.select(m.order, ns.skip)) if ns.skip else set()
        if ns.parts:
            keep = set(vehicle.select(m.order, ns.parts))
            only = keep
        else:
            only = None
        rep = texshare.check(m, ns.views, ns.res, skip=skip, tol=ns.tol, mirror_ok=not ns.no_mirror, only=only)
        out = texshare.summary(rep)
        if ns.out:
            texshare.sheet(m, rep).save(ns.out)
            out["image"] = ns.out
        print(_json(out))
    elif ns.cmd == "clash":
        from . import vehicle
        m0 = _load(ns.file, axes=ns.axes)
        js = vehicle.load_json(ns.json)
        m = vehicle.bake_tracks(m0, js)
        a = vehicle.select(m.order, ns.a) if ns.a else [t["part"] for t in js.get("crawler_tracks", []) if t["part"] in m.parts]
        b = vehicle.select(m.order, ns.b) if ns.b else [p for p in m.order if p not in a and not p.startswith("$")]
        print(_json(vehicle.clash(m, a, b, ns.samples)))
    elif ns.cmd == "silhouette":
        from . import silhouette
        m = _load(ns.file, axes=ns.axes)
        paths, lo, hi = silhouette.outline(m, ns.view)
        img = silhouette.draw(paths, lo, hi, ns.ref, [float(x) for x in ns.ref_box.split(",")] if ns.ref_box else None)
        img.save(ns.out)
        print(_json({"out": ns.out, "lo": lo, "hi": hi, "paths": len(paths)}))


if __name__ == "__main__":
    main()
