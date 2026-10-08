"""meshkit verification - checks that meshkit works the same in THIS environment as in the one that made the
reference data (verify/reference/).  Needs only numpy and Pillow.

    python verify/verify.py                 run every check, write verify/result/ (report + comparison sheets)
    python verify/verify.py --quick         only the calibration card and the light tank
    python verify/verify.py --blender DIR   also compare Blender renders made with verify/blender_render.py
    python verify/verify.py --make-reference --yes
                                            (maintainers) rebuild models/ and reference/ from the current code

Checks
  1 build   each model script is run again; the OBJ written now must equal the stored OBJ (models/) and the
            texture must equal the stored PNG pixel for pixel
  2 load    the stored OBJ (+ PNG) and GLB are read back: triangle / part counts, size and pivots as recorded
  3 render  the stored models are drawn with fixed cameras and display modes and compared with the reference
            images pixel by pixel (the model read from GLB is drawn too and must give the same picture)
  4 cli     python -m meshkit info / check / convert run as separate processes
  5 blender (optional) silhouettes of Blender renders vs the meshkit renders (IoU)

Verdicts: MATCH = identical, OK = within tolerance (see TOL), FAIL.  Exit code 0 when nothing failed."""
import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time

import numpy as np
from PIL import Image

VERIFY = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(VERIFY)
sys.path.insert(0, ROOT)
import meshkit as mk  # noqa: E402
from meshkit import check, cli, io_obj, view  # noqa: E402
from meshkit.render import contact_sheet  # noqa: E402

MODELS_DIR = os.path.join(VERIFY, "models")
REF_DIR = os.path.join(VERIFY, "reference")
RES_DIR = os.path.join(VERIFY, "result")
W, H = 480, 320
BG = np.array(view.BG, np.uint8)

# name -> modelling script (relative to the meshkit folder)
MODELS = {
    "calib_card": "verify/models/calib_card.py",
    "light_tank": "examples/light_tank.py",
    "j7w1": "examples/ww2_projects/j7w1.py",
    "kikka": "examples/ww2_projects/kikka.py",
    "j8m1": "examples/ww2_projects/j8m1.py",
    "xb42": "examples/ww2_projects/xb42.py",
    "xf15c": "examples/ww2_projects/xf15c.py",
    "xf5u": "examples/ww2_projects/xf5u.py",
}
QUICK = ("calib_card", "light_tank")
VIEWS6 = ("iso", "iso_rear", "iso_below", "left", "top", "front")


def render_list(name):
    """(view, mode) pairs drawn for a model.  mode tokens: texture/parts/solid/tags/orientation/xray, '+wire', '+refs'."""
    if name == "calib_card":
        out = [(v, m) for v in VIEWS6 for m in ("texture", "parts", "orientation")]
        out += [("iso", "solid"), ("iso", "tags"), ("iso", "xray"), ("iso", "texture+wire"), ("iso", "texture+refs"),
                ("back", "texture"), ("right", "texture"), ("bottom", "texture"), ("iso_left", "texture"), ("iso_right", "texture")]
        return out
    return [(v, "texture") for v in VIEWS6] + [("iso", "parts"), ("iso", "orientation"), ("left", "solid+wire")]


# a pose for the hinged / moving parts, so that pivots are exercised (rotation about the stored pivot)
POSES = {"calib_card": {"$lid": ((1, 0, 0), 35.0)}, "light_tank": {"$turret": ((0, 1, 0), 30.0)}}

TOL = {"pixel_level": 8,          # a pixel counts as different when a channel differs by more than this
       "max_diff_fraction": 0.002,  # OK when at most 0.2 % of the pixels differ ...
       "min_iou": 0.995,            # ... and the silhouettes overlap this much
       "obj_coord": 1e-5,           # OK when the rebuilt OBJ differs only by this much in coordinates / UVs
       "blender_iou": 0.90}         # Blender vs meshkit silhouettes (different renderers - outline only)


# ------------------------------------------------------------------------------------------------ helpers
def sha(data):
    return hashlib.sha256(data).hexdigest()[:16]


def pose_transforms(model, name):
    from meshkit.geom import rot_axis
    out = {}
    for part, (axis, deg) in POSES.get(name, {}).items():
        if part in model.parts and part in model.pivots:
            M = rot_axis(axis, deg); pv = np.asarray(model.pivots[part], float)
            out[part] = (lambda Q, M=M, pv=pv: (Q - pv) @ M.T + pv)
    return out


def draw(model, name, v, mode):
    base = mode.split("+")[0]
    tf = pose_transforms(model, name) if v == "iso_rear" else None      # one posed view per model
    return view.render_view(model, v, W, H, mode=base, wire="+wire" in mode, refs="+refs" in mode, transforms=tf)


def img_file(name, v, mode):
    return f"{v}_{mode.replace('+', '-')}.png"


def compare_img(ref, cur):
    if ref.shape != cur.shape:
        return {"verdict": "FAIL", "reason": f"size {cur.shape} != {ref.shape}"}
    d = np.abs(ref.astype(np.int16) - cur.astype(np.int16)).max(2)
    frac = float((d > TOL["pixel_level"]).mean())
    ma = np.any(ref != BG, 2); mb = np.any(cur != BG, 2)
    iou = float((ma & mb).sum() / max((ma | mb).sum(), 1))
    if not d.any():
        verdict = "MATCH"
    elif frac <= TOL["max_diff_fraction"] and iou >= TOL["min_iou"]:
        verdict = "OK"
    else:
        verdict = "FAIL"
    return {"verdict": verdict, "diff_pixels_%": round(frac * 100, 4), "max_channel_diff": int(d.max()), "iou": round(iou, 5)}


def diff_image(ref, cur):
    g = (ref.astype(np.float32).mean(2, keepdims=True) * 0.45 + 120).repeat(3, 2)
    d = np.abs(ref.astype(np.int16) - cur.astype(np.int16)).max(2)
    g[d > 0] = (255, 170, 0)
    g[d > TOL["pixel_level"]] = (230, 0, 0)
    return g.clip(0, 255).astype(np.uint8)


def obj_lines(path):
    with open(path, encoding="utf-8") as f:
        return [l.rstrip("\n") for l in f if not l.startswith("#")]


def compare_obj(a, b):
    A, B = obj_lines(a), obj_lines(b)
    if A == B:
        return {"verdict": "MATCH", "lines": len(A)}
    res = {"lines": [len(A), len(B)]}
    nums = {}
    for key in ("v", "vt", "vn"):
        x = [l.split()[1:] for l in A if l.startswith(key + " ")]; y = [l.split()[1:] for l in B if l.startswith(key + " ")]
        if len(x) != len(y):
            res["verdict"] = "FAIL"; res["reason"] = f"{key} count {len(x)} != {len(y)}"
            return res
        if x:
            nums[key] = float(np.abs(np.array(x, float) - np.array(y, float)).max())
    fa = [l for l in A if l.startswith(("f ", "o ", "g ", "usemtl", "mtllib"))]
    fb = [l for l in B if l.startswith(("f ", "o ", "g ", "usemtl", "mtllib"))]
    res["max_diff"] = nums
    res["faces_identical"] = fa == fb
    ok = fa == fb and max(nums.get("v", 0), nums.get("vt", 0)) <= TOL["obj_coord"] and nums.get("vn", 0) <= 1e-3
    res["verdict"] = "OK" if ok else "FAIL"
    return res


def build(name):
    model, rep = cli.run_script(os.path.join(ROOT, MODELS[name]), os.path.join(RES_DIR, "_tmp"), formats=(), render=False)
    return model, rep


def model_facts(m):
    s = m.summary()
    return {"tris": s["tris"], "parts": s["parts"], "size": [round(float(v), 4) for v in s["size"]],
            "order": list(m.order), "pivots": {k: [round(float(c), 4) for c in v] for k, v in sorted(m.pivots.items())},
            "refs": len(m.refs), "texture": None if m.texture is None else list(m.texture.shape)}


def env_info():
    import PIL
    return {"python": platform.python_version(), "numpy": np.__version__, "Pillow": PIL.__version__,
            "meshkit": mk.__version__, "os": platform.platform(), "machine": platform.machine()}


# ------------------------------------------------------------------------------------------------ reference
def make_reference(names):
    os.makedirs(MODELS_DIR, exist_ok=True)
    if os.path.isdir(REF_DIR):
        shutil.rmtree(REF_DIR)
    os.makedirs(REF_DIR)
    ref = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "environment": env_info(), "size": [W, H],
           "tolerance": TOL, "models": {}}
    for name in names:
        t = time.time()
        model, rep = build(name)
        if not rep["ok"]:
            raise SystemExit(f"{name}: check failed {rep['issues'][:5]} floating={rep.get('floating')}")
        obj = os.path.join(MODELS_DIR, name + ".obj")
        io_obj.write_obj(model, obj, uvs=model.uvs, header=f"# meshkit verification model '{name}'",
                         mtl=(name + ".mtl", "material"))
        io_obj.write_mtl(os.path.join(MODELS_DIR, name + ".mtl"), "material", name + ".png")
        Image.fromarray(model.texture).save(os.path.join(MODELS_DIR, name + ".png"))
        mk.write_glb(model, os.path.join(MODELS_DIR, name + ".glb"))
        m = mk.load(obj)
        m.pivots = dict(mk.load(os.path.join(MODELS_DIR, name + ".glb")).pivots)     # OBJ has no pivots
        d = os.path.join(REF_DIR, name); os.makedirs(d)
        renders = {}
        imgs, labels = [], []
        for v, mode in render_list(name):
            img = draw(m, name, v, mode)
            fn = img_file(name, v, mode)
            Image.fromarray(img).save(os.path.join(d, fn))
            cam = view.camera(m, v, W, H, transforms=pose_transforms(m, name) if v == "iso_rear" else None)
            renders[fn] = {"view": v, "mode": mode, "pixels_sha": sha(img.tobytes()),
                           "camera": {k: (list(map(float, c)) if isinstance(c, (tuple, list, np.ndarray)) else float(c))
                                      for k, c in cam.items()}}
            imgs.append(img); labels.append(f"{v} / {mode}")
        g = mk.load(os.path.join(MODELS_DIR, name + ".glb"))
        img = draw(g, name, "iso", "texture")
        Image.fromarray(img).save(os.path.join(d, "glb_iso_texture.png"))
        imgs.append(img); labels.append("iso / texture (read from GLB)")
        contact_sheet(imgs, labels, cols=4, title=f"{name} - meshkit reference ({ref['created']})").save(
            os.path.join(REF_DIR, f"sheet_{name}.png"))
        ref["models"][name] = {"script": MODELS[name], "facts": model_facts(model), "check_ok": rep["ok"],
                               "obj_sha": sha(open(obj, "rb").read()), "texture_sha": sha(model.texture.tobytes()),
                               "renders": renders}
        print(f"{name}: {len(renders)} reference images  ({time.time() - t:.1f} s)")
    with open(os.path.join(REF_DIR, "reference.json"), "w", encoding="utf-8") as f:
        json.dump(ref, f, indent=1, ensure_ascii=False)
    print("reference written to", REF_DIR)


# ------------------------------------------------------------------------------------------------ verify
def verify(names, blender_dir=None):
    with open(os.path.join(REF_DIR, "reference.json"), encoding="utf-8") as f:
        ref = json.load(f)
    if os.path.isdir(RES_DIR):
        shutil.rmtree(RES_DIR)
    os.makedirs(os.path.join(RES_DIR, "compare")); os.makedirs(os.path.join(RES_DIR, "render"))
    report = {"date": time.strftime("%Y-%m-%d %H:%M:%S"), "environment": env_info(),
              "reference_environment": ref["environment"], "checks": {}}
    counts = {"MATCH": 0, "OK": 0, "FAIL": 0}

    def put(key, res):
        report["checks"][key] = res
        counts[res["verdict"]] += 1
        mark = {"MATCH": "MATCH", "OK": "OK   ", "FAIL": "FAIL "}[res["verdict"]]
        extra = {k: v for k, v in res.items() if k not in ("verdict",)}
        print(f"  [{mark}] {key}  {json.dumps(extra, ensure_ascii=False) if res['verdict'] != 'MATCH' else ''}")

    for name in names:
        rm = ref["models"][name]
        print(f"== {name}")
        # 1 build
        try:
            model, rep = build(name)
            tmp = os.path.join(RES_DIR, "_tmp", name + ".obj")
            io_obj.write_obj(model, tmp, uvs=model.uvs, header="", mtl=(name + ".mtl", "material"))
            r = compare_obj(os.path.join(MODELS_DIR, name + ".obj"), tmp)
            r["check_ok"] = rep["ok"]
            if not rep["ok"]:
                r["verdict"] = "FAIL"
            put(f"{name}/build/obj", r)
            stored = np.array(Image.open(os.path.join(MODELS_DIR, name + ".png")))
            if stored.shape != model.texture.shape:
                put(f"{name}/build/texture", {"verdict": "FAIL", "reason": f"shape {model.texture.shape} != {stored.shape}"})
            else:
                dt = int(np.abs(stored.astype(np.int16) - model.texture.astype(np.int16)).max())
                put(f"{name}/build/texture", {"verdict": "MATCH" if dt == 0 else ("OK" if dt <= 1 else "FAIL"), "max_diff": dt})
        except Exception as e:      # noqa: BLE001 - report and go on
            put(f"{name}/build", {"verdict": "FAIL", "error": repr(e)})
        # 2 load
        loaded = {}
        for ext in ("obj", "glb"):
            try:
                m = mk.load(os.path.join(MODELS_DIR, f"{name}.{ext}"))
                loaded[ext] = m
                facts = model_facts(m); want = rm["facts"]
                keys = ("tris", "parts", "order", "texture") + (("refs",) if ext == "glb" else ())   # OBJ keeps no refs
                bad = [k for k in keys if facts[k] != want[k]]
                if max(abs(a - b) for a, b in zip(facts["size"], want["size"])) > 1e-3:
                    bad.append("size")
                if ext == "glb":
                    pv = [k for k in want["pivots"] if k not in facts["pivots"]
                          or max(abs(a - b) for a, b in zip(facts["pivots"][k], want["pivots"][k])) > 1e-4]
                    if pv:
                        bad.append(f"pivots {pv}")
                rc = check.report(m)
                if rc["issue_count"]:
                    bad.append(f"check issues {rc['issue_count']}")
                put(f"{name}/load/{ext}", {"verdict": "FAIL" if bad else "MATCH", **({"mismatch": bad} if bad else {})})
            except Exception as e:  # noqa: BLE001
                put(f"{name}/load/{ext}", {"verdict": "FAIL", "error": repr(e)})
        # 3 render
        if "obj" in loaded:
            m = loaded["obj"]
            # pivots are not stored in OBJ: take them from the GLB so the posed view matches
            if "glb" in loaded:
                m.pivots = dict(loaded["glb"].pivots)
            rows, labels = [], []
            for v, mode in render_list(name):
                fn = img_file(name, v, mode)
                refimg = np.array(Image.open(os.path.join(REF_DIR, name, fn)).convert("RGB"))
                cur = draw(m, name, v, mode)
                Image.fromarray(cur).save(os.path.join(RES_DIR, "render", f"{name}_{fn}"))
                r = compare_img(refimg, cur)
                put(f"{name}/render/{fn[:-4]}", r)
                rows += [refimg, cur, diff_image(refimg, cur)]
                labels += [f"reference {v}/{mode}", f"now  [{r['verdict']}]", "difference (red = differs)"]
            contact_sheet(rows, labels, cols=3, title=f"{name}: reference | this environment | difference").save(
                os.path.join(RES_DIR, "compare", f"{name}.png"))
            if "glb" in loaded:
                refimg = np.array(Image.open(os.path.join(REF_DIR, name, "glb_iso_texture.png")).convert("RGB"))
                put(f"{name}/render/glb_iso_texture", compare_img(refimg, draw(loaded["glb"], name, "iso", "texture")))
        # 5 blender
        if blender_dir:
            rows, labels = [], []
            for fn, rinfo in rm["renders"].items():
                bp = os.path.join(blender_dir, f"{name}_{rinfo['view']}.png")
                if rinfo["mode"] != "texture" or not os.path.exists(bp):
                    continue
                bi = Image.open(bp).convert("RGBA")
                b = np.array(bi)
                mb = b[..., 3] > 127                                    # transparent film: alpha = coverage
                over = Image.new("RGBA", bi.size, tuple(int(c) for c in BG) + (255,)); over.alpha_composite(bi)
                refimg = np.array(Image.open(os.path.join(REF_DIR, name, fn)).convert("RGB"))
                ma = np.any(refimg != BG, 2)
                iou = float((ma & mb).sum() / max((ma | mb).sum(), 1))
                put(f"{name}/blender/{rinfo['view']}", {"verdict": "OK" if iou >= TOL["blender_iou"] else "FAIL", "iou": round(iou, 4)})
                ov = np.full(refimg.shape, 235, np.uint8)
                ov[ma & mb] = (120, 120, 120); ov[ma & ~mb] = (220, 40, 40); ov[mb & ~ma] = (40, 90, 230)
                rows += [refimg, np.array(over.convert("RGB")), ov]
                labels += [f"meshkit {rinfo['view']}", "Blender (Workbench)", f"outline IoU {iou:.3f} (red: meshkit only, blue: Blender only)"]
            if rows:
                contact_sheet(rows, labels, cols=3, title=f"{name}: meshkit | Blender | outlines").save(
                    os.path.join(RES_DIR, "compare", f"blender_{name}.png"))
    # 4 cli (separate processes, calib card only)
    env = dict(os.environ); env["PYTHONPATH"] = ROOT + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONIOENCODING"] = "utf-8"
    glb = os.path.join(MODELS_DIR, "calib_card.glb")

    def run(*args):
        return subprocess.run([sys.executable, "-m", "meshkit", *args], capture_output=True, text=True, encoding="utf-8",
                              errors="replace", env=env, cwd=ROOT)
    try:
        p = run("info", glb)
        info = json.loads(p.stdout)
        want = ref["models"]["calib_card"]["facts"]
        ok = p.returncode == 0 and info["tris"] == want["tris"] and info["parts"] == want["parts"]
        put("cli/info", {"verdict": "MATCH" if ok else "FAIL", **({} if ok else {"stdout": p.stdout[-300:], "stderr": p.stderr[-300:]})})
        p = run("check", glb)
        put("cli/check", {"verdict": "MATCH" if p.returncode == 0 else "FAIL", **({} if p.returncode == 0 else {"stderr": p.stderr[-300:]})})
        out = os.path.join(RES_DIR, "_tmp", "cli_convert.glb")
        p = run("convert", os.path.join(MODELS_DIR, "calib_card.obj"), out)
        refimg = np.array(Image.open(os.path.join(REF_DIR, "calib_card", "glb_iso_texture.png")).convert("RGB"))
        if p.returncode != 0:
            put("cli/convert", {"verdict": "FAIL", "stderr": p.stderr[-300:]})
        else:
            put("cli/convert", compare_img(refimg, draw(mk.load(out), "calib_card", "iso", "texture")))
    except Exception as e:  # noqa: BLE001
        put("cli", {"verdict": "FAIL", "error": repr(e)})
    shutil.rmtree(os.path.join(RES_DIR, "_tmp"), ignore_errors=True)
    report["summary"] = counts
    report["result"] = "PASS" if counts["FAIL"] == 0 else "FAIL"
    with open(os.path.join(RES_DIR, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1, ensure_ascii=False)
    e = report["environment"]; r = ref["environment"]
    lines = [f"meshkit 動作確認  {report['date']}",
             f"この環境     : Python {e['python']} / numpy {e['numpy']} / Pillow {e['Pillow']} / {e['os']}",
             f"基準データ   : Python {r['python']} / numpy {r['numpy']} / Pillow {r['Pillow']} / {r['os']}",
             f"結果         : {report['result']}  (完全一致 {counts['MATCH']} / 許容内 {counts['OK']} / 不一致 {counts['FAIL']})"]
    fails = [k for k, v in report["checks"].items() if v["verdict"] == "FAIL"]
    if fails:
        lines.append("不一致の項目 : " + ", ".join(fails))
    lines.append("比較画像     : verify/result/compare/<モデル名>.png（左：基準、中：この環境、右：差分）")
    with open(os.path.join(RES_DIR, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    return report["result"] == "PASS"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="only " + ", ".join(QUICK))
    ap.add_argument("--blender", metavar="DIR", help="folder with the images written by blender_render.py")
    ap.add_argument("--make-reference", action="store_true", help="rebuild models/ and reference/ (maintainers)")
    ap.add_argument("--yes", action="store_true", help="confirm --make-reference")
    ap.add_argument("models", nargs="*", help="model names (default: all)")
    a = ap.parse_args()
    names = a.models or (list(QUICK) if a.quick else list(MODELS))
    if a.make_reference:
        if not a.yes:
            raise SystemExit("--make-reference overwrites the reference data; add --yes to confirm")
        make_reference(names)
    else:
        sys.exit(0 if verify(names, a.blender) else 1)
