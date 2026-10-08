"""Regression: rebuild ww2gen vehicles WITHOUT Blender, write the OBJ with meshkit and compare it
with the OBJ that Blender 4.5 exported for the released pack.

    python tests/compare_blender.py <ww2gen dir> <reference obj root> a6m2 chiha ...

Reports, per vehicle: identical lines, max vertex / UV / normal difference, face-structure match."""
import importlib
import os
import sys
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from meshkit import io_obj  # noqa: E402


def build_ww2gen(ww2gen, vid):
    sys.path[:0] = [os.path.join(ww2gen, "lib"), ww2gen]
    import texture
    V = importlib.import_module(f"vehicles.{vid}").build()
    model = V["model"]
    atlas = texture.Atlas(V["tex_size"], cell=V.get("cell", 8))
    atlas.split_sides = V.get("split_sides", False)
    atlas.glass_tags = tuple(V.get("glass_tags", ()))
    for tag, col in V["flat"].items():
        atlas.set_flat(tag, col)
    atlas.collect(model); atlas.layout(margin=V.get("tex_margin", 0.06)); atlas.rasterize()
    return model, atlas.uvs(), V.get("meta", {})


def find_ref(root, vid):
    for d, _, fs in os.walk(root):
        if vid + ".obj" in fs:
            return os.path.join(d, vid + ".obj")


def compare(a_path, b_path):
    A = open(a_path).read().splitlines(); B = open(b_path).read().splitlines()
    A = [l for l in A if not l.startswith("#")]; B = [l for l in B if not l.startswith("#")]
    same = sum(1 for x, y in zip(A, B) if x == y)
    res = {"lines": (len(A), len(B)), "identical_lines": same}
    for key, n in (("v", 3), ("vt", 2), ("vn", 3)):
        a = np.array([list(map(float, l.split()[1:1 + n])) for l in A if l.startswith(key + " ")])
        b = np.array([list(map(float, l.split()[1:1 + n])) for l in B if l.startswith(key + " ")])
        res[key] = (len(a), len(b), float(np.abs(a - b).max()) if a.shape == b.shape and len(a) else None)
    # geometric comparison of the faces: resolve every corner to (position, uv, normal) values
    def corners(L):
        v = [list(map(float, l.split()[1:4])) for l in L if l.startswith("v ")]
        t = [list(map(float, l.split()[1:3])) for l in L if l.startswith("vt ")]
        n = [list(map(float, l.split()[1:4])) for l in L if l.startswith("vn ")]
        out = []
        for l in L:
            if l.startswith("f "):
                for tok in l.split()[1:]:
                    i = tok.split("/")
                    out.append(v[int(i[0]) - 1] + (t[int(i[1]) - 1] if len(i) > 1 and i[1] else [0, 0])
                               + (n[int(i[2]) - 1] if len(i) > 2 and i[2] else [0, 0, 0]))
        return np.array(out)
    ca, cb = corners(A), corners(B)
    if ca.shape == cb.shape:
        d = np.abs(ca - cb).max(0)
        res["corner_max_diff(pos,uv,normal)"] = (round(float(d[:3].max()), 7), round(float(d[3:5].max()), 7), round(float(d[5:].max()), 5))
    else:
        res["corner_count"] = (len(ca), len(cb))
    fa = [l for l in A if l.startswith(("f ", "o "))]; fb = [l for l in B if l.startswith(("f ", "o "))]
    res["faces_identical"] = fa == fb
    if fa != fb:
        res["face_line_diff"] = sum(1 for x, y in zip(fa, fb) if x != y)
    return res


if __name__ == "__main__":
    ww2gen, refroot = sys.argv[1], sys.argv[2]
    out = os.path.join(HERE, "_out"); os.makedirs(out, exist_ok=True)
    for vid in sys.argv[3:]:
        t = time.time()
        model, uvs, meta = build_ww2gen(ww2gen, vid)
        p = io_obj.write_obj(model, os.path.join(out, vid + ".obj"), uvs=uvs, header="")
        r = compare(p, find_ref(refroot, vid))
        print(vid, round(time.time() - t, 1), "s", r)
