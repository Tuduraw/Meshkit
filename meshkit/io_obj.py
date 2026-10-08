"""Wavefront OBJ reader / writer.

write_obj() produces the same layout as Blender 4.x's exporter with the settings ww2gen used
(forward -Z, up Y, UVs, normals, flat shading, no materials, untriangulated):

    o <part>          one object per part, in model order
    v x y z           part vertices (6 decimals), in shell / vertex order
    vn x y z          flat face normals (4 decimals), de-duplicated per object
    vt u v            UVs (6 decimals), de-duplicated per object
    s 0
    f v/vt/vn ...

so a model written here can replace a Blender export line for line (see tests/compare_blender.py).

read_obj() turns any OBJ (Blender, other tools, ours) into a Model: one part per `o`/`g` group,
vertices welded by position, one Shell per connected set of faces, per-corner UVs in model.uvs.
"""
import os
import numpy as np

from .geom import Model, Part, Shell

# axis presets: matrix taking native coords (+X left, +Y up, +Z forward) to the file's coords
AXES = {
    "native": np.eye(3),                                         # Y-up, +Z forward (Minecraft / glTF / ww2gen)
    "blender": np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float),  # Blender world: Z-up, nose towards -Y
}


def _newell(P):
    n = np.zeros(3)
    m = len(P)
    for i in range(m):
        a = P[i]; b = P[(i + 1) % m]
        n += np.array([(a[1] - b[1]) * (a[2] + b[2]), (a[2] - b[2]) * (a[0] + b[0]), (a[0] - b[0]) * (a[1] + b[1])])
    ln = np.linalg.norm(n)
    return n / ln if ln > 1e-20 else n


def _f(x, nd):
    s = f"{x:.{nd}f}"
    return s


def write_obj(model, path, uvs=None, normals=True, axes="native", header=None, parts=None, mtl=None):
    """Write `model` to `path`. uvs defaults to model.uvs. axes: 'native' or 'blender' or a 3x3 matrix.
    parts: subset/order of part names (default model.order). mtl: (mtl_filename, material_name) to emit
    mtllib/usemtl lines (off by default, like ww2gen's export)."""
    uvs = model.uvs if uvs is None else uvs
    M = AXES[axes] if isinstance(axes, str) else np.asarray(axes, float)
    flip = np.linalg.det(M) < 0
    header = "# meshkit OBJ" if header is None else header
    lines = header.splitlines() if header else []
    if mtl:
        lines.append(f"mtllib {mtl[0]}")
    vbase = tbase = nbase = 0
    for pname in (parts or model.order):
        part = model.parts[pname]
        if not any(sh.faces for sh in part.shells):
            continue
        lines.append(f"o {pname}")
        V = []; F = []; FUV = []
        for si, sh in enumerate(part.shells):
            off = len(V)
            V.extend(sh.verts)
            for fi, f in enumerate(sh.faces):
                F.append([off + i for i in f])
                FUV.append(uvs[(pname, si, fi)] if uvs is not None else None)
        P = np.asarray(V, float) @ M.T
        if flip:
            F = [list(reversed(f)) for f in F]
            FUV = [None if u is None else list(reversed(u)) for u in FUV]
        for p in P:
            lines.append(f"v {_f(p[0], 6)} {_f(p[1], 6)} {_f(p[2], 6)}")
        nidx = []; nmap = {}
        if normals:
            nl = []
            for f in F:
                n = _newell(P[f])
                # de-duplicate by the rounded value (so -0.0000 and 0.0000 are one normal, as in Blender)
                key = tuple(int(round(c * 10000)) for c in n)
                if key not in nmap:
                    nmap[key] = len(nmap); nl.append(f"vn {_f(n[0], 4)} {_f(n[1], 4)} {_f(n[2], 4)}")
                nidx.append(nmap[key])
            lines.extend(nl)
        tidx = []; tmap = {}
        if uvs is not None:
            tl = []
            for fuv in FUV:
                row = []
                for u, v in fuv:
                    key = (int(round(float(u) * 1e6)), int(round(float(v) * 1e6)))
                    if key not in tmap:
                        tmap[key] = len(tmap); tl.append(f"vt {_f(float(u), 6)} {_f(float(v), 6)}")
                    row.append(tmap[key])
                tidx.append(row)
            lines.extend(tl)
        if mtl:
            lines.append(f"usemtl {mtl[1]}")
        lines.append("s 0")
        for k, f in enumerate(F):
            toks = []
            for j, vi in enumerate(f):
                a = str(vbase + vi + 1)
                b = str(tbase + tidx[k][j] + 1) if uvs is not None else ""
                c = str(nbase + nidx[k] + 1) if normals else ""
                toks.append(a if not (b or c) else (f"{a}/{b}" + (f"/{c}" if c else "")))
            lines.append("f " + " ".join(toks))
        vbase += len(P); tbase += len(tmap); nbase += len(nmap)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fp:
        fp.write("\n".join(lines) + "\n")
    return path


def write_mtl(path, material, texture_file):
    with open(path, "w", encoding="utf-8", newline="\n") as fp:
        fp.write(f"newmtl {material}\nKd 1 1 1\nmap_Kd {texture_file}\n")


def parse_obj(path):
    """Raw parse: positions, uvs, groups {name: [[(vi, ti), ...], ...]} (0-based, ti may be None), group order."""
    pos = []; tex = []; groups = {}; order = []; cur = None
    with open(path, encoding="utf-8", errors="replace") as fp:
        for line in fp:
            line = line.strip()
            if not line or line[0] == "#":
                continue
            p = line.split()
            k = p[0]
            if k == "v":
                pos.append((float(p[1]), float(p[2]), float(p[3])))
            elif k == "vt":
                tex.append((float(p[1]), float(p[2]) if len(p) > 2 else 0.0))
            elif k in ("o", "g"):
                cur = " ".join(p[1:]) or "default"
            elif k == "f":
                if cur is None:
                    cur = "default"
                if cur not in groups:
                    groups[cur] = []; order.append(cur)
                face = []
                for tok in p[1:]:
                    s = tok.split("/")
                    vi = int(s[0]); vi = vi - 1 if vi > 0 else len(pos) + vi
                    ti = None
                    if len(s) > 1 and s[1]:
                        ti = int(s[1]); ti = ti - 1 if ti > 0 else len(tex) + ti
                    face.append((vi, ti))
                groups[cur].append(face)
    return np.array(pos, float).reshape(-1, 3), np.array(tex, float).reshape(-1, 2), groups, order


def mesh_to_part(name, P, faces, face_uvs=None, weld=1e-5, tag="body"):
    """Build a Part from global positions P and faces (lists of vertex indices).
    Vertices are welded by position (tolerance `weld`), faces are grouped into shells by shared
    vertices. Returns (part, uv_dict {(shell, face): [...]})."""
    key = {}; remap = {}
    for f in faces:
        for vi in f:
            if vi in remap:
                continue
            k = tuple(np.round(np.asarray(P[vi]) / weld).astype(np.int64)) if weld else vi
            if k not in key:
                key[k] = vi
            remap[vi] = key[k]
    F = []
    for f in faces:
        g = []
        for vi in f:
            r = remap[vi]
            if not g or g[-1] != r:
                g.append(r)
        if len(g) > 1 and g[0] == g[-1]:
            g.pop()
        F.append(g)
    # union-find over vertices
    parent = {}

    def find(a):
        parent.setdefault(a, a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    for f in F:
        for v in f[1:]:
            ra, rb = find(f[0]), find(v)
            if ra != rb:
                parent[ra] = rb
    comps = {}; order = []
    for fi, f in enumerate(F):
        if len(f) < 3:
            continue
        r = find(f[0])
        if r not in comps:
            comps[r] = []; order.append(r)
        comps[r].append(fi)
    part = Part(name); uvd = {}
    for si, r in enumerate(order):
        sh = Shell(name=f"{name}.{si}")
        local = {}
        for fi in comps[r]:
            ids = []
            for v in F[fi]:
                if v not in local:
                    local[v] = sh.add_vert(P[v])
                ids.append(local[v])
            sh.faces.append(tuple(ids)); sh.tags.append(tag)
            if face_uvs is not None and face_uvs[fi] is not None:
                uvd[(si, len(sh.faces) - 1)] = face_uvs[fi]
        part.shells.append(sh)
    return part, uvd


def read_obj(path, axes="native", weld=1e-5, texture=None):
    """Load an OBJ as a Model. axes: the convention the file is in ('native' or 'blender' or a matrix
    native->file); it is converted to native. texture: optional image path (else the mtl's map_Kd if found)."""
    P, T, groups, order = parse_obj(path)
    M = AXES[axes] if isinstance(axes, str) else np.asarray(axes, float)
    Minv = np.linalg.inv(M)
    flip = np.linalg.det(M) < 0
    P = P @ Minv.T
    model = Model(os.path.splitext(os.path.basename(path))[0])
    has_uv = len(T) > 0
    uvs = {} if has_uv else None
    for g in order:
        faces = [[v for v, _ in f] for f in groups[g]]
        fuv = [[tuple(T[t]) if t is not None else (0.0, 0.0) for _, t in f] for f in groups[g]] if has_uv else None
        if flip:
            faces = [list(reversed(f)) for f in faces]
            fuv = [list(reversed(u)) for u in fuv] if fuv else None
        part, uvd = mesh_to_part(g, P, faces, fuv, weld)
        name = g; k = 1
        while name in model.parts:
            k += 1; name = f"{g}.{k}"
        part.name = name
        model.parts[name] = part; model.order.append(name)
        if has_uv:
            for (si, fi), u in uvd.items():
                uvs[(name, si, fi)] = u
    model.uvs = uvs
    tex_path = texture or _find_mtl_texture(path)
    if tex_path and os.path.exists(tex_path):
        from PIL import Image
        model.texture = np.array(Image.open(tex_path).convert("RGBA"))
    return model


def _find_mtl_texture(obj_path):
    d = os.path.dirname(os.path.abspath(obj_path))
    try:
        for line in open(obj_path, encoding="utf-8", errors="replace"):
            if line.startswith("mtllib"):
                mtl = os.path.join(d, line.split(None, 1)[1].strip())
                if os.path.exists(mtl):
                    for l2 in open(mtl, encoding="utf-8", errors="replace"):
                        if l2.strip().startswith("map_Kd"):
                            return os.path.join(d, l2.strip().split(None, 1)[1].strip())
                break
    except OSError:
        pass
    return None
