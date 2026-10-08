"""glTF 2.0 (.glb / .gltf) reader and writer, numpy only.

glTF's axis convention (+Y up, +Z forward, +X left, metres) is meshkit's native one, so no axis
conversion happens here.

write_glb(): one node per part (translation = the part's pivot, vertices relative to it), flat
normals (corners split per face), the texture embedded as PNG with nearest-neighbour sampling,
back-face culling on (doubleSided false), alphaMode MASK when the texture has transparency.
model.props are written as node extras, model.refs as empty nodes (extras.kind).

read_gltf(): every mesh node becomes a part (world transform applied), vertices welded by
position so closure checks work, per-corner UVs and the base-colour texture are kept.
"""
import base64
import io
import json
import os
import struct
import numpy as np

from .io_obj import mesh_to_part, _newell
from .geom import Model

_CT = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
_NC = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


# ------------------------------------------------------------------------------------------ write
def write_glb(model, path, uvs=None, texture=None, pivots=None, name=None):
    """texture: HxWx3/4 uint8 array or a PNG path (default model.texture)."""
    from PIL import Image
    uvs = model.uvs if uvs is None else uvs
    pivots = model.pivots if pivots is None else pivots
    texture = model.texture if texture is None else texture
    blob = bytearray(); views = []; accessors = []

    def add_view(data, target=None):
        while len(blob) % 4:
            blob.append(0)
        off = len(blob); blob.extend(data)
        v = {"buffer": 0, "byteOffset": off, "byteLength": len(data)}
        if target:
            v["target"] = target
        views.append(v)
        return len(views) - 1

    def add_acc(arr, typ, ctype=5126, target=34962, minmax=False):
        arr = np.ascontiguousarray(arr)
        vi = add_view(arr.tobytes(), target)
        a = {"bufferView": vi, "componentType": ctype, "count": int(arr.shape[0]), "type": typ}
        if minmax:
            a["min"] = arr.min(0).astype(float).tolist(); a["max"] = arr.max(0).astype(float).tolist()
        accessors.append(a)
        return len(accessors) - 1

    gl = {"asset": {"version": "2.0", "generator": "meshkit"}, "scene": 0, "scenes": [{"nodes": []}],
          "nodes": [], "meshes": []}
    mat_index = None
    if texture is not None or uvs is not None:
        mat = {"name": (name or model.name) + "_mat", "doubleSided": False,
               "pbrMetallicRoughness": {"metallicFactor": 0.0, "roughnessFactor": 0.8}}
        if texture is not None:
            img = Image.open(texture) if isinstance(texture, str) else Image.fromarray(np.asarray(texture))
            buf = io.BytesIO(); img.save(buf, "PNG")
            ivi = add_view(buf.getvalue())
            gl["images"] = [{"bufferView": ivi, "mimeType": "image/png"}]
            gl["samplers"] = [{"magFilter": 9728, "minFilter": 9728}]
            gl["textures"] = [{"sampler": 0, "source": 0}]
            mat["pbrMetallicRoughness"]["baseColorTexture"] = {"index": 0}
            a = np.asarray(img.convert("RGBA"))[..., 3]
            if (a < 255).any():
                mat["alphaMode"] = "MASK"; mat["alphaCutoff"] = 0.5
        gl["materials"] = [mat]; mat_index = 0
    root = gl["scene"] and 0
    for pname in model.order:
        part = model.parts[pname]
        piv = np.asarray(pivots.get(pname, (0, 0, 0)), float)
        pos = []; nrm = []; tex = []
        for si, sh in enumerate(part.shells):
            P = sh.V()
            for fi, f in enumerate(sh.faces):
                Q = P[list(f)]; n = _newell(Q)
                uv = uvs[(pname, si, fi)] if uvs is not None else None
                for k in range(1, len(f) - 1):
                    for j in (0, k, k + 1):
                        pos.append(Q[j] - piv); nrm.append(n)
                        if uv is not None:
                            tex.append((uv[j][0], 1.0 - uv[j][1]))      # glTF v runs downwards
        if not pos:
            continue
        attrs = {"POSITION": add_acc(np.array(pos, np.float32), "VEC3", minmax=True),
                 "NORMAL": add_acc(np.array(nrm, np.float32), "VEC3")}
        if tex:
            attrs["TEXCOORD_0"] = add_acc(np.array(tex, np.float32), "VEC2")
        prim = {"attributes": attrs, "mode": 4}
        if mat_index is not None:
            prim["material"] = mat_index
        gl["meshes"].append({"name": pname, "primitives": [prim]})
        node = {"name": pname, "mesh": len(gl["meshes"]) - 1}
        if np.any(piv):
            node["translation"] = piv.tolist()
        if model.props.get(pname):
            node["extras"] = model.props[pname]
        gl["nodes"].append(node); gl["scene"] = 0
        gl["scenes"][0]["nodes"].append(len(gl["nodes"]) - 1)
    for rname, p, kind in model.refs:
        gl["nodes"].append({"name": rname, "translation": [float(c) for c in p], "extras": {"kind": kind}})
        gl["scenes"][0]["nodes"].append(len(gl["nodes"]) - 1)
    del root
    while len(blob) % 4:
        blob.append(0)
    gl["buffers"] = [{"byteLength": len(blob)}]
    gl["bufferViews"] = views; gl["accessors"] = accessors
    js = json.dumps(gl, separators=(",", ":")).encode()
    js += b" " * ((4 - len(js) % 4) % 4)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as fp:
        fp.write(struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(blob)))
        fp.write(struct.pack("<II", len(js), 0x4E4F534A)); fp.write(js)
        fp.write(struct.pack("<II", len(blob), 0x004E4942)); fp.write(bytes(blob))
    return path


# ------------------------------------------------------------------------------------------- read
def _load(path):
    data = open(path, "rb").read()
    if data[:4] == b"glTF":
        _, _, total = struct.unpack_from("<III", data, 0)
        off = 12; gl = None; binchunk = None
        while off < total:
            ln, typ = struct.unpack_from("<II", data, off); off += 8
            chunk = data[off:off + ln]; off += ln
            if typ == 0x4E4F534A:
                gl = json.loads(chunk.decode("utf-8"))
            elif typ == 0x004E4942:
                binchunk = chunk
        buffers = []
        for b in gl.get("buffers", []):
            buffers.append(binchunk if "uri" not in b else _uri(b["uri"], path))
    else:
        gl = json.loads(data.decode("utf-8"))
        buffers = [_uri(b["uri"], path) for b in gl.get("buffers", [])]
    return gl, buffers


def _uri(uri, path):
    if uri.startswith("data:"):
        return base64.b64decode(uri.split(",", 1)[1])
    from urllib.parse import unquote
    return open(os.path.join(os.path.dirname(os.path.abspath(path)), unquote(uri)), "rb").read()


def _view_bytes(gl, buffers, vi):
    v = gl["bufferViews"][vi]
    b = buffers[v["buffer"]]; o = v.get("byteOffset", 0)
    return b[o:o + v["byteLength"]], v.get("byteStride")


def _accessor(gl, buffers, ai):
    a = gl["accessors"][ai]
    dt = np.dtype(_CT[a["componentType"]]); nc = _NC[a["type"]]; cnt = a["count"]
    if "bufferView" in a:
        raw, stride = _view_bytes(gl, buffers, a["bufferView"])
        off = a.get("byteOffset", 0); isz = dt.itemsize * nc
        if stride and stride != isz:
            out = np.empty((cnt, nc), dt)
            for i in range(cnt):
                out[i] = np.frombuffer(raw, dt, nc, off + i * stride)
        else:
            out = np.frombuffer(raw, dt, cnt * nc, off).reshape(cnt, nc).copy()
    else:
        out = np.zeros((cnt, nc), dt)
    if "sparse" in a:
        sp = a["sparse"]
        idt = np.dtype(_CT[sp["indices"]["componentType"]])
        rawi, _ = _view_bytes(gl, buffers, sp["indices"]["bufferView"])
        idx = np.frombuffer(rawi, idt, sp["count"], sp["indices"].get("byteOffset", 0))
        rawv, _ = _view_bytes(gl, buffers, sp["values"]["bufferView"])
        vals = np.frombuffer(rawv, dt, sp["count"] * nc, sp["values"].get("byteOffset", 0)).reshape(-1, nc)
        out[idx] = vals
    out = out.astype(np.float64)
    if a.get("normalized"):
        out = out / float(np.iinfo(dt).max)
    return out


def _quat(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _local(n):
    if "matrix" in n:
        return np.array(n["matrix"], float).reshape(4, 4).T
    M = np.eye(4)
    R = _quat(n.get("rotation", [0, 0, 0, 1])) @ np.diag(n.get("scale", [1, 1, 1]))
    M[:3, :3] = R; M[:3, 3] = n.get("translation", [0, 0, 0])
    return M


def read_gltf(path, weld=1e-5):
    from PIL import Image
    gl, buffers = _load(path)
    model = Model(os.path.splitext(os.path.basename(path))[0])
    uvs = {}
    any_uv = False
    scene = gl.get("scenes", [{"nodes": list(range(len(gl.get("nodes", []))))}])[gl.get("scene", 0)]

    def visit(ni, parent):
        n = gl["nodes"][ni]
        W = parent @ _local(n)
        nonlocal any_uv
        if "mesh" in n:
            mesh = gl["meshes"][n["mesh"]]
            P_all = []; faces = []; fuv = []
            for prim in mesh["primitives"]:
                if prim.get("mode", 4) != 4:
                    continue
                P = _accessor(gl, buffers, prim["attributes"]["POSITION"])
                P = P @ W[:3, :3].T + W[:3, 3]
                T = _accessor(gl, buffers, prim["attributes"]["TEXCOORD_0"]) if "TEXCOORD_0" in prim["attributes"] else None
                idx = _accessor(gl, buffers, prim["indices"]).astype(int).ravel() if "indices" in prim else np.arange(len(P))
                base = len(P_all); P_all.extend(P)
                flip = np.linalg.det(W[:3, :3]) < 0
                for t in idx.reshape(-1, 3):
                    tri = [int(t[0]), int(t[1]), int(t[2])]
                    if flip:
                        tri = tri[::-1]
                    faces.append([base + i for i in tri])
                    fuv.append([(float(T[i][0]), 1.0 - float(T[i][1])) for i in tri] if T is not None else None)
                    any_uv = any_uv or T is not None
            if faces:
                nm = n.get("name") or mesh.get("name") or f"node{ni}"
                name = nm; k = 1
                while name in model.parts:
                    k += 1; name = f"{nm}.{k}"
                part, uvd = mesh_to_part(name, np.array(P_all), faces, fuv, weld)
                model.parts[name] = part; model.order.append(name)
                model.pivots[name] = tuple(W[:3, 3])
                if n.get("extras"):
                    model.props[name] = n["extras"]
                for (si, fi), u in uvd.items():
                    uvs[(name, si, fi)] = u
        elif not n.get("children"):
            model.refs.append((n.get("name", f"node{ni}"), tuple(W[:3, 3]), (n.get("extras") or {}).get("kind", "empty")))
        for c in n.get("children", []):
            visit(c, W)
    for ni in scene["nodes"]:
        visit(ni, np.eye(4))
    if any_uv:
        # faces without UVs get (0,0) corners so that every face has an entry
        for pname in model.order:
            for si, sh in enumerate(model.parts[pname].shells):
                for fi, f in enumerate(sh.faces):
                    uvs.setdefault((pname, si, fi), [(0.0, 0.0)] * len(f))
        model.uvs = uvs
    # base colour texture of the first textured material
    for m in gl.get("materials", []):
        t = m.get("pbrMetallicRoughness", {}).get("baseColorTexture")
        if t is None:
            continue
        src = gl["textures"][t["index"]].get("source")
        if src is None:
            continue
        im = gl["images"][src]
        raw = _view_bytes(gl, buffers, im["bufferView"])[0] if "bufferView" in im else _uri(im["uri"], path)
        model.texture = np.array(Image.open(io.BytesIO(raw)).convert("RGBA"))
        break
    return model
