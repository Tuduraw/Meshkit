"""High-level previews: standard cameras, display modes, wireframe, reference markers, contact sheets.

Display modes (all use back-face culling unless noted, like the game):
  texture      model.uvs + model.texture (falls back to 'parts' when either is missing)
  parts        each part in its own colour  - shows how the model is split into objects
  solid        uniform clay grey            - judges the form only
  tags         colour per face tag          - shows how faces were labelled for painting
  orientation  NO culling; front faces blue, back faces red - inverted faces / holes show red
  xray         NO culling, flat grey        - see through to hidden structure

Standard views (native axes +X left, +Y up, +Z forward):
  iso (front-left-above), iso_rear (rear-right-above), iso_below, front, back, left, right, top, bottom.
  front/back/left/right/top/bottom are orthographic and auto-fitted; iso* are perspective.
A custom view is a dict {eye, target, fov | ortho, up}.
"""
import math
import zlib
import numpy as np
from PIL import Image, ImageDraw

from . import render as R

AXIS_VIEWS = {
    # name: (direction from target to eye, up vector)
    "front": ((0, 0, 1), (0, 1, 0)),
    "back": ((0, 0, -1), (0, 1, 0)),
    "left": ((1, 0, 0), (0, 1, 0)),      # +X is the model's left side
    "right": ((-1, 0, 0), (0, 1, 0)),
    "top": ((0, 1, 0), (0, 0, 1)),       # nose up the image
    "bottom": ((0, -1, 0), (0, 0, 1)),
}
PERSP_VIEWS = {
    "iso": (0.62, 0.42, 0.66),
    "iso_rear": (-0.62, 0.42, -0.66),
    "iso_below": (0.55, -0.5, 0.67),
    "iso_left": (0.95, 0.30, 0.1),
    "iso_right": (-0.95, 0.30, 0.1),
}
VIEW_NAMES = list(PERSP_VIEWS) + list(AXIS_VIEWS)
BG = (196, 210, 226)


def _color_for(name, sat=0.45, val=0.85):
    h = (zlib.crc32(name.encode()) % 360) / 360.0
    i = int(h * 6); f = h * 6 - i; p = val * (1 - sat); q = val * (1 - f * sat); t = val * (1 - (1 - f) * sat)
    r, g, b = [(val, t, p), (q, val, p), (p, val, t), (p, q, val), (t, p, val), (val, p, q)][i % 6]
    return np.array([r * 255, g * 255, b * 255], np.float32)


def _points(model, transforms=None, skip=(), only=None):
    pts = []
    for pname in model.order:
        if pname in skip or (only is not None and pname not in only):
            continue
        tf = (transforms or {}).get(pname)
        for sh in model.parts[pname].shells:
            P = sh.V()
            pts.append(tf(P) if tf is not None else P)
    return np.vstack(pts) if pts else np.zeros((1, 3))


def camera(model, view="iso", W=640, H=420, fov=32.0, margin=0.9, transforms=None, skip=(), only=None, zoom=1.0,
           target=None):
    """Return a camera dict {eye, target, up, fov|ortho} framing the model for a named view (or pass a dict)."""
    if isinstance(view, dict):
        cam = {"up": (0, 1, 0), "fov": fov}; cam.update(view)
        return cam
    P = _points(model, transforms, skip, only)
    lo, hi = P.min(0), P.max(0)
    c = (lo + hi) / 2 if target is None else np.asarray(target, float)
    if view in AXIS_VIEWS:
        d, up = AXIS_VIEWS[view]
        d = np.asarray(d, float); up = np.asarray(up, float)
        rgt = np.cross(-d, up); rgt /= np.linalg.norm(rgt)
        u = np.cross(rgt, -d)
        Q = P - c
        ext_x = np.abs(Q @ rgt).max() * 2 + 1e-6; ext_y = np.abs(Q @ u).max() * 2 + 1e-6
        scale = min(W / ext_x, H / ext_y) * margin * zoom
        depth = np.abs(Q @ d).max() + 10.0
        return {"eye": tuple(c + d * depth), "target": tuple(c), "up": tuple(up), "ortho": scale}
    if view not in PERSP_VIEWS:
        raise ValueError(f"unknown view '{view}' (use one of {VIEW_NAMES} or a dict)")
    d = np.asarray(PERSP_VIEWS[view], float); d /= np.linalg.norm(d)
    r = np.linalg.norm(P - c, axis=1).max() + 1e-6
    half = math.radians(fov) / 2 * min(1.0, W / H)
    dist = r / math.sin(half) / margin
    cam = {"eye": tuple(c + d * dist), "target": tuple(c), "up": (0, 1, 0), "fov": fov}
    # tighten: move in until the projected bounds fill `margin` of the frame (a few fixed-point steps)
    for _ in range(4):
        px, ok = project(P, cam, W, H)
        if not ok.all():
            break
        ex = max(np.abs(px[:, 0] - W / 2).max() / (W / 2), np.abs(px[:, 1] - H / 2).max() / (H / 2))
        if ex <= 1e-6:
            break
        k = ex / margin
        dist = max(dist * (0.35 + 0.65 * k), r * 1.05)
        cam["eye"] = tuple(c + d * dist)
    cam["eye"] = tuple(c + d * dist / zoom)
    return cam


def project(points, cam, W, H):
    """Model-space points -> pixel coords (and a visibility flag for points in front of the camera)."""
    eye, r, u, f = R.look_at(cam["eye"], cam["target"], cam.get("up", (0, 1, 0)))
    rel = np.asarray(points, float) - eye
    xc, yc, zc = rel @ r, rel @ u, rel @ f
    if cam.get("ortho"):
        s = cam["ortho"]
        return np.stack([W / 2 + xc * s, H / 2 - yc * s], 1), np.ones(len(rel), bool)
    fl = 0.5 * H / math.tan(math.radians(cam.get("fov", 32.0)) / 2)
    ok = zc > 0.05
    zc = np.where(ok, zc, 1.0)
    return np.stack([W / 2 + fl * xc / zc, H / 2 - fl * yc / zc], 1), ok


def _tris(model, mode, transforms=None, skip=(), only=None):
    uvs = model.uvs
    use_tex = mode == "texture" and uvs is not None and model.texture is not None
    if mode == "texture" and not use_tex:
        mode = "parts"
    out = []; ids = []; pid = 0
    for pname in model.order:
        if pname in skip or (only is not None and pname not in only):
            continue
        tf = (transforms or {}).get(pname)
        pc = _color_for(pname)
        for si, sh in enumerate(model.parts[pname].shells):
            P = sh.V()
            if tf is not None:
                P = tf(P)
            for fi, face in enumerate(sh.faces):
                if use_tex:
                    uv = np.asarray(uvs.get((pname, si, fi), [(0, 0)] * len(face)), float)
                elif mode == "parts":
                    col = pc
                elif mode == "tags":
                    col = _color_for(sh.tags[fi] if fi < len(sh.tags) else "", 0.6, 0.9)
                elif mode == "orientation":
                    col = np.array([90, 130, 225], np.float32)
                else:
                    col = np.array([182, 182, 176], np.float32)
                for k in range(1, len(face) - 1):
                    idx = [0, k, k + 1]
                    Q = P[[face[i] for i in idx]]
                    out.append((Q, uv[idx]) if use_tex else (Q, None, col))
                    ids.append(pid)
                pid += 1
    return out, ids, ("texture" if use_tex else mode)


def render_view(model, view="iso", W=640, H=420, mode="texture", wire=False, refs=True, transforms=None, skip=(),
                only=None, fov=32.0, zoom=1.0, extra_solid=(), bg=BG, target=None):
    """Render one view; returns an HxWx3 uint8 array."""
    cam = camera(model, view, W, H, fov, transforms=transforms, skip=skip, only=only, zoom=zoom, target=target)
    tris, ids, used = _tris(model, mode, transforms, skip, only)
    cull = used not in ("orientation", "xray")
    img, idb = R.render(tris, model.texture if used == "texture" else None, W=W, H=H, eye=cam["eye"],
                        target=cam["target"], fov=cam.get("fov", fov), ortho=cam.get("ortho"), bg=bg, cull=cull,
                        solid=list(extra_solid), ids=ids, return_ids=True, up=cam.get("up", (0, 1, 0)),
                        backface=(225, 60, 60) if used == "orientation" else None)
    if wire:
        e = np.zeros_like(idb, bool)
        e[:, :-1] |= idb[:, :-1] != idb[:, 1:]
        e[:-1, :] |= idb[:-1, :] != idb[1:, :]
        e &= (idb >= 0) | np.roll(idb >= 0, 1, 0) | np.roll(idb >= 0, 1, 1)
        img = img.copy(); img[e] = (img[e] * 0.35).astype(np.uint8)
    if refs and model.refs:
        im = Image.fromarray(img); dr = ImageDraw.Draw(im)
        pts = np.array([p for _, p, _ in model.refs], float)
        px, ok = project(pts, cam, W, H)
        colors = {"seat": (255, 200, 0), "muzzle": (255, 60, 60), "camera": (60, 220, 255)}
        for (name, _, kind), (x, y), v in zip(model.refs, px, ok):
            if not v:
                continue
            c = colors.get(kind, (255, 255, 255))
            dr.ellipse([x - 3, y - 3, x + 3, y + 3], outline=c, width=2)
        img = np.array(im)
    return img


def sheet(model, views=("iso", "iso_rear", "left", "top", "front", "iso_below"), W=520, H=340, mode="texture", wire=False,
          cols=3, title=None, **kw):
    """Contact sheet of several views (PIL Image). Title defaults to name, triangle count and size."""
    imgs = [render_view(model, v, W, H, mode, wire, **kw) for v in views]
    labels = [v if isinstance(v, str) else v.get("label", "custom") for v in views]
    if title is None:
        s = model.summary()
        title = f"{model.name}  tris={s['tris']}  parts={s['parts']}  size(x,y,z)={s['size']}  mode={mode}"
    return R.contact_sheet(imgs, labels, cols=min(cols, len(imgs)), title=title)


def save(img, path):
    (img if isinstance(img, Image.Image) else Image.fromarray(img)).save(path)
    return path
