"""Software preview renderer that mimics the mod's own model rendering:
back-face culling (CCW = front), flat per-face lighting with Minecraft's two
entity light directions, nearest-neighbour texture sampling, z-buffer.
Any hole in a mesh shows up here as see-through background, exactly as in game."""
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont

BAYER = np.array([0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5], np.float32)
L0 = np.array([0.2, 1.0, -0.7]); L0 /= np.linalg.norm(L0)
L1 = np.array([-0.2, 1.0, 0.7]); L1 /= np.linalg.norm(L1)


def look_at(eye, target, up=(0, 1, 0)):
    eye = np.asarray(eye, float); target = np.asarray(target, float)
    f = target - eye; f /= np.linalg.norm(f)
    r = np.cross(f, up); r /= np.linalg.norm(r)
    u = np.cross(r, f)
    return eye, r, u, f


def render(tris, tex, W=640, H=420, eye=(10, 6, 10), target=(0, 1, 0), fov=35.0, ortho=None, bg=(200, 214, 228), cull=True,
           solid=(), ids=None, backface=None, return_ids=False, up=(0, 1, 0)):
    """tris: list of (P(3x3), UV(3x2)) or (P, None, rgb) in model space. tex: HxWx3/4 uint8 (may be None
    when every triangle carries a flat colour).
    solid: extra flat-coloured triangles [(P(3x3), rgb)] (e.g. a stand-in for the seated player).
    ortho: pixels per model unit for an orthographic view (None = perspective with `fov`).
    ids: optional per-triangle integer (e.g. the polygon id) written to an id buffer; return_ids=True
    returns (image, idbuf) - used for wireframe overlays of polygon outlines.
    backface: rgb - with cull=False, back faces are painted this colour instead (shows inverted
    faces and holes, like Blender's face-orientation overlay)."""
    tris = list(tris) + [(np.asarray(P, float), None, np.asarray(c, np.float32)) for P, c in solid]
    if ids is not None:
        ids = list(ids) + [-2] * len(solid)
    idbuf = np.full((H, W), -1, np.int64) if return_ids else None
    if tex is None:
        tex = np.full((1, 1, 4), 200, np.uint8)
    eye, r, u, f = look_at(eye, target, up)
    img = np.zeros((H, W, 3), np.float32)
    # background gradient
    gy = np.linspace(0, 1, H)[:, None]
    img[:] = (np.array(bg) * (1 - 0.25 * gy))[:, None, :] if False else np.array(bg, np.float32)
    zbuf = np.full((H, W), np.inf, np.float32)
    th, tw = tex.shape[:2]
    has_alpha = tex.ndim == 3 and tex.shape[2] == 4 and bool(np.any(tex[..., 3] < 255))
    # the mod upscales translucent textures x4 (x2/x1 when that would exceed 2048 px) before baking
    up = 4
    while up > 1 and max(th, tw) * up > 2048:
        up //= 2
    fl = 0.5 * H / math.tan(math.radians(fov) / 2)
    for ti, item in enumerate(tris):
        if len(item) == 3:
            P, UV, flat = item
        else:
            (P, UV), flat = item, None
        rel = P - eye
        xc = rel @ r; yc = rel @ u; zc = rel @ f
        if ortho is None:
            if np.any(zc < 0.05):
                continue
            sx = W / 2 + fl * xc / zc; sy = H / 2 - fl * yc / zc
            invw = 1.0 / zc
        else:
            sx = W / 2 + xc * ortho; sy = H / 2 - yc * ortho
            invw = np.ones(3)
        # screen-space orientation (y up): front faces are CCW
        area = (sx[1] - sx[0]) * (-(sy[2] - sy[0])) - (-(sy[1] - sy[0])) * (sx[2] - sx[0])
        back = area <= 0
        if cull and back:
            continue
        if back and backface is not None:
            flat = np.asarray(backface, np.float32); UV = None
        if abs(area) < 1e-9:
            continue
        x0 = max(int(math.floor(sx.min())), 0); x1 = min(int(math.ceil(sx.max())), W - 1)
        y0 = max(int(math.floor(sy.min())), 0); y1 = min(int(math.ceil(sy.max())), H - 1)
        if x0 > x1 or y0 > y1:
            continue
        X, Y = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
        # barycentrics
        d = (sy[1] - sy[2]) * (sx[0] - sx[2]) + (sx[2] - sx[1]) * (sy[0] - sy[2])
        if abs(d) < 1e-12:
            continue
        w0 = ((sy[1] - sy[2]) * (X - sx[2]) + (sx[2] - sx[1]) * (Y - sy[2])) / d
        w1 = ((sy[2] - sy[0]) * (X - sx[2]) + (sx[0] - sx[2]) * (Y - sy[2])) / d
        w2 = 1 - w0 - w1
        inside = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
        if not inside.any():
            continue
        if ortho is None:
            # perspective-correct depth (1/z is linear in screen space) - large faces spanning a big depth
            # range (flight decks, long hull plates) would otherwise lose the depth test to what lies behind them
            zint = 1.0 / np.maximum(w0 * invw[0] + w1 * invw[1] + w2 * invw[2], 1e-9)
        else:
            zint = w0 * zc[0] + w1 * zc[1] + w2 * zc[2]
        sub = zbuf[y0:y1 + 1, x0:x1 + 1]
        vis = inside & (zint < sub)
        if not vis.any():
            continue
        if flat is not None:
            n = np.cross(P[1] - P[0], P[2] - P[0]); n /= (np.linalg.norm(n) + 1e-12)
            light = 1.0 if back else min(1.0, 0.4 + 0.6 * (max(0.0, n @ L0) + max(0.0, n @ L1)))
            sub[vis] = zint[vis]
            img[y0:y1 + 1, x0:x1 + 1][vis] = flat * light
            if idbuf is not None:
                idbuf[y0:y1 + 1, x0:x1 + 1][vis] = ids[ti] if ids is not None else ti
            continue
        # perspective-correct uv
        q = w0 * invw[0] + w1 * invw[1] + w2 * invw[2]
        uu = (w0 * UV[0, 0] * invw[0] + w1 * UV[1, 0] * invw[1] + w2 * UV[2, 0] * invw[2]) / q
        vv = (w0 * UV[0, 1] * invw[0] + w1 * UV[1, 1] * invw[1] + w2 * UV[2, 1] * invw[2]) / q
        tx = np.clip((uu * tw).astype(int), 0, tw - 1)
        ty = np.clip(((1 - vv) * th).astype(int), 0, th - 1)
        col = tex[ty, tx, :3].astype(np.float32)
        if has_alpha:
            a = tex[ty, tx, 3].astype(np.float32) / 255.0
            ux = np.floor(uu * tw * up).astype(np.int64) & 3
            uy = np.floor((1 - vv) * th * up).astype(np.int64) & 3
            thr = (BAYER[uy * 4 + ux] + 0.5) / 16.0
            keep = (a >= 1.0) | ((a > 0.0) & (a >= thr))
            vis = vis & keep
            if not vis.any():
                continue
        n = np.cross(P[1] - P[0], P[2] - P[0]); n /= (np.linalg.norm(n) + 1e-12)
        light = min(1.0, 0.4 + 0.6 * (max(0.0, n @ L0) + max(0.0, n @ L1)))
        sub[vis] = zint[vis]
        region = img[y0:y1 + 1, x0:x1 + 1]
        region[vis] = col[vis] * light
        if idbuf is not None:
            idbuf[y0:y1 + 1, x0:x1 + 1][vis] = ids[ti] if ids is not None else ti
    out = np.clip(img, 0, 255).astype(np.uint8)
    return (out, idbuf) if return_ids else out


def model_tris(model, uvs, transforms=None, skip=(), only=None):
    """Fan-triangulate every face. transforms: part -> function(points)->points."""
    out = []
    for pname in model.order:
        if pname in skip or (only is not None and pname not in only):
            continue
        tf = (transforms or {}).get(pname)
        for si, sh in enumerate(model.parts[pname].shells):
            P = sh.V()
            if tf is not None:
                P = tf(P)
            for fi, face in enumerate(sh.faces):
                uv = np.array(uvs[(pname, si, fi)])
                for k in range(1, len(face) - 1):
                    idx = [0, k, k + 1]
                    out.append((P[[face[i] for i in idx]], uv[idx]))
    return out


def contact_sheet(images, labels, cols=3, pad=6, title=None):
    h, w = images[0].shape[:2]
    rows = math.ceil(len(images) / cols)
    top = 26 if title else 0
    sheet = Image.new("RGB", (cols * (w + pad) + pad, rows * (h + pad + 16) + pad + top), (40, 40, 40))
    dr = ImageDraw.Draw(sheet)
    if title:
        dr.text((pad, 6), title, fill=(255, 255, 255))
    for i, (im, lab) in enumerate(zip(images, labels)):
        cx = pad + (i % cols) * (w + pad); cy = top + pad + (i // cols) * (h + pad + 16)
        sheet.paste(Image.fromarray(im), (cx, cy + 16))
        dr.text((cx + 2, cy + 2), lab, fill=(230, 230, 230))
    return sheet


def _box_tris(center, size, M=None, pivot=None, color=(200, 160, 130)):
    cx, cy, cz = center; hx, hy, hz = [v / 2 for v in size]
    V = np.array([[cx + sx * hx, cy + sy * hy, cz + sz * hz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
    if M is not None:
        pv = np.asarray(pivot if pivot is not None else center, float)
        V = (V - pv) @ M.T + pv
    # vertex index = (sx>0)*4 + (sy>0)*2 + (sz>0)
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    out = []
    for q in quads:
        P = V[list(q)]
        c = P.mean(0)
        n = np.cross(P[1] - P[0], P[2] - P[0])
        if n @ (c - V.mean(0)) < 0:
            q = q[::-1]; P = V[list(q)]
        out.append((P[[0, 1, 2]], color)); out.append((P[[0, 2, 3]], color))
    return out


def rider(seat, scale=1.0, yaw=0.0, skin=(214, 168, 138), suit=(92, 84, 60)):
    """Stand-in for a seated Minecraft player (riding pose), feet at the seat offset, facing +Z."""
    from .geom import rot_x, rot_y, rot_z
    s = scale; px = 0.9375 / 16 * s
    fx, fy, fz = seat
    R = rot_y(yaw)
    parts = []
    def add(center, size, M=None, pivot=None, color=suit):
        c = np.asarray(center, float); pv = None if pivot is None else np.asarray(pivot, float)
        parts.extend(_box_tris(c, [v * px for v in size], M, pv, color))
    hip = fy + 12 * px; sh = fy + 24 * px
    add((fx, sh + 4 * px, fz), (8, 8, 8), color=skin)                      # head
    add((fx, (hip + sh) / 2, fz), (8, 12, 4))                             # body
    for sx in (1, -1):
        a_piv = (fx + sx * 5 * px, sh - 2 * px, fz)
        add((a_piv[0], a_piv[1] - 4 * px, fz), (4, 12, 4), rot_x(-36), a_piv)           # arms swung forward
        l_piv = (fx + sx * 2 * px, hip, fz)
        add((l_piv[0], hip - 6 * px, fz), (4, 12, 4), rot_x(-81) @ rot_z(sx * 6), l_piv)  # legs forward
    if yaw:
        out = []
        for P, c in parts:
            out.append(((P - [fx, fy, fz]) @ R.T + [fx, fy, fz], c))
        parts = out
    return parts
