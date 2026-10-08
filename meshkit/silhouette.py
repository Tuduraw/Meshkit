"""Outline of a model seen orthographically (side / top / front), in model units, for overlaying
on reference drawings (needs opencv-python).

Projection (native axes): side = (Z, Y) - nose to the right; top = (-X, Z) - nose up, the model's
left (+X) on the left; front = (-X, Y)."""
import numpy as np


def outline(model, view="side", res=0.01, transforms=None, skip=(), only=None):
    try:
        import cv2
    except ImportError as e:
        raise SystemExit("silhouette needs opencv: pip install opencv-python-headless") from e
    pts2 = []
    tris = []
    for pname in model.order:
        if pname in skip or (only is not None and pname not in only):
            continue
        tf = (transforms or {}).get(pname)
        for sh in model.parts[pname].shells:
            P = sh.V()
            if tf is not None:
                P = tf(P)
            if view == "side":
                Q = P[:, [2, 1]]
            elif view == "top":
                Q = np.stack([-P[:, 0], P[:, 2]], 1)     # +X (left) drawn on the left when nose points up
            else:
                Q = P[:, [0, 1]] * [-1, 1]
            for f in sh.faces:
                for k in range(1, len(f) - 1):
                    tris.append(Q[[f[0], f[k], f[k + 1]]])
    T = np.array(tris)
    lo = T.reshape(-1, 2).min(0) - 0.05; hi = T.reshape(-1, 2).max(0) + 0.05
    W = int((hi[0] - lo[0]) / res) + 1; H = int((hi[1] - lo[1]) / res) + 1
    img = np.zeros((H, W), np.uint8)
    px = np.round((T - lo) / res * 16).astype(np.int32)          # 4-bit subpixel
    px[..., 1] = (H - 1) * 16 - px[..., 1]
    for t in px:
        cv2.fillConvexPoly(img, t, 255, lineType=cv2.LINE_8, shift=4)
    cs, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cs:
        if len(c) < 3 or cv2.contourArea(c) < 4:
            continue
        c = cv2.approxPolyDP(c, 1.0, True)[:, 0, :].astype(float)
        a = lo[0] + c[:, 0] * res
        b = lo[1] + ((H - 1) - c[:, 1]) * res
        out.append(np.stack([a, b], 1).round(3).tolist())
    return out, lo.round(3).tolist(), hi.round(3).tolist()




def draw(paths, lo, hi, ref=None, ref_box=None, px_per_m=None, color=(255, 40, 40)):
    """Image of the outline. ref: drawing to put underneath; ref_box: (a0, b0, a1, b1) = the model-unit
    rectangle that the reference image covers (calibrate it from known dimensions). Returns a PIL image."""
    from PIL import Image, ImageDraw
    lo = np.array(lo, float); hi = np.array(hi, float)
    if ref_box is not None:
        lo = np.minimum(lo, ref_box[:2]); hi = np.maximum(hi, ref_box[2:])
    span = hi - lo
    s = px_per_m or 1200.0 / max(span)
    W, H = int(span[0] * s) + 20, int(span[1] * s) + 20
    img = Image.new("RGB", (W, H), (255, 255, 255))
    tx = lambda a, b: (10 + (a - lo[0]) * s, H - 10 - (b - lo[1]) * s)
    if ref is not None and ref_box is not None:
        r = Image.open(ref).convert("RGB")
        x0, y1 = tx(ref_box[0], ref_box[1]); x1, y0 = tx(ref_box[2], ref_box[3])
        r = r.resize((max(1, int(x1 - x0)), max(1, int(y1 - y0))))
        img.paste(Image.blend(r, Image.new("RGB", r.size, (255, 255, 255)), 0.35), (int(x0), int(y0)))
    d = ImageDraw.Draw(img)
    for p in paths:
        pts = [tx(a, b) for a, b in p]
        d.line(pts + [pts[0]], fill=color, width=2)
    return img
