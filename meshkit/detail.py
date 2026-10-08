"""Texture detail passes - "more paint, fewer polygons".  Call them inside a `texture_model(paint=...)`
callback (they work on the Atlas after rasterisation).  Typical order:

    base colours / camouflage -> tone -> panel_lines / rivets / access_panels (aircraft),
    deck_tiles / plating (ships), hubs / grilles (tanks) -> markings -> ink -> grime / streaks / chips

ink()   is the model-kit "panel-line wash": it darkens texels where the projected surface has an edge -
        a change of face tag, a depth step (one surface overlapping another) or a crease (change of
        slope) - plus a soft halo.  It turns flat low-poly blocks into readable shapes at no polygon cost.
"""
import numpy as np
from .texture import value_noise

KINDS = ("top", "bottom", "side", "front")


# ---------------------------------------------------------------------------------- wash
def ink(atlas, tags=None, depth_m=0.035, crease_m=0.012, strength=0.42, halo=0.16, kinds=KINDS, skip_tags=()):
    ids = None if tags is None else set(atlas.tag_ids[t] for t in tags if t in atlas.tag_ids)
    skip = set(atlas.tag_ids[t] for t in skip_tags if t in atlas.tag_ids)
    for r in atlas.regions.values():
        if r.base not in kinds or r.w < 3 or r.h < 3:
            continue
        y0, x0, h, w = r.y0, r.x0, r.h, r.w
        T = atlas.tagmap[y0:y0 + h, x0:x0 + w]
        filled = T > 0
        if not filled.any():
            continue
        D = np.where(filled, atlas.outer[y0:y0 + h, x0:x0 + w].astype(np.float64), np.nan)
        ppm = r.px_per_m()
        line = np.zeros_like(filled)
        for axis in (0, 1):
            n = D.shape[axis]
            a = np.take(D, range(0, n - 1), axis=axis); b = np.take(D, range(1, n), axis=axis)
            ta = np.take(T, range(0, n - 1), axis=axis); tb = np.take(T, range(1, n), axis=axis)
            both = ~np.isnan(a) & ~np.isnan(b)
            step = both & ((np.abs(a - b) > depth_m) | (ta != tb))
            edge = np.isnan(a) ^ np.isnan(b)
            ma = (step & (a <= b)) | (edge & ~np.isnan(a))
            mb = (step & (b < a)) | (edge & ~np.isnan(b))
            pad = [(0, 0), (0, 0)]; pad[axis] = (0, 1); line |= np.pad(ma, pad)
            pad = [(0, 0), (0, 0)]; pad[axis] = (1, 0); line |= np.pad(mb, pad)
            if n > 2:
                c0 = np.take(D, range(0, n - 2), axis=axis); c1 = np.take(D, range(1, n - 1), axis=axis)
                c2 = np.take(D, range(2, n), axis=axis)
                ok = ~np.isnan(c0) & ~np.isnan(c1) & ~np.isnan(c2)
                cr = ok & (np.abs(c0 - 2 * c1 + c2) > crease_m * max(1.0, 40.0 / ppm))
                pad = [(0, 0), (0, 0)]; pad[axis] = (1, 1); line |= np.pad(cr, pad)
        sel = filled.copy()
        if ids is not None:
            sel &= np.isin(T, list(ids))
        if skip:
            sel &= ~np.isin(T, list(skip))
        line &= sel
        hal = line.copy()
        hal[1:, :] |= line[:-1, :]; hal[:-1, :] |= line[1:, :]; hal[:, 1:] |= line[:, :-1]; hal[:, :-1] |= line[:, 1:]
        hal &= sel & ~line
        img = atlas.img[y0:y0 + h, x0:x0 + w]
        img[line] *= (1.0 - strength)
        img[hal] *= (1.0 - halo)


# ---------------------------------------------------------------------------------- weathering
def grime(atlas, tags, y_top, color=(96, 84, 66), amount=0.55, scale=0.7, seed=3, kinds=KINDS, y_bottom=None):
    """Dirt rising from y_bottom (default: lowest painted point) up to y_top, broken up by noise."""
    col = np.array(color, np.float32)

    def f(X, Y, Z, cur):
        yb = Y.min() if y_bottom is None else y_bottom
        t = np.clip((y_top - Y) / max(y_top - yb, 1e-3), 0, 1) ** 1.3
        n = value_noise(X + Z * 0.7, Y * 2.0 + Z * 0.3, scale, seed=seed)
        k = np.clip(t * amount * (0.55 + 0.9 * n), 0, 0.85)[:, None]
        return cur * (1 - k) + col * k, None
    atlas.paint3d(f, tags=tags, kinds=kinds)


def streaks(atlas, tags, color=(70, 62, 52), amount=0.18, width=0.25, seed=5, kinds=("side", "front"), axis="z"):
    """Vertical rain / rust streaks (noise stretched along Y)."""
    col = np.array(color, np.float32)

    def f(X, Y, Z, cur):
        A = Z if axis == "z" else X
        n = value_noise(A / width, Y * 0.08, 1.0, seed=seed, octaves=2)
        k = (np.clip((n - 0.55) * 3.0, 0, 1) * amount)[:, None]
        return cur * (1 - k) + col * k, None
    atlas.paint3d(f, tags=tags, kinds=kinds)


def chips(atlas, tags, color=(70, 70, 66), density=0.04, scale=0.06, seed=9, kinds=KINDS):
    """Small worn-paint speckles."""
    col = np.array(color, np.float32)

    def f(X, Y, Z, cur):
        m = value_noise(X + Y * 1.3, Z + Y * 0.7, scale, seed=seed, octaves=1) > (1.0 - density)
        out = cur.copy(); out[m] = out[m] * 0.4 + col * 0.6
        return out, None
    atlas.paint3d(f, tags=tags, kinds=kinds)


def tone(atlas, tags, amount=0.05, scale=1.2, seed=13, kinds=KINDS):
    """Per-panel brightness variation (faded / repainted panels)."""
    def f(X, Y, Z, cur):
        n = value_noise(np.floor(X / scale) * 7.1 + np.floor(Y / scale) * 3.3, np.floor(Z / scale) * 5.7, 1.0, seed=seed, octaves=1)
        return cur * (1.0 + (n - 0.5) * 2 * amount)[:, None], None
    atlas.paint3d(f, tags=tags, kinds=kinds)


def soot(atlas, tags, center, radius, color=(36, 32, 30), amount=0.5, kinds=KINDS, seed=17):
    """Exhaust soot fading out from a point."""
    c = np.asarray(center, float); col = np.array(color, np.float32)

    def f(X, Y, Z, cur):
        d = np.sqrt((X - c[0]) ** 2 + (Y - c[1]) ** 2 + (Z - c[2]) ** 2)
        n = value_noise(X * 2, Z * 2 + Y, 0.4, seed=seed)
        k = (np.clip(1 - d / radius, 0, 1) ** 1.5 * amount * (0.6 + 0.8 * n))[:, None]
        return cur * (1 - k) + col * k, None
    atlas.paint3d(f, tags=tags, kinds=kinds)


# ---------------------------------------------------------------------------------- lines and panels
def lines_fn(coord, step, width, offset=0.0, k=0.6, lo=-1e9, hi=1e9, mask_fn=None):
    """paint3d fn: darken where coord ('x'|'y'|'z') crosses multiples of step (frames, strakes, deck seams)."""
    def f(X, Y, Z, cur):
        C = {"x": X, "y": Y, "z": Z}[coord]
        m = (np.abs((C - offset) % step) < width) & (C > lo) & (C < hi)
        if mask_fn is not None:
            m &= mask_fn(X, Y, Z)
        out = cur.copy(); out[m] *= k
        return out, m
    return f


def panel_lines(atlas, tags, frame_step=1.15, span_step=0.9, row_step=0.55, offset=0.3, k=0.72, span_from=1.3):
    """Aircraft skin: frames across the fuselage, spanwise lines on the wings (|x| > span_from), strakes on
    the sides."""
    atlas.paint3d(lines_fn("z", frame_step, 0.018, offset=offset, k=k), tags=tags, kinds=("side", "top", "bottom"))
    atlas.paint3d(lines_fn("x", span_step, 0.016, offset=span_step / 2, k=k + 0.08,
                           mask_fn=lambda X, Y, Z: np.abs(X) > span_from), tags=tags, kinds=("top", "bottom"))
    atlas.paint3d(lines_fn("y", row_step, 0.014, offset=0.2, k=k + 0.1), tags=tags, kinds=("side",))


def rivets(atlas, tags, frame_step=1.15, offset=0.3, k=0.85, pitch=0.5):
    def f(X, Y, Z, cur):
        a = (Z - offset) % frame_step
        m = ((np.abs(a - 0.07) < 0.014) | (np.abs(a - frame_step + 0.07) < 0.014)) & (((Y * 7 + X * 7) % pitch) < 0.12)
        out = cur.copy(); out[m] *= k
        return out, m
    atlas.paint3d(f, tags=tags, kinds=("side", "top"))


def access_panels(atlas, tags, cell=(1.15, 0.55, 0.9), density=0.18, k=0.68, offset=0.3, kinds=("side", "top")):
    """Outlines of a scattered set of access panels on a hashed grid (deterministic)."""
    cz, cy, cx = cell

    def f(X, Y, Z, cur):
        iz = np.floor((Z - offset) / cz); iy = np.floor(Y / cy); ix = np.floor(X / cx)
        h = (np.sin(iz * 12.9898 + iy * 78.233 + ix * 37.719) * 43758.5453) % 1.0
        fz = ((Z - offset) % cz) / cz; fy = (Y % cy) / cy
        border = (np.abs(fz - 0.22) < 0.02) | (np.abs(fz - 0.78) < 0.02) | (np.abs(fy - 0.2) < 0.04) | (np.abs(fy - 0.8) < 0.04)
        inside = (fz > 0.2) & (fz < 0.8) & (fy > 0.18) & (fy < 0.82)
        m = (h > 1.0 - density) & border & inside
        out = cur.copy(); out[m] *= k
        return out, m
    atlas.paint3d(f, tags=tags, kinds=kinds)


def rect_fn(rects, k=0.55, fill=None, border=0.03):
    """paint3d fn drawing outlined boxes: rects = [(a_axis, a0, a1, b_axis, b0, b1, c_axis, c0, c1)]."""
    def f(X, Y, Z, cur):
        C = {"x": X, "y": Y, "z": Z}
        out = cur.copy(); tot = np.zeros(len(X), bool)
        for (aa, a0, a1, bb, b0, b1, cc, c0, c1) in rects:
            A, B, Cc = C[aa], C[bb], C[cc]
            inside = (A > a0) & (A < a1) & (B > b0) & (B < b1) & (Cc > c0) & (Cc < c1)
            edge = inside & ((A < a0 + border) | (A > a1 - border) | (B < b0 + border) | (B > b1 - border))
            if fill is not None:
                out[inside] = fill
            out[edge] *= k
            tot |= inside
        return out, tot
    return f


def grid_fn(a_coord, b_coord, a0, a1, b0, b1, pitch_a, pitch_b, gap=0.08, color=(66, 68, 70), extra=None):
    """paint3d fn: a grid of cells (VLS lids, grilles, deck tiles) inside an (a, b) rectangle."""
    col = np.array(color, np.float32)

    def f(X, Y, Z, cur):
        C = {"x": X, "y": Y, "z": Z}
        A, B = C[a_coord], C[b_coord]
        inside = (A > a0) & (A < a1) & (B > b0) & (B < b1)
        if extra is not None:
            inside &= extra(X, Y, Z)
        cell = inside & (((A - a0) % pitch_a) > gap) & (((B - b0) % pitch_b) > gap)
        out = cur.copy(); out[inside] *= 0.7; out[cell] = col
        return out, inside
    return f


def wheel_hubs(atlas, tags, wheels, r, kinds=("side",)):
    """Road-wheel faces: tyre ring, hub, bolt circle.  wheels = [(y, z)] centres in the side view."""
    def f(X, Y, Z, cur):
        out = cur.copy(); tot = np.zeros(len(X), bool)
        for (wy, wz) in wheels:
            d = np.hypot(Y - wy, Z - wz)
            near = d < r * 1.05
            if not near.any():
                continue
            ang = np.degrees(np.arctan2(Y - wy, Z - wz))
            out[near & (d > r * 0.82)] = (36, 36, 34)
            out[near & (d < r * 0.28)] *= 0.75
            out[near & (((np.abs(d - r * 0.42) < 0.03) & ((ang % 45) < 12)) | (np.abs(d - r * 0.6) < 0.02))] *= 0.5
            tot |= near
        return out, tot
    atlas.paint3d(f, tags=tags, kinds=kinds)


# ---------------------------------------------------------------------------------- numbers
SEG = {"0": "abcdef", "1": "bc", "2": "abged", "3": "abgcd", "4": "fgbc", "5": "afgcd", "6": "afgedc", "7": "abc",
       "8": "abcdefg", "9": "abcdfg", "A": "abcefg", "C": "adef", "D": "bcdeg", "E": "adefg", "F": "aefg", "G": "acdef",
       "H": "bcefg", "L": "def", "P": "abefg", "U": "bcdef", "-": "g", " ": ""}


def digit_mask(A, B, text, a0, b0, h, gap=0.25, thick=0.16):
    """7-segment text, lower-left corner (a0, b0), height h, reading along +A."""
    w = h * 0.55; t = h * thick
    boxes = {"a": (0, w, h - t, h), "g": (0, w, h / 2 - t / 2, h / 2 + t / 2), "d": (0, w, 0, t),
             "f": (0, t, h / 2, h), "b": (w - t, w, h / 2, h), "e": (0, t, 0, h / 2), "c": (w - t, w, 0, h / 2)}
    m = np.zeros(len(A), bool); x = a0
    for ch in text.upper():
        for s in SEG.get(ch, ""):
            a1, a2, b1, b2 = boxes[s]
            m |= (A > x + a1) & (A < x + a2) & (B > b0 + b1) & (B < b0 + b2)
        x += w + h * gap
    return m


def text_width(text, h, gap=0.25):
    return len(text) * h * (0.55 + gap) - h * gap


def side_number(atlas, text, tags, z_center, y0, h, color=(236, 236, 232), shadow=(40, 40, 40)):
    """Hull / side number readable from BOTH sides (needs texture_model(split_sides=True)): on the port
    side (+X, bow to the left of the picture) the text runs towards -Z."""
    w = text_width(text, h)

    def make(side):
        def f(X, Y, Z, cur):
            A = -Z if side > 0 else Z
            a0 = (-z_center if side > 0 else z_center) - w / 2
            m = digit_mask(A, Y, text, a0, y0, h)
            sh = digit_mask(A - h * 0.06, Y + h * 0.06, text, a0, y0, h) & ~m
            sel = (X > 0) if side > 0 else (X < 0)
            m &= sel; sh &= sel
            out = cur.copy(); out[sh] = shadow; out[m] = color
            return out, m | sh
        return f
    atlas.paint3d(make(1), tags=tags, kinds=("side",))
    atlas.paint3d(make(-1), tags=tags, kinds=("side",))
