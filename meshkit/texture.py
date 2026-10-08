"""Projection-atlas texturing.

Every face is mapped into one of four orthographic views of the vehicle - plan view
from above ('top'), from below ('bottom'), profile ('side', left and right share it
mirrored) and front/back ('front', shared mirrored) - chosen by its dominant normal
axis. The views are packed into one opaque texture together with a strip of flat
colour swatches for small parts (tyres, propellers, gun metal...).

Faces are first rasterised into a per-view 'tag map' (painter's algorithm, outermost
surface last) so the painting stage knows which surface type owns every texel, then
base colours, camouflage, insignia, panel lines and weathering are painted on top as
functions of real model coordinates.
"""
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

KINDS = ("top", "bottom", "side", "front")


def face_normal(P):
    n = np.zeros(3)
    m = len(P)
    for i in range(m):
        n += np.cross(P[i], P[(i + 1) % m])
    ln = np.linalg.norm(n)
    return n / ln if ln > 1e-12 else np.array([0.0, 1.0, 0.0])


def dominant_kind(n):
    ax = np.abs(n)
    # small bias so near-45deg faces prefer top/bottom (plan views carry most detail)
    ax = ax * np.array([1.0, 1.04, 1.0])
    k = int(np.argmax(ax))
    if k == 1:
        return "top" if n[1] > 0 else "bottom"
    if k == 0:
        return "side"
    return "front"


def _depth(kind, Q):
    base = kind.split("#")[0]
    if base == "top":
        return Q[:, 1]
    if base == "bottom":
        return -Q[:, 1]
    if base == "side":
        return np.abs(Q[:, 0])
    return np.abs(Q[:, 2])


def _unproject(kind, A, B, dep, Q):
    """Projection coords (+depth) back to 3D on the face's own side; also the outward view dir."""
    base = kind.split("#")[0]
    n = len(A)
    if base == "top":
        return np.stack([A, dep, B], 1), np.tile([0.0, 1.0, 0.0], (n, 1))
    if base == "bottom":
        return np.stack([A, -dep, B], 1), np.tile([0.0, -1.0, 0.0], (n, 1))
    if base == "side":
        sg = 1.0 if Q[:, 0].mean() >= 0 else -1.0
        return np.stack([sg * dep, B, A], 1), np.tile([sg, 0.0, 0.0], (n, 1))
    sg = 1.0 if Q[:, 2].mean() >= 0 else -1.0
    return np.stack([A, B, sg * dep], 1), np.tile([0.0, 0.0, sg], (n, 1))


def _inside(points, tris):
    """Even-odd ray test of points (N,3) against a closed triangle soup (M,3,3)."""
    d = np.array([0.01234, 1.0, 0.03721]); d /= np.linalg.norm(d)
    v0 = tris[:, 0]; e1 = tris[:, 1] - v0; e2 = tris[:, 2] - v0
    h = np.cross(d, e2)
    a = np.einsum("ij,ij->i", e1, h)
    ok = np.abs(a) > 1e-12
    f = np.where(ok, 1.0 / np.where(ok, a, 1.0), 0.0)
    sv = points[:, None, :] - v0[None, :, :]
    u = f[None, :] * np.einsum("nmk,mk->nm", sv, h)
    q = np.cross(sv, e1[None, :, :])
    v = f[None, :] * np.einsum("k,nmk->nm", d, q)
    t = f[None, :] * np.einsum("mk,nmk->nm", e2, q)
    hit = ok[None, :] & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-9)
    return (hit.sum(1) % 2) == 1


class Region:
    rot_plan = False     # class-level default; Atlas sets per instance for long ships

    def __init__(self, kind, a0, a1, b0, b1, rot=False):
        self.kind = kind
        self.base = kind.split("#")[0]      # 'side#hull#3' = extra region for occluded faces
        self.a0, self.a1, self.b0, self.b1 = a0, a1, b0, b1
        self.x0 = self.y0 = 0
        self.w = self.h = 0
        self.rot = rot and self.base in ("top", "bottom")

    def ab(self, P):
        P = np.asarray(P, float).reshape(-1, 3)
        if self.base in ("top", "bottom"):
            if self.rot:
                return P[:, 2], P[:, 0]
            return P[:, 0], P[:, 2]
        if self.base == "side":
            return P[:, 2], P[:, 1]
        return P[:, 0], P[:, 1]

    def ab_to_px(self, a, b):
        a = np.asarray(a, float); b = np.asarray(b, float)
        fa = (a - self.a0) / (self.a1 - self.a0)
        fb = (self.b1 - b) / (self.b1 - self.b0)
        if self.base == "top" and not self.rot:
            fa = 1.0 - fa  # +X (left side) drawn on the left, nose up
        return self.x0 + fa * self.w, self.y0 + fb * self.h

    def to_px(self, P):
        a, b = self.ab(P)
        return np.stack(self.ab_to_px(a, b), 1)

    def grid(self):
        """Model-space (a, b) coordinates of every texel centre in this region."""
        xs = np.arange(self.w) + 0.5; ys = np.arange(self.h) + 0.5
        X, Y = np.meshgrid(xs, ys)
        fa = X / self.w; fb = Y / self.h
        if self.base == "top" and not self.rot:
            fa = 1.0 - fa
        A = self.a0 + fa * (self.a1 - self.a0)
        B = self.b1 - fb * (self.b1 - self.b0)
        return A, B

    def px_per_m(self):
        return self.w / (self.a1 - self.a0)


class Atlas:
    def __init__(self, size=(512, 512), cell=8, pad=3):
        self.W, self.H = size
        self.cell = cell
        self.pad = pad
        self.regions = {}
        self.swatches = {}          # tag -> (x, y) cell origin
        self.flat_tags = {}         # tag -> colour (or (top, bottom) gradient)
        self.tag_ids = {"": 0}
        self.faces = []             # (part, shell_idx, face_idx, kind or 'flat', tag, P)
        self.img = None
        self.tagmap = None
        self.glass_tags = ()        # tags of see-through surfaces: charted in regions of their own, so they never
                                    # take the texels of what lies behind them (the skin under a canopy)
        self.split_sides = False    # True: the right (-X) sides get texels of their own (asymmetric
                                    # markings such as numbers / lettering read correctly on both sides)

    # -------------------------------------------------------------- setup
    def tid(self, tag):
        if tag not in self.tag_ids:
            self.tag_ids[tag] = len(self.tag_ids)
        return self.tag_ids[tag]

    def set_flat(self, tag, color):
        self.flat_tags[tag] = color

    def collect(self, model):
        self.faces = []
        for pname in model.order:
            for si, sh in enumerate(model.parts[pname].shells):
                P = sh.V()
                for fi, (f, tag) in enumerate(zip(sh.faces, sh.tags)):
                    Q = P[list(f)]
                    if tag in self.flat_tags:
                        kind = "flat"
                    else:
                        nrm = face_normal(Q)
                        kind = dominant_kind(nrm)
                        if self.split_sides and kind == "side" and nrm[0] < 0:
                            kind = "side#R"
                        if tag in self.glass_tags:
                            kind = kind.split("#")[0] + "#" + ("liner" if tag == "liner" else "glass")
                    self.faces.append((pname, si, fi, kind, tag, Q))

    def assign_layers(self, keep=0.9, res=700, max_samples=240):
        """Occlusion-aware charts. Faces sharing a projection overlap wherever one surface lies
        behind another (a funnel behind the bridge in the front view, a hull side behind a boat
        hanging outboard...). A texel can hold only one surface, so a face whose *exposed* part
        is hidden in its view by a different shell is moved to a compact region of its own
        ('side#<part>#<shell>'). Parts of a face that are buried inside another shell (the deck
        under a deckhouse) are invisible in 3D and do not count."""
        idx_by_kind = {}
        for i, (pname, si, fi, kind, tag, Q) in enumerate(self.faces):
            if kind != "flat":
                idx_by_kind.setdefault(kind, []).append(i)
        self.moved_faces = 0
        if not idx_by_kind:
            return 0
        allp = np.vstack([self.faces[i][5] for ids in idx_by_kind.values() for i in ids])
        cell = max(float(np.ptp(allp, 0).max()) / res, 1e-3)
        eps = max(1.5 * cell, 0.01)
        # every shell (incl. flat-tagged ones) as triangles + AABB for the buried test
        shells = {}
        for (pname, si, fi, kind, tag, Q) in self.faces:
            shells.setdefault((pname, si), []).extend(Q[[0, k, k + 1]] for k in range(1, len(Q) - 1))
        shell_tris = {k: np.array(v) for k, v in shells.items()}
        shell_box = {k: (t.reshape(-1, 3).min(0), t.reshape(-1, 3).max(0)) for k, t in shell_tris.items()}
        rng = np.random.default_rng(7)
        moved = 0
        for kind, ids in idx_by_kind.items():
            reg = Region(kind, 0, 1, 0, 1, False)
            AB = []
            for i in ids:
                Q = self.faces[i][5]
                a, b = reg.ab(Q)
                AB.append((a, b, _depth(kind, Q)))
            a0 = min(x[0].min() for x in AB); b0 = min(x[1].min() for x in AB)
            a1 = max(x[0].max() for x in AB); b1 = max(x[1].max() for x in AB)
            nx = int(math.ceil((a1 - a0) / cell)) + 2; ny = int(math.ceil((b1 - b0) / cell)) + 2
            D = np.full(ny * nx, -np.inf); OWN = np.full(ny * nx, -1, np.int64)
            cov = []
            for k, (a, b, d) in enumerate(AB):
                cells = []; dep = []
                ga = (a - a0) / cell; gb = (b - b0) / cell
                for t in range(1, len(a) - 1):
                    tri = [0, t, t + 1]
                    xa, ya, da = ga[tri], gb[tri], d[tri]
                    x0 = int(np.floor(xa.min())); x1 = int(np.ceil(xa.max()))
                    y0 = int(np.floor(ya.min())); y1 = int(np.ceil(ya.max()))
                    X, Y = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
                    area = (xa[1] - xa[0]) * (ya[2] - ya[0]) - (ya[1] - ya[0]) * (xa[2] - xa[0])
                    if abs(area) < 1e-12:
                        continue
                    w0 = ((xa[1] - X) * (ya[2] - Y) - (ya[1] - Y) * (xa[2] - X)) / area
                    w1 = ((xa[2] - X) * (ya[0] - Y) - (ya[2] - Y) * (xa[0] - X)) / area
                    w2 = 1.0 - w0 - w1
                    inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
                    if not inside.any():
                        continue
                    ci = (np.floor(Y[inside]).astype(np.int64) * nx + np.floor(X[inside]).astype(np.int64))
                    cells.append(ci); dep.append((w0 * da[0] + w1 * da[1] + w2 * da[2])[inside])
                if not cells:   # sub-cell face: sample its centroid
                    cx = int(np.floor(ga.mean())); cy = int(np.floor(gb.mean()))
                    cells = [np.array([cy * nx + cx])]; dep = [np.array([d.mean()])]
                ci = np.concatenate(cells); dd = np.concatenate(dep)
                cov.append((ci, dd))
                better = dd > D[ci]
                D[ci[better]] = dd[better]; OWN[ci[better]] = k
            keys = {}
            shell_key = [(self.faces[i][0], self.faces[i][1]) for i in ids]
            shell_id = np.array([keys.setdefault(sk, len(keys)) for sk in shell_key], np.int64)
            for k, (ci, dd) in enumerate(cov):
                bad = (dd < D[ci] - eps) & (shell_id[OWN[ci]] != shell_id[k])
                nbad = int(bad.sum())
                if nbad == 0 or nbad < (1.0 - keep) * len(ci):
                    continue
                # which of the hidden cells are exposed in 3D (not buried inside another shell)?
                sel = np.nonzero(bad)[0]
                if len(sel) > max_samples:
                    sel = rng.choice(sel, max_samples, replace=False)
                cc = ci[sel]; dsel = dd[sel]
                A = a0 + (cc % nx + 0.5) * cell; B = b0 + (cc // nx + 0.5) * cell
                Q = self.faces[ids[k]][5]
                P, out_dir = _unproject(kind, A, B, dsel, Q)
                P = P + out_dir * max(0.01, 0.5 * cell)
                buried = np.zeros(len(P), bool)
                for sk, (lo, hi) in shell_box.items():
                    if sk == shell_key[k]:
                        continue
                    cand = ~buried & np.all((P >= lo - 1e-6) & (P <= hi + 1e-6), axis=1)
                    if cand.any():
                        buried[cand] = _inside(P[cand], shell_tris[sk])
                exposed_bad = (~buried).mean() * nbad
                if exposed_bad >= (1.0 - keep) * len(ci):
                    i = ids[k]
                    pname, si, fi, _, tag, Q = self.faces[i]
                    self.faces[i] = (pname, si, fi, f"{kind}#{pname}#{si}", tag, Q)
                    moved += 1
        self.moved_faces = moved
        return moved

    def layout(self, margin=0.06, layers=True):
        if layers:
            self.assign_layers()
        bounds = {}
        allp = np.vstack([Q for (_, _, _, k, _, Q) in self.faces]) if self.faces else np.zeros((1, 3))
        ext = np.ptp(allp, 0)
        rot = ext[2] > 2.5 * max(ext[0], 1e-6)       # long ships: lay plan views lengthwise
        self.rot_plan = rot
        for (_, _, _, kind, _, Q) in self.faces:
            if kind == "flat":
                continue
            r = Region(kind, 0, 1, 0, 1, rot)
            a, b = r.ab(Q)
            lo = bounds.setdefault(kind, [1e9, -1e9, 1e9, -1e9])
            lo[0] = min(lo[0], a.min()); lo[1] = max(lo[1], a.max())
            lo[2] = min(lo[2], b.min()); lo[3] = max(lo[3], b.max())
        for kind, (a0, a1, b0, b1) in bounds.items():
            m = margin if "#" not in kind else min(margin, 0.02)
            self.regions[kind] = Region(kind, a0 - m, a1 + m, b0 - m, b1 + m, rot)
        # swatch block
        ntags = len(self.flat_tags)
        cols = max(1, min(16, ntags))
        rows = max(1, math.ceil(ntags / cols))
        sw_w, sw_h = cols * self.cell, rows * self.cell
        # binary search the largest uniform scale that shelf-packs everything
        lo_s, hi_s = 0.1, 4096.0
        best = None
        for _ in range(40):
            s = (lo_s + hi_s) / 2
            pack = self._try_pack(s, sw_w, sw_h)
            if pack is not None:
                best = (s, pack); lo_s = s
            else:
                hi_s = s
        assert best is not None, "texture too small"
        s, pack = best
        for kind, (x, y, w, h) in pack.items():
            if kind == "_sw":
                self.sw_origin = (x, y); continue
            r = self.regions[kind]
            r.x0, r.y0, r.w, r.h = x, y, w, h
        i = 0
        for tag in self.flat_tags:
            cx = self.sw_origin[0] + (i % cols) * self.cell
            cy = self.sw_origin[1] + (i // cols) * self.cell
            self.swatches[tag] = (cx, cy); i += 1
        self.scale = s

    def _try_pack(self, s, sw_w, sw_h):
        items = []
        for kind, r in self.regions.items():
            w = max(4, int(math.floor((r.a1 - r.a0) * s))); h = max(4, int(math.floor((r.b1 - r.b0) * s)))
            items.append((kind, w, h))
        items.append(("_sw", sw_w, sw_h))
        items.sort(key=lambda t: -t[2])
        p = self.pad
        x = y = 0; row_h = 0; out = {}
        for kind, w, h in items:
            W2, H2 = w + 2 * p, h + 2 * p
            if W2 > self.W:
                return None
            if x + W2 > self.W:
                x = 0; y += row_h; row_h = 0
            if y + H2 > self.H:
                return None
            out[kind] = (x + p, y + p, w, h)
            x += W2; row_h = max(row_h, H2)
        return out

    # -------------------------------------------------------------- raster
    def rasterize(self):
        """Per-texel owner tag + the exact 3D surface point it maps to (outermost surface
        wins where projections overlap). Conservative coverage (0.75 px) so UV sampling at
        polygon edges always lands on a texel painted for that surface."""
        H, W = self.H, self.W
        self.tagmap = np.zeros((H, W), np.int32)
        self.pmap = np.zeros((H, W, 3), np.float32)
        self.outer = np.full((H, W), -np.inf, np.float32)
        for (pname, si, fi, kind, tag, Q) in self.faces:
            if kind == "flat":
                continue
            r = self.regions[kind]
            px = r.to_px(Q)
            if r.base == "top":
                depth = Q[:, 1]
            elif r.base == "bottom":
                depth = -Q[:, 1]
            elif r.base == "side":
                depth = np.abs(Q[:, 0])
            else:
                depth = np.abs(Q[:, 2])
            tid = self.tid(tag)
            for k in range(1, len(Q) - 1):
                idx = [0, k, k + 1]
                self._tri(px[idx], Q[idx], depth[idx], tid)
        self.img = np.zeros((H, W, 3), np.float32)
        self.alpha = np.full((H, W), 255.0, np.float32)
        self.painted = np.zeros((H, W), bool)

    def _tri(self, p, Q, d, tid, tol=0.75):
        x0 = max(int(np.floor(p[:, 0].min() - 1)), 0); x1 = min(int(np.ceil(p[:, 0].max() + 1)), self.W - 1)
        y0 = max(int(np.floor(p[:, 1].min() - 1)), 0); y1 = min(int(np.ceil(p[:, 1].max() + 1)), self.H - 1)
        if x1 < x0 or y1 < y0:
            return
        X, Y = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
        (ax, ay), (bx, by), (cx, cy) = p
        area2 = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
        if abs(area2) < 1e-9:
            # degenerate in texture space (edge-on) - just splat the vertices
            for (qx, qy) in p:
                ix, iy = int(qx), int(qy)
                if 0 <= ix < self.W and 0 <= iy < self.H and self.tagmap[iy, ix] == 0:
                    self.tagmap[iy, ix] = tid; self.pmap[iy, ix] = Q.mean(0)
            return
        sgn = 1.0 if area2 > 0 else -1.0
        # signed distances (px) to each edge, positive inside
        def edge(ux, uy, vx, vy):
            L = np.hypot(vx - ux, vy - uy) + 1e-12
            return sgn * ((vx - ux) * (Y - uy) - (vy - uy) * (X - ux)) / L
        e0 = edge(bx, by, cx, cy); e1 = edge(cx, cy, ax, ay); e2 = edge(ax, ay, bx, by)
        inside = (e0 >= -tol) & (e1 >= -tol) & (e2 >= -tol)
        if not inside.any():
            return
        w0 = ((bx - X) * (cy - Y) - (by - Y) * (cx - X)) / area2
        w1 = ((cx - X) * (ay - Y) - (cy - Y) * (ax - X)) / area2
        w2 = 1.0 - w0 - w1
        dd = w0 * d[0] + w1 * d[1] + w2 * d[2]
        sub_o = self.outer[y0:y1 + 1, x0:x1 + 1]
        win = inside & (dd >= sub_o - 1e-4)
        if not win.any():
            return
        sub_o[win] = dd[win]
        self.tagmap[y0:y1 + 1, x0:x1 + 1][win] = tid
        P = (w0[..., None] * Q[0] + w1[..., None] * Q[1] + w2[..., None] * Q[2])
        self.pmap[y0:y1 + 1, x0:x1 + 1][win] = P[win]

    def paint3d(self, fn, tags=None, kinds=KINDS):
        """fn(X, Y, Z, cur) -> (rgb, mask|None) evaluated on the true 3D surface point of every texel."""
        m = self.mask(tags, kinds)
        if not m.any():
            return
        P = self.pmap[m]
        cur = self.img[m]
        res = fn(P[:, 0], P[:, 1], P[:, 2], cur.copy())
        if res is None:
            return
        rgb, mm = res
        if mm is None:
            mm = np.ones(len(P), bool)
        rgb = np.broadcast_to(np.asarray(rgb, np.float32), cur.shape) if np.ndim(rgb) == 1 else rgb
        cur[mm] = rgb[mm]
        self.img[m] = cur

    def uvs(self):
        """Per face (in collect order) list of (u, v) per corner (OBJ convention, v up)."""
        out = {}
        for (pname, si, fi, kind, tag, Q) in self.faces:
            if kind == "flat":
                cx, cy = self.swatches[tag]
                c = self.cell
                u = (cx + c / 2) / self.W; v = 1.0 - (cy + c / 2) / self.H
                out[(pname, si, fi)] = [(u, v)] * len(Q)
            else:
                px = self.regions[kind].to_px(Q)
                out[(pname, si, fi)] = [(p[0] / self.W, 1.0 - p[1] / self.H) for p in px]
        return out

    # -------------------------------------------------------------- painting
    def mask(self, tags=None, kinds=KINDS):
        m = np.zeros((self.H, self.W), bool)
        ids = None if tags is None else [self.tag_ids[t] for t in tags if t in self.tag_ids]
        for r in self.regions.values():
            if r.base not in kinds:
                continue
            sub = self.tagmap[r.y0:r.y0 + r.h, r.x0:r.x0 + r.w]
            mm = (sub > 0) if ids is None else np.isin(sub, ids)
            m[r.y0:r.y0 + r.h, r.x0:r.x0 + r.w] |= mm
        return m

    def paint_alpha(self, fn, tags=None, kinds=KINDS):
        """Per-texel opacity (0..255) for translucent surfaces (canopy glass...). fn(X, Y, Z) -> alpha
        array, or a plain number. The mod bakes partial alpha into a 4x4 ordered dither."""
        m = self.mask(tags, kinds)
        if not m.any():
            return
        if callable(fn):
            P = self.pmap[m]
            a = np.broadcast_to(np.asarray(fn(P[:, 0], P[:, 1], P[:, 2]), np.float32), (len(P),))
        else:
            a = float(fn)
        self.alpha[m] = np.clip(a, 0, 255)

    def cut_alpha(self, fn, tags=None, kinds=KINDS, value=0.0):
        """Set the opacity of the texels where fn(X, Y, Z) is True to `value` (default 0: openings such as a cockpit
        cut into the fuselage skin under the canopy); every other texel keeps its opacity."""
        m = self.mask(tags, kinds)
        if not m.any():
            return 0
        P = self.pmap[m]
        hit = np.asarray(fn(P[:, 0], P[:, 1], P[:, 2]), bool)
        a = self.alpha[m]
        a[hit] = float(value)
        self.alpha[m] = a
        return int(hit.sum())

    def fill(self, tags, color, kinds=KINDS):
        m = self.mask(tags, kinds)
        self.img[m] = np.array(color[:3], np.float32)
        self.painted |= m

    def paint_fn(self, kind, fn, tags=None):
        """fn(A, B, cur) -> (rgb array HxWx3 or None, mask HxW or None); A/B = model coords."""
        r = self.regions.get(kind)
        if r is None:
            return
        A, B = r.grid()
        sl = (slice(r.y0, r.y0 + r.h), slice(r.x0, r.x0 + r.w))
        cur = self.img[sl]
        res = fn(A, B, cur)
        if res is None:
            return
        rgb, m = res
        tm = self.mask(tags, (kind,))[sl]
        if m is None:
            m = np.ones_like(tm)
        m = m & tm
        cur[m] = rgb[m] if rgb.ndim == 3 else rgb
        self.img[sl] = cur

    def swatch_paint(self):
        c = self.cell
        for tag, (x, y) in self.swatches.items():
            col = self.flat_tags[tag]
            if isinstance(col[0], (tuple, list)):
                top, bot = np.array(col[0], float), np.array(col[1], float)
                for j in range(c):
                    t = j / (c - 1)
                    self.img[y + j, x:x + c] = top * (1 - t) + bot * t
            else:
                self.img[y:y + c, x:x + c] = np.array(col[:3], float)
            self.painted[y:y + c, x:x + c] = True

    def dilate(self, iters=6):
        img = self.img; m = self.painted.copy()
        al = getattr(self, "alpha", None)
        for _ in range(iters):
            acc = np.zeros_like(img); cnt = np.zeros(m.shape, np.float32)
            acc_a = np.zeros(m.shape, np.float32)
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                sm = np.roll(m, (dy, dx), (0, 1)); si = np.roll(img, (dy, dx), (0, 1))
                acc[sm] += si[sm]; cnt[sm] += 1
                if al is not None:
                    sa = np.roll(al, (dy, dx), (0, 1)); acc_a[sm] += sa[sm]
            new = (~m) & (cnt > 0)
            img[new] = acc[new] / cnt[new][:, None]
            if al is not None:
                al[new] = acc_a[new] / cnt[new]
            m |= new
        self.painted = m

    def save(self, path):
        rgb = np.clip(self.img, 0, 255).astype(np.uint8)
        al = getattr(self, "alpha", None)
        a = (np.full(rgb.shape[:2], 255, np.uint8) if al is None else np.clip(np.round(al), 0, 255).astype(np.uint8))[..., None]
        Image.fromarray(np.concatenate([rgb, a], 2), "RGBA").save(path)
        return path

    def has_partial_alpha(self):
        al = getattr(self, "alpha", None)
        return al is not None and bool(np.any((al > 0.5) & (al < 254.5)))


# ---------------------------------------------------------------------------
# procedural pattern helpers (all operate on model-coordinate grids)
# ---------------------------------------------------------------------------
def value_noise(A, B, scale, seed=0, octaves=3):
    """Smooth 2D noise in model units (scale = feature size in metres). Deterministic."""
    rng = np.random.default_rng(seed)
    out = np.zeros_like(A, dtype=np.float64)
    amp = 1.0; tot = 0.0
    for o in range(octaves):
        f = scale / (2 ** o)
        gx = np.floor(A / f); gy = np.floor(B / f)
        fx = A / f - gx; fy = B / f - gy
        # hash lattice
        def h(ix, iy):
            v = (ix * 374761393 + iy * 668265263 + (seed + o * 1013) * 1442695041) & 0xFFFFFFFF
            v = (v ^ (v >> 13)) * 1274126177 & 0xFFFFFFFF
            return (v & 0xFFFF) / 65535.0
        ix = gx.astype(np.int64); iy = gy.astype(np.int64)
        v00 = h(ix, iy); v10 = h(ix + 1, iy); v01 = h(ix, iy + 1); v11 = h(ix + 1, iy + 1)
        sx = fx * fx * (3 - 2 * fx); sy = fy * fy * (3 - 2 * fy)
        v = (v00 * (1 - sx) + v10 * sx) * (1 - sy) + (v01 * (1 - sx) + v11 * sx) * sy
        out += amp * v; tot += amp; amp *= 0.5
    return out / tot


def mix(c1, c2, t):
    return np.asarray(c1, float) * (1 - t) + np.asarray(c2, float) * t


def shade(c, k):
    return tuple(float(np.clip(v * k, 0, 255)) for v in c[:3])


def star_polygon(cx, cy, r_out, r_in=None, n=5, rot=90.0):
    r_in = r_out * 0.382 if r_in is None else r_in
    pts = []
    for i in range(2 * n):
        r = r_out if i % 2 == 0 else r_in
        a = math.radians(rot + i * 180.0 / n)
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return pts


def point_in_poly(A, B, poly):
    """Vectorised even-odd test for model-coordinate grids."""
    inside = np.zeros_like(A, dtype=bool)
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]; x2, y2 = poly[(i + 1) % n]
        cond = ((y1 > B) != (y2 > B))
        with np.errstate(divide="ignore", invalid="ignore"):
            xint = (x2 - x1) * (B - y1) / (y2 - y1 + 1e-12) + x1
        inside ^= cond & (A < xint)
    return inside


# ---------------------------------------------------------------------------
# one-call pipeline
# ---------------------------------------------------------------------------
def tag_color(tag):
    """Deterministic pleasant colour for a tag name (used when no colour is given)."""
    import zlib
    h = (zlib.crc32(tag.encode()) % 360) / 360.0
    i = int(h * 6); f = h * 6 - i; s, v = 0.35, 0.8
    p, q, t = v * (1 - s), v * (1 - f * s), v * (1 - (1 - f) * s)
    r, g, b = [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)][i % 6]
    return (int(r * 255), int(g * 255), int(b * 255))


def texture_model(model, size=(512, 512), colors=None, flat=None, paint=None, glass_tags=(), split_sides=False,
                  cell=8, margin=0.06, dilate=8):
    """Projection-atlas texture in one call. Sets model.uvs and model.texture and returns the Atlas.

    colors: {tag: rgb} base colour of projected (painted) surfaces; tags not listed get tag_color(tag).
    flat:   {tag: rgb or (top_rgb, bottom_rgb)} tags drawn from a solid colour swatch instead of a
            projection (small parts: tyres, guns, propellers...). flat="all" puts every tag on swatches
            (smallest possible texture, no painting).
    paint:  optional fn(atlas, model) for camouflage / markings / panel lines (atlas.paint3d, fill...).
    """
    tags = []
    for p in model.order:
        for sh in model.parts[p].shells:
            for t in sh.tags:
                if t not in tags:
                    tags.append(t)
    atlas = Atlas(size, cell=cell)
    atlas.glass_tags = tuple(glass_tags); atlas.split_sides = split_sides
    if flat == "all":
        flat = {t: (colors or {}).get(t, tag_color(t)) for t in tags}
    for t, c in (flat or {}).items():
        atlas.set_flat(t, c)
    atlas.collect(model)
    atlas.layout(margin=margin)
    atlas.rasterize()
    for t in tags:
        if t in atlas.flat_tags:
            continue
        atlas.fill([t], (colors or {}).get(t, tag_color(t)))
    if paint is not None:
        paint(atlas, model)
    atlas.swatch_paint()
    atlas.dilate(dilate)
    model.uvs = atlas.uvs()
    rgb = np.clip(atlas.img, 0, 255).astype(np.uint8)
    a = np.clip(np.round(atlas.alpha), 0, 255).astype(np.uint8)[..., None]
    model.texture = np.concatenate([rgb, a], 2)
    return atlas
