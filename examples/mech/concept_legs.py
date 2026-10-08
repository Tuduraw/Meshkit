"""One biped leg in each of four design languages (meshkit.mech), side by side - the same joint
positions (hip / knee / ankle), only the shapes change. A worked example for AI_GUIDE chapter 7:

    light     rounded pentagons pointing forward (pstack / pseg): slim front, deep side
    skeletal  braced girders (girder) with separate stand-off plates (panel): see-through, fill ~0.1
    faceted   chined hexagon sections (chine_seg / chine_stack): straight-line silhouette
    fortress  cast superellipse masses (cast_seg / cast) and layered plates: thick, rounded-square

    python -m meshkit run examples/mech/concept_legs.py -o out --views iso,left,front --mode tags
Prints each leg's proportion numbers (mech.proportions) - compare their fill ratios. The four legs
stand apart on purpose: the report counts four contact groups, one per leg."""
import numpy as np

import meshkit as mk
from meshkit import mech

HIP = np.array([0.0, 3.0, 0.0])
KNEE = np.array([0.0, 1.7, 0.38])
ANKLE = np.array([0.0, 0.42, -0.04])


def axle(p, r, w, n=8, tag="frame"):
    return mk.cylinder((p[0] - w, p[1], p[2]), (p[0] + w, p[1], p[2]), r, n=n, tag=tag)


def light(m, dx):
    o = np.array([dx, 0, 0])
    h, k, a = HIP + o, KNEE + o, ANKLE + o
    m.add("light", axle(h, 0.17, 0.17), axle(k, 0.15, 0.15),
          mech.pseg([h + [0, -0.05, 0.02], (h + k) / 2 + [0, 0, 0.05], k + [0, 0.14, 0.02]],
                    [(0.16, 0.24, 0.15), (0.15, 0.26, 0.16), (0.11, 0.17, 0.12)], "armor"),
          mech.pseg([k + [0, -0.05, 0.02], k + (a - k) * 0.4, a + [0, 0.12, 0]], [(0.15, 0.22, 0.2), (0.17, 0.32, 0.3), (0.09, 0.12, 0.1)],
                    "armor"),
          mech.pstack([(0.0, a[0], a[2] + 0.15, 0.17, 0.62, 0.4), (0.16, a[0], a[2] + 0.14, 0.17, 0.6, 0.4),
                       (a[1] + 0.14, a[0], a[2] + 0.02, 0.1, 0.24, 0.16)], "armor2", sh=0.3))


def skeletal(m, dx):
    """Ultralight: a triangular girder with diagonal webs (it must look like it carries its own
    weight - a single bare rod reads as about to snap) and separate pointed plates on stand-off posts,
    an air gap between plate and frame and between neighbouring plates."""
    o = np.array([dx, 0, 0])
    h, k, a = HIP + o, KNEE + o, ANKLE + o
    out_ = (1, 0, 0)
    parts = [axle(h, 0.13, 0.15), axle(k, 0.12, 0.14), axle(a, 0.09, 0.1)]
    g, L, (d, f, ac) = mech.girder(h + [0, -0.04, 0], k + [0, 0.05, 0], out_, w=0.26, r=0.046, web_r=0.026)
    parts += g
    fwd = ac if ac[2] > 0 else -ac
    front, rear = ("f1", "f2") if ac[2] > 0 else ("f2", "f1")
    base = lambda t: h + (k - h) * t + f * (0.26 * 0.32 + 0.1)
    parts += mech.panel([(-0.62, -0.24), (-0.22, 0.18), (0.28, 0.17), (0.32, -0.16), (-0.1, -0.21)], base(0.25), d, fwd,
                        [mech.at(L[front], 0.25), mech.at(L[rear], 0.25)])
    parts += mech.panel([(-0.2, 0.17), (0.22, 0.18), (0.36, 0.0), (0.2, -0.17), (-0.2, -0.17)], base(0.66), d, fwd,
                        [mech.at(L[front], 0.66), mech.at(L[rear], 0.66)])
    g, L2, (d2, f2, a2) = mech.girder(k + [0, -0.04, 0], a + [0, 0.06, 0], (0, 0, 1), w=0.22)
    parts += g
    base2 = lambda t: k + (a - k) * t + f2 * (0.22 * 0.32 + 0.09)
    parts += mech.panel([(-0.3, -0.15), (-0.3, 0.16), (0.2, 0.17), (0.36, 0.0), (0.2, -0.16)], base2(0.3), d2, a2,
                        [mech.at(L2["f1"], 0.3), mech.at(L2["f2"], 0.3)])
    parts += mech.panel([(-0.17, -0.14), (-0.17, 0.14), (0.18, 0.15), (0.28, 0.0), (0.18, -0.15)], base2(0.73), d2, a2,
                        [mech.at(L2["f1"], 0.73), mech.at(L2["f2"], 0.73)])
    parts += [mech.claw(a + [0, -0.04, 0.04], a + [0, -0.4, 0.75], 0.07, 0.03, "accent"),
              mech.claw(a + [0, -0.04, -0.04], a + [0, -0.4, -0.45], 0.05, 0.03, "armor2")]
    m.add("skeletal", *parts)


def faceted(m, dx):
    o = np.array([dx, 0, 0])
    h, k, a = HIP + o, KNEE + o, ANKLE + o
    F = (0, 0, 1)
    m.add("faceted", axle(h, 0.18, 0.2, n=6), axle(k, 0.16, 0.18, n=6), axle(a, 0.1, 0.12, n=6),
          mech.chine_seg([h + [0, 0.05, 0], (h + k) / 2 + [0, 0, 0.04], k + [0, 0.16, 0]], [(0.2, 0.24, 0.2), (0.21, 0.27, 0.21), (0.15, 0.18, 0.15)],
                         "armor", up=F, k=0.5),
          mech.chine_seg([k + [0, -0.05, 0.02], k + (a - k) * 0.4, a + [0, 0.1, 0]], [(0.17, 0.24, 0.24), (0.2, 0.28, 0.32), (0.1, 0.12, 0.12)],
                         "armor", up=F, k=0.55, kb=0.35),
          mech.chine_stack([(0.0, a[0], a[2] + 0.12, 0.2, 0.72, 0.42, {"k": 0.25, "kb": 0.7}),
                            (a[1] + 0.02, a[0], a[2], 0.1, 0.24, 0.16, {"k": 0.3, "kb": 0.6})], "armor2"))


def fortress(m, dx):
    o = np.array([dx, 0, 0])
    h, k, a = HIP + o, KNEE + o, ANKLE + o
    m.add("fortress", axle(h, 0.3, 0.36, n=10), axle(k, 0.27, 0.32, n=10),
          mech.cast_seg([h + [0, 0.05, 0], (h + k) / 2 + [0, 0, 0.04], k + [0, 0.22, 0]], [(0.38, 0.42), (0.42, 0.46), (0.32, 0.36)], "armor", n=10),
          mech.cast_seg([k + [0, -0.22, 0.2], k + [0, 0.26, 0.4]], [(0.34, 0.24), (0.24, 0.16)], "armor2", n=10),
          mech.cast_seg([k + [0, -0.05, 0], k + (a - k) * 0.45, a + [0, 0.2, 0]], [(0.4, 0.46), (0.44, 0.5), (0.34, 0.38)], "armor", n=10),
          mech.cast([(0.0, a[0], a[2] + 0.12, 0.5, 0.86, 0.6), (a[1] + 0.1, a[0], a[2], 0.3, 0.4, 0.34)], "armor2", n=10, axis="y"),
          mech.plate(k + (a - k) * 0.25 + [0, 0, 0.5], (1, 0, 0), (0, 1, -0.12), 0.34, 0.24, 0.07, "armor2"))


def build():
    m = mk.Model("concept_legs")
    light(m, -3.0)
    skeletal(m, -1.0)
    faceted(m, 1.0)
    fortress(m, 3.2)
    for g in m.order:
        print(g, mech.proportions(m, [g]))
    return m
