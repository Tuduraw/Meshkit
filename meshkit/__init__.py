"""meshkit - Blender-independent modelling toolkit for AI agents.

Build closed low-poly meshes from code, check them, texture them, preview them in software and
read/write OBJ and glTF. Needs only numpy and Pillow (opencv optional for silhouettes).
"""
__version__ = "0.5.0"

from .geom import (Model, Part, Shell, V, rot_x, rot_y, rot_z, rot_axis, loft, box, prism, cylinder, lathe,  # noqa: F401
                   ellipsoid, extrude_xz, extrude_zy, extrude_xy, ring_superellipse, ring_on_axis, wing, fin,
                   wing_section, airfoil_2d, blade)
from .io_obj import read_obj, write_obj  # noqa: F401
from .io_gltf import read_gltf, write_glb  # noqa: F401
# v0.3: modern-vehicle kits (jets, MBTs, warships) and texture detail passes
from . import planform, fuselage, gear, armor, ship, detail  # noqa: F401,E402
# v0.4: robots (meshkit.mech) - limb / armour shapes in four design languages, joint rotations, checks
from . import mech  # noqa: F401,E402
# v0.5: see-through check (back faces in view) - lessons of the vehicleaddonww2 aircraft fix
from . import seethrough  # noqa: F401,E402


def load(path, **kw):
    """Read .obj / .glb / .gltf into a Model."""
    p = path.lower()
    if p.endswith(".obj"):
        return read_obj(path, **kw)
    if p.endswith((".glb", ".gltf")):
        return read_gltf(path, **kw)
    raise ValueError(f"unsupported format: {path}")


def save(model, path, **kw):
    """Write .obj / .glb by extension."""
    p = path.lower()
    if p.endswith(".obj"):
        return write_obj(model, path, **kw)
    if p.endswith(".glb"):
        return write_glb(model, path, **kw)
    raise ValueError(f"unsupported format: {path}")
