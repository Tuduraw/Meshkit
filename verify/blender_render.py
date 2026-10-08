"""Render the verification models in Blender with the SAME cameras as the meshkit reference images
(optional cross-check with an independent renderer; Blender is used only to look, never to model).

    blender -b -P verify/blender_render.py -- --out verify/result/blender
    python verify/verify.py --blender verify/result/blender

or, inside a running Blender, open this file in the Text Editor and run it (the output goes to verify/result/blender;
a temporary scene is used and removed again, so the open file is not changed).

Every model is imported from verify/models/<name>.glb (glTF importer, Y-up -> Blender Z-up) and rendered
with Workbench (flat light, texture colour, transparent background) for the texture views listed in
verify/reference/reference.json.  verify.py then compares the outlines with the meshkit images (IoU)."""
import json
import math
import os
import sys

import bpy
from mathutils import Matrix, Vector

HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.path.join(os.getcwd(), "verify")
VIEWS = ("iso", "iso_below", "left", "top", "front")       # iso_rear is posed in meshkit (moved parts) - skipped


def to_blender(p):
    """meshkit native (+X left, +Y up, +Z forward) == glTF axes -> Blender (Z up): (x, y, z) -> (x, -z, y)."""
    return Vector((p[0], -p[2], p[1]))


def setup_scene(W, H):
    sc = bpy.context.scene
    sc.render.engine = "BLENDER_WORKBENCH"
    sc.display.shading.light = "FLAT"
    sc.display.shading.color_type = "TEXTURE"
    sc.render.film_transparent = True
    sc.render.resolution_x, sc.render.resolution_y, sc.render.resolution_percentage = W, H, 100
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGBA"
    sc.view_settings.view_transform = "Standard"
    try:
        sc.display.render_aa = "OFF"           # crisp outlines like the meshkit renderer
    except Exception:                          # noqa: BLE001
        pass
    return sc


def camera(sc, cam_info, W, H):
    eye = to_blender(cam_info["eye"]); target = to_blender(cam_info["target"]); up = to_blender(cam_info.get("up", (0, 1, 0)))
    f = (target - eye).normalized(); r = f.cross(up).normalized(); u = r.cross(f)
    M = Matrix(((r.x, u.x, -f.x, eye.x), (r.y, u.y, -f.y, eye.y), (r.z, u.z, -f.z, eye.z), (0, 0, 0, 1)))
    cd = bpy.data.cameras.new("cam"); ob = bpy.data.objects.new("cam", cd)
    sc.collection.objects.link(ob); ob.matrix_world = M
    cd.clip_start, cd.clip_end = 0.01, 5000.0
    if cam_info.get("ortho"):
        cd.type = "ORTHO"; cd.sensor_fit = "AUTO"
        cd.ortho_scale = max(W, H) / cam_info["ortho"]
    else:
        cd.type = "PERSP"; cd.sensor_fit = "VERTICAL"
        cd.angle_y = math.radians(cam_info.get("fov", 32.0))
    sc.camera = ob
    return ob


DATA_KINDS = ("objects", "meshes", "materials", "images", "cameras", "textures", "node_groups")


def main(out, names=None):
    """Background (blender -b): a fresh empty session per model.  Inside a running Blender: a temporary scene,
    and everything this script imported is removed again afterwards - the open file is left as it was."""
    with open(os.path.join(HERE, "reference", "reference.json"), encoding="utf-8") as fh:
        ref = json.load(fh)
    W, H = ref["size"]
    os.makedirs(out, exist_ok=True)
    # never reset a session someone is working in: the empty-session path is only for blender -b
    win = None if bpy.app.background else (bpy.context.window or bpy.context.window_manager.windows[0])
    written = []
    for name, info in ref["models"].items():
        if names and name not in names:
            continue
        if win is None:
            bpy.ops.wm.read_factory_settings(use_empty=True)
            sc = setup_scene(W, H)
        else:
            before = {k: set(getattr(bpy.data, k)) for k in DATA_KINDS}
            prev = win.scene
            sc = bpy.data.scenes.new("meshkit_verify")
            win.scene = sc
            setup_scene(W, H)
        try:
            bpy.ops.import_scene.gltf(filepath=os.path.join(HERE, "models", name + ".glb"))
            for fn, rinfo in info["renders"].items():
                if rinfo["mode"] != "texture" or rinfo["view"] not in VIEWS:
                    continue
                cam = camera(sc, rinfo["camera"], W, H)
                sc.render.filepath = os.path.join(out, f"{name}_{rinfo['view']}.png")
                bpy.ops.render.render(write_still=True, scene=sc.name)
                bpy.data.objects.remove(cam, do_unlink=True)
                written.append(sc.render.filepath)
                print("written", sc.render.filepath)
        finally:
            if win is not None:
                win.scene = prev
                for k in DATA_KINDS:
                    coll = getattr(bpy.data, k)
                    for item in [i for i in coll if i not in before[k]]:
                        coll.remove(item)
                bpy.data.scenes.remove(sc)
    return written


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    out = os.path.join(HERE, "result", "blender")
    names = []
    i = 0
    while i < len(argv):
        if argv[i] == "--out":
            out = os.path.abspath(argv[i + 1]); i += 2
        else:
            names.append(argv[i]); i += 1
    main(out, names or None)
