"""Minimal MCP server (stdio, JSON-RPC 2.0) exposing meshkit to AI clients - no dependencies.

    python -m meshkit.mcp_server [--workspace DIR]

Claude Desktop / Claude Code config example:
    {"mcpServers": {"meshkit": {"command": "python", "args": ["-m", "meshkit.mcp_server", "--workspace", "/path/to/work"],
                                "env": {"PYTHONPATH": "/path/to/meshkit"}}}}

Tools
  guide        the modelling API reference and workflow (read this first)
  run_script   save Python code as <workspace>/<name>.py, run it in a subprocess (it must define
               build() -> Model), check + export OBJ/GLB + preview sheet; returns report and image
  render       preview sheet of an existing OBJ / GLB (views, display mode, wireframe, part filter)
  check        closed-shell / contact report of an OBJ / GLB
  info         sizes, parts, triangle counts of an OBJ / GLB
  convert      OBJ <-> GLB
  list_files   files in the workspace
"""
import base64
import json
import os
import subprocess
import sys
import traceback

PROTO = "2025-06-18"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
WS = os.path.abspath(os.environ.get("MESHKIT_WORKSPACE", os.path.join(os.getcwd(), "meshkit_work")))

VIEW_DOC = "views: comma list of iso, iso_rear, iso_below, iso_left, iso_right, front, back, left, right, top, bottom"
MODE_DOC = ("mode: texture (default; falls back to parts), parts (colour per part), solid (clay), tags (colour per "
            "face tag), orientation (no culling, back faces red = holes / flipped faces), xray (no culling)")

TOOLS = [
    {"name": "guide", "description": "Return the meshkit modelling guide (API, conventions, workflow, rules). Read before writing scripts.",
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "run_script",
     "description": "Save `code` as <workspace>/<name>.py and run it. The script must `import meshkit as mk` and define "
                    "build() returning a meshkit Model. Result: JSON report (size, tris, mesh issues, floating parts, "
                    "output paths) and a preview image. " + VIEW_DOC + ". " + MODE_DOC,
     "inputSchema": {"type": "object", "properties": {
         "name": {"type": "string", "description": "file stem, e.g. 'tank_v1'"},
         "code": {"type": "string", "description": "Python source; omit to re-run the saved file"},
         "views": {"type": "string"}, "mode": {"type": "string"}, "wire": {"type": "boolean"},
         "size": {"type": "string", "description": "per-view size WxH, default 520x340"},
         "formats": {"type": "string", "description": "default obj,glb"},
         "contacts": {"type": "boolean", "description": "run the floating-part test (default true)"},
         "timeout": {"type": "number", "description": "seconds, default 300"}},
         "required": ["name"]}},
    {"name": "render", "description": "Render a preview sheet of an OBJ/GLB file (path absolute or relative to the workspace). "
                                      + VIEW_DOC + ". " + MODE_DOC,
     "inputSchema": {"type": "object", "properties": {
         "file": {"type": "string"}, "views": {"type": "string"}, "mode": {"type": "string"},
         "wire": {"type": "boolean"}, "size": {"type": "string"}, "texture": {"type": "string"},
         "parts": {"type": "string", "description": "only these parts (comma list)"},
         "skip": {"type": "string", "description": "hide these parts (comma list)"},
         "zoom": {"type": "number"}, "axes": {"type": "string", "description": "OBJ convention: native | blender"}},
         "required": ["file"]}},
    {"name": "check", "description": "Closed-shell and contact report of an OBJ/GLB (holes, winding, degenerate/non-planar/"
                                     "non-convex faces, floating parts).",
     "inputSchema": {"type": "object", "properties": {"file": {"type": "string"}, "contacts": {"type": "boolean"},
                                                       "axes": {"type": "string"}}, "required": ["file"]}},
    {"name": "info", "description": "Size, parts, triangle counts, pivots and reference points of an OBJ/GLB.",
     "inputSchema": {"type": "object", "properties": {"file": {"type": "string"}, "axes": {"type": "string"}},
                     "required": ["file"]}},
    {"name": "convert", "description": "Convert between OBJ and GLB.",
     "inputSchema": {"type": "object", "properties": {"src": {"type": "string"}, "dst": {"type": "string"},
                                                       "texture": {"type": "string"}}, "required": ["src", "dst"]}},
    {"name": "list_files", "description": "List files in the workspace (or a sub folder).",
     "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}}}},
]


def _p(path):
    return path if os.path.isabs(path) else os.path.join(WS, path)


def _cli(args, timeout=300, cwd=None):
    env = dict(os.environ)
    env["PYTHONPATH"] = ROOT + os.pathsep + env.get("PYTHONPATH", "")
    r = subprocess.run([sys.executable, "-m", "meshkit"] + args, capture_output=True, text=True, timeout=timeout,
                       cwd=cwd or WS, env=env)
    return r.returncode, r.stdout, r.stderr


def _img(path):
    with open(path, "rb") as f:
        return {"type": "image", "data": base64.b64encode(f.read()).decode(), "mimeType": "image/png"}


def _text(s):
    return {"type": "text", "text": s}


def call(name, a):
    os.makedirs(WS, exist_ok=True)
    if name == "guide":
        with open(os.path.join(HERE, "AI_GUIDE.md"), encoding="utf-8") as f:
            return [_text(f.read())], False
    if name == "list_files":
        base = _p(a.get("path", ""))
        out = []
        for d, _, fs in os.walk(base):
            for f in sorted(fs):
                p = os.path.join(d, f)
                out.append(f"{os.path.relpath(p, WS)}\t{os.path.getsize(p)}")
        return [_text("\n".join(out) or "(empty)")], False
    if name == "run_script":
        stem = os.path.splitext(os.path.basename(a["name"]))[0]
        path = os.path.join(WS, stem + ".py")
        if a.get("code") is not None:
            with open(path, "w", encoding="utf-8") as f:
                f.write(a["code"])
        out_dir = os.path.join(WS, "out")
        args = ["run", path, "-o", out_dir, "--name", stem, "--formats", a.get("formats", "obj,glb"),
                "--mode", a.get("mode", "texture"), "--size", a.get("size", "520x340")]
        if a.get("views"):
            args += ["--views", a["views"]]
        if a.get("wire"):
            args.append("--wire")
        if a.get("contacts") is False:
            args.append("--no-contacts")
        code, so, se = _cli(args, timeout=a.get("timeout", 300))
        if code != 0:
            return [_text(f"script failed (exit {code})\n--- stdout ---\n{so[-6000:]}\n--- stderr ---\n{se[-8000:]}")], True
        # the report is the last JSON object on stdout (the script may print before it)
        i = so.rfind("\n{"); rep_txt = so[i + 1:] if i >= 0 else so
        pre = so[:i] if i > 0 else ""
        content = [_text((("script output:\n" + pre[-3000:] + "\n\n") if pre.strip() else "") + rep_txt)]
        prev = os.path.join(out_dir, stem + "_preview.png")
        if os.path.exists(prev):
            content.append(_img(prev))
        return content, False
    if name == "render":
        out = os.path.join(WS, "out", "render_" + os.path.splitext(os.path.basename(a["file"]))[0] + ".png")
        args = ["render", _p(a["file"]), "-o", out, "--mode", a.get("mode", "texture"), "--size", a.get("size", "520x340"),
                "--axes", a.get("axes", "native"), "--zoom", str(a.get("zoom", 1.0))]
        for k in ("views", "texture", "parts", "skip"):
            if a.get(k):
                args += ["--" + k, a[k] if k != "texture" else _p(a[k])]
        if a.get("wire"):
            args.append("--wire")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        code, so, se = _cli(args)
        if code != 0:
            return [_text(se[-8000:] or so)], True
        return [_text(out), _img(out)], False
    if name in ("check", "info"):
        args = [name, _p(a["file"]), "--axes", a.get("axes", "native")]
        if name == "check" and a.get("contacts") is False:
            args.append("--no-contacts")
        code, so, se = _cli(args)
        if not so.strip():
            return [_text(se[-8000:])], True
        return [_text(so)], False
    if name == "convert":
        args = ["convert", _p(a["src"]), _p(a["dst"])]
        if a.get("texture"):
            args += ["--texture", _p(a["texture"])]
        code, so, se = _cli(args)
        return [_text(so if code == 0 else se[-8000:])], code != 0
    return [_text(f"unknown tool {name}")], True


def handle(msg):
    mid = msg.get("id"); meth = msg.get("method"); params = msg.get("params") or {}
    if meth == "initialize":
        return {"jsonrpc": "2.0", "id": mid, "result": {
            "protocolVersion": params.get("protocolVersion", PROTO), "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": "meshkit", "version": "0.1.0"},
            "instructions": "Blender-free modelling: call `guide` first, then iterate with run_script (code -> report + "
                            "preview image). Workspace: " + WS}}
    if meth and meth.startswith("notifications/"):
        return None
    if meth == "ping":
        return {"jsonrpc": "2.0", "id": mid, "result": {}}
    if meth == "tools/list":
        return {"jsonrpc": "2.0", "id": mid, "result": {"tools": TOOLS}}
    if meth == "tools/call":
        try:
            content, err = call(params["name"], params.get("arguments") or {})
        except subprocess.TimeoutExpired:
            content, err = [_text("timed out")], True
        except Exception:
            content, err = [_text(traceback.format_exc()[-6000:])], True
        return {"jsonrpc": "2.0", "id": mid, "result": {"content": content, "isError": err}}
    if mid is None:
        return None
    return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"method not found: {meth}"}}


def main():
    global WS
    if "--workspace" in sys.argv:
        WS = os.path.abspath(sys.argv[sys.argv.index("--workspace") + 1])
    os.makedirs(WS, exist_ok=True)
    stdin = sys.stdin.buffer; stdout = sys.stdout.buffer
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        msgs = msg if isinstance(msg, list) else [msg]
        outs = [r for r in (handle(m) for m in msgs) if r is not None]
        for r in outs:
            stdout.write((json.dumps(r, ensure_ascii=False) + "\n").encode("utf-8")); stdout.flush()


if __name__ == "__main__":
    main()
