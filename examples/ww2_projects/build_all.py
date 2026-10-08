"""Build every aircraft in this folder: check, export (OBJ / GLB / texture) and preview sheet into out/,
then test that no propeller blade touches the airframe through a full turn.

    python build_all.py            (with meshkit on PYTHONPATH, or run from inside the meshkit folder tree)
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))      # meshkit/ (the repository root)
from meshkit import cli, check  # noqa: E402

AIRCRAFT = {"j7w1": ["$propeller"], "kikka": [], "j8m1": [], "xb42": ["$prop_front", "$prop_rear"],
            "xf15c": ["$propeller"], "xf5u": ["$propeller_l", "$propeller_r"]}

if __name__ == "__main__":
    summary = {}
    for name, props in AIRCRAFT.items():
        model, rep = cli.run_script(os.path.join(HERE, name + ".py"), os.path.join(HERE, "out"))
        clear = {}
        for p in props:
            others = [q for q in props if q != p and name != "xb42"]
            clear[p] = check.clearance(model, p, model.pivots[p], (0, 0, 1), range(0, 360, 6), tags=("prop", "prop_tip"),
                                       ignore=others)["ok"]
        summary[name] = {"ok": rep["ok"], "tris": rep["summary"]["tris"], "size_xyz": rep["summary"]["size"],
                         "parts": rep["summary"]["parts"], "propeller_clearance": clear}
        print(name, json.dumps(summary[name], ensure_ascii=False))
    with open(os.path.join(HERE, "out", "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1, ensure_ascii=False)
