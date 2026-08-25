"""Capture per-point (x, y) keyed by cell id, for every channel combination.

The sina jitter comes from a KDE fitted per (separate section, colour group) and an
rng reseeded per group; any restructure of the drawing loop must leave every point on
the exact x it had, or the cluster silhouette changes.

Because the fit is scoped to the colour group, the shape/opacity/subcolor cases here must
agree with ``plain`` point for point -- see tests/check_sina_scope.py, which asserts
that directly. This file is the wider net: it pins the actual numbers, so a change to
the KDE, the grid, or the seed is caught even though it would keep the channels
consistent with each other.
"""
import json, sys
import numpy as np, pandas as pd
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from src.vis.univar import feature_comparison_plot

rng = np.random.default_rng(5)
rows = []
for g in ["E1", "E2"]:
    for sh in ["r", "s"]:
        for op in ["hi", "lo"]:
            for d in ["P1", "P2", "P3"]:
                for _ in range(9):
                    rows.append({"cell_id": len(rows), "image_name": f"f{len(rows)%3}",
                                 "experiment": g, "sh_col": sh, "op_col": op,
                                 "patient_id": d, "sep_col": ["s1", "s2"][len(rows) % 2],
                                 "redox_ratio": rng.normal(1.5, 0.22)})
df = pd.DataFrame(rows)
base = dict(cell_id_col="cell_id", fov_name_col="image_name", selected_var="redox_ratio",
            color_by=["experiment"], colormap="tab10")

CASES = {
    "plain":            {},
    "shape":            {"shape_by": "sh_col"},
    "opacity":          {"opacity_by": "op_col"},
    "shape+opacity":    {"shape_by": "sh_col", "opacity_by": "op_col"},
    "subcolor":            {"subcolor_by": "patient_id"},
    "subcolor+shape":      {"subcolor_by": "patient_id", "shape_by": "sh_col"},
    "subcolor+shape+op":   {"subcolor_by": "patient_id", "shape_by": "sh_col", "opacity_by": "op_col"},
    "separate":         {"separate_by": "sep_col", "subcolor_by": "patient_id", "opacity_by": "op_col"},
}

def snapshot():
    out = {}
    for name, kw in CASES.items():
        fig = feature_comparison_plot(df, **base, **kw)
        pts, legend = {}, []
        for t in fig.data:
            x = getattr(t, "x", None)
            if x is None or len(x) == 0 or not isinstance(getattr(t.marker, "color", None), (str, list, tuple, np.ndarray)):
                continue
            txt = getattr(t, "text", None)
            if txt is None:
                continue
            for cid, xv, yv in zip(txt, t.x, t.y):
                pts[str(cid)] = [round(float(xv), 9), round(float(yv), 9)]
            if t.showlegend:
                legend.append(str(t.name))
        out[name] = {"points": pts, "n_points": len(pts), "legend": sorted(legend)}
    return out

if __name__ == "__main__":
    path = sys.argv[1]
    json.dump(snapshot(), open(path, "w"))
    snap = json.load(open(path))
    for k, v in snap.items():
        print(f"  {k:16s} points={v['n_points']:4d}  legend={len(v['legend'])}")
