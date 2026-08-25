"""Capture what an exported Matplotlib figure actually draws, per point.

scatter_with_encodings is extracted into all seven method builders, so any change to
it must leave every point's position, colour and alpha untouched — only the grouping
of the draw calls may differ.
"""
import json, pathlib, sys
import numpy as np, pandas as pd
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Not read from sys.argv at import time: pytest owns argv when this is imported as a
# module, and would be taken for an output directory.
SCRATCH = str(pathlib.Path(__file__).resolve().parent)

def make_csv():
    rng = np.random.default_rng(5); rows = []
    for g in ["E1", "E2"]:
        for sh in ["r", "s"]:
            for op in ["hi", "lo"]:
                for d in ["P1", "P2"]:
                    for _ in range(12):
                        rows.append({"cell_id": len(rows), "image_name": f"f{len(rows)%3}",
                                     "experiment": g, "sh_col": sh, "op_col": op,
                                     "patient_id": d,
                                     "X1": rng.normal(0, 1), "X2": rng.normal(0, 1),
                                     "redox_ratio": rng.normal(1.5, 0.22)})
    pd.DataFrame(rows).to_csv(f"{SCRATCH}/data_cap.csv", index=False)

CASES = {
    "fc_shape_opacity": ("Feature Comparison", {"shape_by": "sh_col", "opacity_by": "op_col"}, {}),
    "fc_subcolor_op":      ("Feature Comparison", {"opacity_by": "op_col", "subcolor_by": "patient_id"}, {}),
    "fc_plain":         ("Feature Comparison", {}, {}),
}

def render(name):
    from src.export_script import generate_script
    method, chans, extra = CASES[name]
    state = {"csv_filename": "data_cap.csv", "unique_row_id_col": "cell_id",
             "fov_name_col": "image_name", "method": method,
             "categorical_filters": {}, "numerical_filters": {}, "color_by": ["experiment"],
             "opacity_by": None, "shape_by": None, "separate_by": None, "subcolor_by": None,
             "categorical_cols": ["experiment", "sh_col", "op_col", "patient_id"],
             "analysis_columns": None, "point_size": 8, "axis_label_size": 14,
             "legend_size": 9, "colormap": "tab10", "show_group_counts": False,
             "custom_order": None,
             "method_params": {"selected_var": "redox_ratio", "log_y": False, "log_x": False,
                               "effect_size_method": "None", "mean_or_median": "Mean",
                               "statistical_test": "None", **extra}}
    state.update(chans)
    path = f"{SCRATCH}/exp_{name}.py"
    open(path, "w").write(generate_script(state))
    plt.close("all")
    import runpy
    cwd = __import__("os").getcwd()
    __import__("os").chdir(SCRATCH)
    # Some method scripts close their figure after saving; keep it alive so its
    # collections can be inspected.
    real_close, real_show = plt.close, plt.show
    plt.close = lambda *a, **k: None
    plt.show = lambda *a, **k: None
    try:
        runpy.run_path(path)
    finally:
        plt.close, plt.show = real_close, real_show
        __import__("os").chdir(cwd)
    # Take the axes carrying the most point collections across every open figure: some
    # method scripts build several figures, and gcf() is not reliably the plotted one.
    axes = [a for num in plt.get_fignums() for a in plt.figure(num).axes]
    if not axes:
        raise RuntimeError(f"{name}: the exported script left no axes to inspect")
    ax = max(axes, key=lambda a: sum(c.get_offsets().shape[0] for c in a.collections))
    drawn, calls = [], 0
    for coll in ax.collections:
        off = coll.get_offsets()
        if off.shape[0] == 0:
            continue
        calls += 1
        fc = coll.get_facecolors()
        for i, (px, py) in enumerate(np.asarray(off)):
            rgba = fc[i % len(fc)] if len(fc) else [0, 0, 0, 0]
            drawn.append([round(float(px), 6), round(float(py), 6)]
                         + [round(float(v), 4) for v in rgba])
    drawn.sort()
    return {"calls": calls, "n": len(drawn), "points": drawn}

def direct_scatter():
    """scatter_with_encodings on its own, since all seven builders extract it."""
    from src.export_script import scatter_with_encodings
    rng = np.random.default_rng(2)
    n = 120
    x, y = rng.normal(0, 1, n), rng.normal(0, 1, n)
    shapes = np.array(["r", "s", "d"])[rng.integers(0, 3, n)]
    ops = np.array(["hi", "lo"])[rng.integers(0, 2, n)]
    shape_map = {"r": "o", "s": "s", "d": "D"}
    opacity_map = {"hi": 1.0, "lo": 0.3}
    out = {}
    for label, kw in {
        "both":    dict(shape_vals=shapes, shape_map=shape_map, opacity_vals=ops, opacity_map=opacity_map),
        "opacity": dict(opacity_vals=ops, opacity_map=opacity_map),
        "shape":   dict(shape_vals=shapes, shape_map=shape_map),
        "neither": {},
    }.items():
        plt.close("all")
        fig, ax = plt.subplots()
        scatter_with_encodings(ax, x, y, (0.1, 0.4, 0.7), "L", 8, **kw)
        drawn, calls, labels = [], 0, []
        for coll in ax.collections:
            off = np.asarray(coll.get_offsets())
            if off.shape[0] == 0:
                continue
            calls += 1
            labels.append(coll.get_label())
            fc = coll.get_facecolors()
            paths = coll.get_paths()
            for i, (px, py) in enumerate(off):
                rgba = fc[i % len(fc)]
                # marker identity via its path vertex count, so a shape change is caught
                pv = len(paths[i % len(paths)].vertices)
                drawn.append([round(float(px), 6), round(float(py), 6)]
                             + [round(float(v), 4) for v in rgba] + [pv])
        drawn.sort()
        out[label] = {"calls": calls, "n": len(drawn), "points": drawn,
                      "labels": [l for l in labels if not l.startswith("_")]}
    return out

def snapshot():
    make_csv()
    snap = {name: render(name) for name in CASES}
    snap["_direct"] = direct_scatter()
    return snap

if __name__ == "__main__":
    if len(sys.argv) > 1:
        SCRATCH = sys.argv[1]
    out = f"{SCRATCH}/{sys.argv[2] if len(sys.argv) > 2 else 'export_baseline.json'}"
    json.dump(snapshot(), open(out, "w"))
    snap = json.load(open(out))
    for k, v in snap.items():
        if k == "_direct":
            for kk, vv in v.items():
                print(f"  direct/{kk:11s} scatter_calls={vv['calls']:3d}  points={vv['n']:4d}  labels={vv['labels']}")
        else:
            print(f"  {k:18s} scatter_calls={v['calls']:3d}  points={v['n']:4d}")
