"""Check one consistent subcolor and legend entry per nested value across the figure.

Run: python tests/check_subcolor.py
"""
import os
import pathlib
import re
import runpy
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from src.vis.helpers import create_subcolor_map, natural_tuple_sort  # noqa: E402
from src.vis.univar import feature_comparison_plot  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + ("" if cond else f"   {detail}"))
    if not cond:
        FAILS.append(name)


def frame(spec, var=None):
    rng = np.random.default_rng(3)
    return pd.DataFrame([
        {"cell_id": i, "fov": "f0", "g": g, "d": d,
         "y": float(rng.normal(1.5, 0.2)) if var is None else var}
        for i, (g, d) in enumerate(
            [(g, d) for g, ds in spec for d in ds for _ in range(8)])])


def run(spec, **kw):
    return create_subcolor_map(frame(spec), "d", "g", [g for g, _ in spec], **kw)


# Overlapping on purpose: S1 is in two groups, S2 in two others, the rest in one each.
# Under a global map that distinction stops mattering, which is the point.
mixed = [("ExpA", ["S1", "a1", "a2"]), ("ExpB", ["S1", "S2", "b1"]),
         ("ExpC", ["S2", "c1", "c2", "c3"])]

print("1. the mapping")
colour = run(mixed)
distinct = {v for _g, vs in mixed for v in vs}
check("one colour per distinct value", len(colour) == len(distinct), sorted(colour))
check("no two values share a colour", len(set(colour.values())) == len(distinct),
      len(set(colour.values())))
check("a value in several groups has one colour", colour["S1"] and colour["S2"])
check("values sharing a group differ",
      all(len({colour[v] for v in vs}) == len(vs) for _g, vs in mixed))
# Map keys determine trace and legend order, so require natural sorting.
check("keys are the figure's values in natural-sort order",
      list(colour) == natural_tuple_sort(distinct), list(colour))

print("2. engines")
plotly_colour = run(mixed, engine="plotly")
mpl_colour = run(mixed, engine="mpl")
check("plotly returns rgba strings",
      all(isinstance(c, str) and c.startswith("rgba(") for c in plotly_colour.values()))
check("mpl returns plain rgb tuples",
      all(isinstance(c, tuple) and len(c) == 3 for c in mpl_colour.values()),
      list(mpl_colour.values())[:1])
# Compare app/export RGB channels within half a byte to allow Plotly rounding.
check("both engines name the same values", set(plotly_colour) == set(mpl_colour))
_off = []
for value, rgba in plotly_colour.items():
    channels = [int(n) for n in re.findall(r"\d+", rgba)[:3]]
    for got, want in zip(channels, mpl_colour[value]):
        if abs(got - want * 255) > 0.5:
            _off.append((value, got, round(want * 255, 2)))
check("the same colour reaches both engines", not _off, _off[:3])

print("3. off switches")
check("no column is off", create_subcolor_map(frame(mixed), None, "g", ["ExpA"]) is None)
check("absent column is off", create_subcolor_map(frame(mixed), "nope", "g", ["ExpA"]) is None)
check("no groups is off", create_subcolor_map(frame(mixed), "d", "g", []) is None)

print("4. nulls")
null_df = frame(mixed)
null_df.loc[null_df.index[:4], "d"] = np.nan
null_colour = create_subcolor_map(null_df, "d", "g", ["ExpA", "ExpB", "ExpC"])
check("nulls fold to N/A and still get a colour", "N/A" in null_colour, sorted(null_colour))

print("5. the figure")
PLOT = dict(unique_row_id_col="cell_id", fov_name_col="fov", selected_var="y",
            color_by=["g"], colormap="tab10")
fig = feature_comparison_plot(frame(mixed), subcolor_by="d", **PLOT)
shown = [t.name for t in fig.data if t.showlegend]
check("one legend entry per value", sorted(shown) == sorted(distinct), shown)
check("legend entries are bare, never group-qualified",
      not any(":" in name for name in shown), shown)

by_group = {}
for trace in fig.data:
    if trace.legendgroup and str(trace.legendgroup).startswith("subcolor"):
        by_group.setdefault(trace.legendgroup, set()).add(trace.name)
check("every batch of a value shares one legendgroup",
      all(len(v) == 1 for v in by_group.values()), by_group)
check("no two values share a legendgroup", len(by_group) == len(distinct), by_group)

colours_in_fig = {t.name: t.marker.color for t in fig.data if getattr(t, "text", None) is not None}
check("a value keeps one colour across the groups it appears in",
      len({t.marker.color for t in fig.data
           if getattr(t, "text", None) is not None and t.name == "S1"}) == 1)
check("the figure draws as many colours as there are values",
      len(set(colours_in_fig.values())) == len(distinct), len(set(colours_in_fig.values())))

print("6. the exported script")
# Generated and RUN, not read: the template is one large f-string, so a stale name in
# the subcolor branch fails at runtime and nowhere else.


def exported_legend(spec):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.export_script import generate_script

    csv = HERE / "data_subcolor.csv"
    frame(spec).rename(columns={"fov": "image_name"}).to_csv(csv, index=False)
    state = {
        "csv_filename": csv.name, "unique_row_id_col": "cell_id",
        "fov_name_col": "image_name", "method": "Feature Comparison",
        "categorical_filters": {}, "numerical_filters": {},
        "color_by": ["g"], "opacity_by": None, "shape_by": None,
        "separate_by": None, "subcolor_by": "d",
        "categorical_cols": ["g", "d"], "analysis_columns": None,
        "point_size": 8, "axis_label_size": 14, "legend_size": 9,
        "colormap": "tab10", "show_group_counts": False, "custom_order": None,
        "method_params": {"selected_var": "y", "log_y": False, "add_boxplot": False,
                          "effect_size_method": "None", "mean_or_median": "Mean",
                          "statistical_test": "None"},
    }
    script = HERE / "exp_subcolor.py"
    script.write_text(generate_script(state))
    plt.close("all")
    real_close, real_show = plt.close, plt.show
    plt.close = plt.show = lambda *a, **k: None
    cwd = os.getcwd()
    os.chdir(HERE)
    try:
        runpy.run_path(str(script))
    finally:
        plt.close, plt.show = real_close, real_show
        os.chdir(cwd)
    axes = [a for num in plt.get_fignums() for a in plt.figure(num).axes]
    ax = max(axes, key=lambda a: sum(c.get_offsets().shape[0] for c in a.collections))
    legend = ax.get_legend()
    return [t.get_text() for t in legend.get_texts()] if legend else []


exported = exported_legend(mixed)
check("the exported script runs with subcolor on", bool(exported), exported)
check("the exported legend holds the same values as the app's",
      sorted(exported) == sorted(distinct), exported)

print("\n" + ("ALL PASS" if not FAILS else f"{len(FAILS)} FAILING: {FAILS}"))
sys.exit(1 if FAILS else 0)
