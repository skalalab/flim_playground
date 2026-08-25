"""Every statistic in the sina plot is scoped to the COLOUR GROUP, never to the
(colour, shape, opacity) subgroup.

Two things used to be computed per subgroup, and both were wrong for the same reason:
the subgroup is not a thing the reader can see. It has no x position of its own -- it
shares the colour group's -- so a number computed from it is drawn as if it described
the colour group.

  * the KDE that sets the sina jitter width, so switching shape_by on re-estimated
    every density over a quarter of the data and moved every point sideways
  * the boxplot quartiles, so shape_by + opacity_by drew 12 overlapping boxes on 2 x
    positions and the visible one was whichever got painted last

The contract asserted here:

  1. no visual-encoding channel moves any point (shape/opacity/match change appearance,
     never geometry)
  2. one box per (section, colour group), whatever the channels
  3. box statistics equal the pooled colour group's, computed straight from numpy
  4. the exported Matplotlib script draws the same points as the app

Run standalone; exits non-zero on the first failure:

    python tests/check_sina_scope.py
"""
import contextlib
import os
import pathlib
import runpy
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.vis import univar  # noqa: E402
from src.vis.univar import feature_comparison_plot  # noqa: E402

VAR = "redox_ratio"
BASE = dict(cell_id_col="cell_id", fov_name_col="image_name", selected_var=VAR,
            color_by=["experiment"], colormap="tab10")

# Reference plots, and the channels layered on top of each. A channel may change what a
# point LOOKS like; it may not change where it sits.
REFERENCES = {"flat": {}, "sep": {"separate_by": "sep_col"}}
VARIANTS = {
    "shape":          {"shape_by": "sh_col"},
    "opacity":        {"opacity_by": "op_col"},
    "shape+opacity":  {"shape_by": "sh_col", "opacity_by": "op_col"},
    "subcolor":          {"subcolor_by": "patient_id"},
    "subcolor+shape+op": {"subcolor_by": "patient_id", "shape_by": "sh_col",
                       "opacity_by": "op_col"},
}


def build_df():
    """Unbalanced on purpose: subgroup-scoped code only misbehaves visibly when the
    subgroups differ in size, and a perfectly crossed frame hides it."""
    rng = np.random.default_rng(5)
    rows = []
    for group in ["E1", "E2"]:
        for shape in ["r", "s"]:
            for opacity in ["hi", "lo"]:
                for patient in ["P1", "P2", "P3"]:
                    # E2/s/lo gets a third of the rows the others do.
                    reps = 3 if (group, shape, opacity) == ("E2", "s", "lo") else 9
                    for _ in range(reps):
                        rows.append({
                            "cell_id": len(rows),
                            "image_name": f"f{len(rows) % 3}",
                            "experiment": group, "sh_col": shape, "op_col": opacity,
                            "patient_id": patient,
                            "sep_col": ["s1", "s2"][len(rows) % 2],
                            VAR: rng.normal(1.5, 0.22),
                        })
    return pd.DataFrame(rows)


@contextlib.contextmanager
def boxplot_on():
    """Force the "Add boxplot" checkbox. Bare-mode Streamlit hands back the widget's
    default and ignores session_state, so the checkbox is the one thing that has to be
    stubbed; everything asserted below is read off the real figure."""
    real = univar.st.checkbox

    def fake(label, value=False, key=None, **kwargs):
        if key and key.startswith("add_boxplot_"):
            return True
        return real(label, value=value, key=key, **kwargs)

    univar.st.checkbox = fake
    try:
        yield
    finally:
        univar.st.checkbox = real


def point_xy(fig, precision=9):
    """(x, y) per cell id. Point traces are the ones carrying cell ids in ``text``,
    which excludes the boxes, the mean line and the ghost legend entries.

    ``precision=None`` keeps full float64, for the export comparison: rounding here and
    again at the comparison turns a value sitting on a .5 boundary into a false
    difference of one ulp of the coarser precision.
    """
    out = {}
    for trace in fig.data:
        text = getattr(trace, "text", None)
        if text is None or getattr(trace, "x", None) is None or not len(trace.x):
            continue
        for cid, x, y in zip(text, trace.x, trace.y):
            out[str(cid)] = ((float(x), float(y)) if precision is None
                             else (round(float(x), precision), round(float(y), precision)))
    return out


def boxes(fig):
    """One entry per box trace, in draw order: (name, x, q1, median, q3, mean)."""
    return [
        (str(t.name), round(float(t.x[0]), 6), round(float(t.q1[0]), 9),
         round(float(t.median[0]), 9), round(float(t.q3[0]), 9), round(float(t.mean[0]), 9))
        for t in fig.data if t.type == "box"
    ]


def expected_box(values):
    """What univar.py computes, on the pooled group: percentiles plus the mean."""
    return (round(float(np.percentile(values, 25)), 9),
            round(float(np.percentile(values, 50)), 9),
            round(float(np.percentile(values, 75)), 9),
            round(float(np.mean(values)), 9))


failures = []


def check(name, condition, detail=""):
    print(f"  {'ok  ' if condition else 'FAIL'}  {name}{'' if condition else '  ' + detail}")
    if not condition:
        failures.append(name)


def check_points_do_not_move(df):
    print("1. no channel moves a point")
    for ref_name, ref_kw in REFERENCES.items():
        reference = point_xy(feature_comparison_plot(df, **BASE, **ref_kw))
        for var_name, var_kw in VARIANTS.items():
            actual = point_xy(feature_comparison_plot(df, **BASE, **ref_kw, **var_kw))
            moved = [c for c, xy in reference.items() if actual.get(c) != xy]
            check(f"{ref_name} + {var_name}", not moved,
                  f"{len(moved)}/{len(reference)} points moved, e.g. "
                  + ", ".join(f"{c}: {reference[c]} -> {actual.get(c)}" for c in moved[:2]))


def check_one_box_per_colour_group(df):
    print("2. one box per (section, colour group)")
    with boxplot_on():
        for ref_name, ref_kw in REFERENCES.items():
            sections = df["sep_col"].nunique() if "separate_by" in ref_kw else 1
            wanted = df["experiment"].nunique() * sections
            for var_name, var_kw in {"none": {}, **VARIANTS}.items():
                drawn = boxes(feature_comparison_plot(df, **BASE, **ref_kw, **var_kw))
                check(f"{ref_name} + {var_name}", len(drawn) == wanted,
                      f"drew {len(drawn)} boxes, wanted {wanted}")


def check_box_stats_are_pooled(df):
    print("3. box statistics are the pooled colour group's")
    with boxplot_on():
        drawn = boxes(feature_comparison_plot(df, **BASE, shape_by="sh_col",
                                              opacity_by="op_col"))
        for name, _x, *stats in drawn:
            wanted = expected_box(df[df["experiment"] == name][VAR].values)
            check(f"{name}", tuple(stats) == wanted, f"{tuple(stats)} != {wanted}")


def render_export(df, channels):
    """Generate the exported script for these channels, run it, and return its axes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.export_script import generate_script

    csv = HERE / "data_scope.csv"
    df.to_csv(csv, index=False)
    state = {
        "csv_filename": csv.name, "unique_row_id_col": "cell_id",
        "fov_name_col": "image_name", "method": "Feature Comparison",
        "categorical_filters": {}, "numerical_filters": {},
        "color_by": ["experiment"], "opacity_by": None, "shape_by": None,
        "separate_by": None, "subcolor_by": None,
        "categorical_cols": ["experiment", "sh_col", "op_col", "patient_id", "sep_col"],
        "analysis_columns": None, "point_size": 8, "axis_label_size": 14,
        "legend_size": 9, "colormap": "tab10", "show_group_counts": False,
        "custom_order": None,
        "method_params": {"selected_var": VAR, "log_y": False, "add_boxplot": True,
                          "effect_size_method": "None", "mean_or_median": "Mean",
                          "statistical_test": "None"},
    }
    state.update(channels)
    script = HERE / "exp_scope.py"
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
    if not axes:
        raise RuntimeError("the exported script left no axes to inspect")
    return max(axes, key=lambda a: sum(c.get_offsets().shape[0] for c in a.collections))


def check_export_parity(df):
    print("4. exported script draws the app's points")
    # separate_by as well as shape/opacity: the export fits its KDE inside a per-section
    # loop, so a section is the other place the two sides could disagree about scope.
    for label, channels in {
        "shape+opacity": {"shape_by": "sh_col", "opacity_by": "op_col"},
        "separate+opacity": {"separate_by": "sep_col", "opacity_by": "op_col"},
    }.items():
        ax = render_export(df, channels)
        exported = sorted((float(px), float(py))
                          for coll in ax.collections
                          for px, py in np.asarray(coll.get_offsets()))

        with boxplot_on():
            fig = feature_comparison_plot(df, **BASE, **channels)
        in_app = sorted(point_xy(fig, precision=None).values())

        check(f"{label}: point count", len(exported) == len(in_app),
              f"export {len(exported)} vs app {len(in_app)}")
        if len(exported) != len(in_app):
            continue
        # Both sides run the same _density_at_points and the same rng(42) over the same
        # rows, so this is float64-exact rather than merely close; the tolerance only
        # absorbs the round trip through Matplotlib's offset array.
        worst = max(max(abs(a[0] - b[0]), abs(a[1] - b[1]))
                    for a, b in zip(in_app, exported))
        check(f"{label}: point positions", worst < 1e-6, f"largest delta {worst}")
        # The export's box overlay was already per colour group; this pins them together.
        check(f"{label}: box count", len(ax.patches) == len(boxes(fig)),
              f"export {len(ax.patches)} vs app {len(boxes(fig))}")


if __name__ == "__main__":
    frame = build_df()
    check_points_do_not_move(frame)
    check_one_box_per_colour_group(frame)
    check_box_stats_are_pooled(frame)
    check_export_parity(frame)
    print(f"\n{len(failures)} failure(s)" + (": " + ", ".join(failures) if failures else ""))
    sys.exit(1 if failures else 0)
