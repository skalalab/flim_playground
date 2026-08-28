"""Phasor parity on the real inhibitors dataset.

Checks: k-means cluster label per cell, convex-hull polygons, centroids, the
harmonic-scaled lifetime markers, the frequency annotation, and the title.

Run:  uv run python tests/parity/parity_phasor.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
import numpy as np
import pandas as pd
from harness_common import (
    EXAMPLES,
    WORK_ROOT,
    Results,
    base_state,
    enable_derived,
    load_app_df,
    run_export,
)
from harness_widgets import patch_streamlit

CATS = ["cell_line", "treatment", "dish", "image_name"]
CSV = EXAMPLES / "inhibitors.csv"
WORK = WORK_ROOT / "phasor"
CHANNEL = "nadh"

R = Results()


def poly_key(x, y):
    """Order-insensitive fingerprint of a closed polygon's vertices."""
    pts = np.column_stack([np.asarray(x, float), np.asarray(y, float)])
    pts = np.unique(np.round(pts, 9), axis=0)
    return tuple(map(tuple, pts[np.lexsort((pts[:, 1], pts[:, 0]))]))


def app_phasor(df, harmonic, color_by, k_means, n_clusters):
    # k-means is a checkbox rendered inside phasor_plot, so force it from here
    patch_streamlit({"Perform K-Means clustering": k_means,
                     "Number of clusters": n_clusters})
    from src.vis.bivar import phasor_plot

    return phasor_plot(df.copy(), "cell_id", "image_name", CHANNEL,
                       color_by=color_by, colormap="tab10", f=0.08, harmonic=harmonic)


def run_case(app_df, harmonic, k_means, k):
    print(f"\n-- harmonic={harmonic} kmeans={k_means}({k}) --")
    fig, app_out = app_phasor(app_df, harmonic, ["treatment"], k_means, k)

    state = base_state(
        "Phasor Plot", "inhibitors.csv", CATS,
        color_by=["treatment"],
        analysis_columns=list(app_df.columns),
        method_params={
            "selected_channel": CHANNEL,
            "phasor_harmonic": harmonic,
            "phasor_f": 0.08,
            "k_means": k_means,
            "k_means_clusters": k,
        },
    )
    wd = WORK / f"h{harmonic}_k{int(k_means)}"
    ns, _ = run_export(state, CSV, wd, transform=enable_derived if k_means else None)
    ax = ns["ax"]

    # --- lifetime markers (must be scaled by the harmonic) ---
    app_marks = next(t for t in fig.data if t.name == "Lifetime Markers")
    exp_marks = np.array([(ln.get_xdata()[0], ln.get_ydata()[0]) for ln in ax.lines
                          if ln.get_marker() == "o" and len(ln.get_xdata()) == 1])
    app_pts = np.column_stack([app_marks.x, app_marks.y])
    same = (exp_marks.shape == app_pts.shape
            and np.allclose(np.sort(exp_marks, axis=0), np.sort(app_pts, axis=0)))
    R.check(f"lifetime markers ({len(app_pts)} pts) match", same,
            "" if same else f"app={app_pts[:3]} exp={exp_marks[:3]}")

    # the n-th harmonic phasor is evaluated at n*omega, so the 1 ns marker moves
    wt = 2 * np.pi * 0.08 * harmonic * 1.0
    expect_1ns = (1 / (1 + wt**2), wt / (1 + wt**2))
    R.check(f"1 ns marker at n*omega ({expect_1ns[0]:.4f}, {expect_1ns[1]:.4f})",
            np.any(np.all(np.isclose(app_pts, expect_1ns), axis=1)))

    # --- frequency annotation (Plotly uses <br>, Matplotlib \n) ---
    app_freq = [a.text for a in fig.layout.annotations if a.text.startswith("f =")]
    exp_freq = [t.get_text() for t in ax.texts if t.get_text().startswith("f =")]
    same = (len(app_freq) == 1 and len(exp_freq) == 1
            and app_freq[0].replace("<br>", "\n") == exp_freq[0])
    R.check("frequency annotation text", same, f"app={app_freq} exp={exp_freq}")

    R.check("title", fig.layout.title.text == ax.get_title(),
            f"app={fig.layout.title.text!r} exp={ax.get_title()!r}")

    if not k_means:
        return

    # --- per-cell cluster assignment ---
    exp_df = pd.read_csv(wd / "kmeans_clustered_data.csv")
    a = app_out.set_index("cell_id")["k_means_cluster"]
    e = exp_df.set_index("cell_id")["k_means_cluster"]
    common = a.index.intersection(e.index)
    same = len(common) == len(a) == len(e) and (a.loc[common] == e.loc[common]).all()
    R.check(f"k-means labels for all {len(common)} cells", same,
            "" if same else f"{(a.loc[common] != e.loc[common]).sum()} differ")

    # --- hull polygons (exclude the black universal semicircle, same linewidth) ---
    app_hulls = {poly_key(t.x, t.y) for t in fig.data
                 if t.name and t.name.endswith("boundary")}
    exp_hulls = {poly_key(ln.get_xdata(), ln.get_ydata()) for ln in ax.lines
                 if ln.get_linestyle() == "-" and ln.get_marker() in ("", "None", None)
                 and len(ln.get_xdata()) > 2 and ln.get_linewidth() == 1.5
                 and matplotlib.colors.to_hex(ln.get_color()) != "#000000"}
    R.check(f"convex-hull polygons ({len(app_hulls)} app / {len(exp_hulls)} export)",
            app_hulls == exp_hulls and len(app_hulls) > 0)

    # --- centroids ---
    app_cent = np.vstack([np.column_stack([t.x, t.y]) for t in fig.data
                          if t.name == "Centroids"])
    exp_cent = np.array([(ln.get_xdata()[0], ln.get_ydata()[0]) for ln in ax.lines
                         if ln.get_marker() == "x"])
    R.check(f"k-means centroids ({len(app_cent)})",
            app_cent.shape == exp_cent.shape
            and np.allclose(np.sort(app_cent, axis=0), np.sort(exp_cent, axis=0)))


def main():
    app_df, _ = load_app_df(CSV, CATS, "cell_id", "image_name")
    print(f"\n=== Phasor parity — inhibitors.csv ({len(app_df)} cells) ===")
    for harmonic in (1, 2):
        for k_means, k in ((True, 3), (False, 0)):
            run_case(app_df, harmonic, k_means, k)
    return 0 if R.summary("Phasor") else 1


if __name__ == "__main__":
    sys.exit(main())
