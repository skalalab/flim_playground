"""Phasor parity on the real inhibitors dataset.

Checks: harmonic-scaled lifetime markers, the frequency annotation, and the title.

Run:  uv run python tests/parity/parity_phasor.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from harness_common import (
    EXAMPLES,
    WORK_ROOT,
    Results,
    base_state,
    load_app_df,
    run_export,
)
from harness_widgets import patch_streamlit

CATS = ["cell_line", "treatment", "dish", "image_name"]
CSV = EXAMPLES / "inhibitors.csv"
WORK = WORK_ROOT / "phasor"
CHANNEL = "nadh"

R = Results()


def app_phasor(df, harmonic, color_by):
    patch_streamlit()
    from src.vis.bivar import phasor_plot

    return phasor_plot(df.copy(), "cell_id", "image_name", CHANNEL,
                       color_by=color_by, colormap="tab10", f=0.08, harmonic=harmonic)


def run_case(app_df, harmonic):
    print(f"\n-- harmonic={harmonic} --")
    fig, _ = app_phasor(app_df, harmonic, ["treatment"])

    state = base_state(
        "Phasor Plot", "inhibitors.csv", CATS,
        color_by=["treatment"],
        analysis_columns=list(app_df.columns),
        method_params={
            "selected_channel": CHANNEL,
            "phasor_harmonic": harmonic,
            "phasor_f": 0.08,
        },
    )
    wd = WORK / f"h{harmonic}"
    ns, _ = run_export(state, CSV, wd)
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


def main():
    app_df, _ = load_app_df(CSV, CATS, "cell_id", "image_name")
    print(f"\n=== Phasor parity — inhibitors.csv ({len(app_df)} cells) ===")
    for harmonic in (1, 2):
        run_case(app_df, harmonic)
    return 0 if R.summary("Phasor") else 1


if __name__ == "__main__":
    sys.exit(main())
