"""Real-page Histogram preserves individual units through transforms and export."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import dataset_io
from src.widgets import analysis_config_widgets as acw
from src.widgets import visualization_widgets as vw

PAGE = str(Path(__file__).resolve().parents[1] / "pages/data_analysis.py")
FEATURE = "Lifetime fit_ch1: T1"


def frame():
    rows = []
    for day, shift in [("Day 2", 0), ("Day 10", 10), (None, 20)]:
        for treatment, offset in [("ctrl", 0), ("drug", 1)]:
            for dish, mean in [("dish1", 2), ("dish2", 4), ("dish3", 8)]:
                for delta in [-1, 1, np.nan]:
                    rows.append(dict(cell_id=f"id{len(rows)}", day=day, treatment=treatment,
                                     dish=dish, **{FEATURE: mean + delta + shift + offset}))
    return pd.DataFrame(rows)


@pytest.mark.parametrize("gmm", [False, True])
@pytest.mark.parametrize("logged", [False, True])
def test_page_keeps_individual_units_despite_stale_collapse_settings(
    monkeypatch, gmm, logged
):
    from streamlit.testing.v1 import AppTest
    from src.vis import univar
    from src import export_script

    source = frame()
    source["day"] = source["day"].fillna("N/A")  # loader normalization
    monkeypatch.setattr(acw, "get_categorical_cols_analysis", lambda *a, **k: ["treatment", "dish", "day"])
    monkeypatch.setattr(dataset_io, "load_table", lambda *a, **k: (
        source, {"Uncategorized Features": [FEATURE]}, True, ",", "cell_id"))
    observed, captured = [], []
    function_name = "feature_gmm_plot" if gmm else "feature_histogram_plot"
    original = getattr(univar, function_name)

    def record(data, selected_var, color_by, **kwargs):
        observed.append((data.copy(), kwargs.copy()))
        return original(data, selected_var, color_by, **kwargs)

    monkeypatch.setattr(univar, function_name, record)
    original_export = export_script.generate_script
    monkeypatch.setattr(export_script, "generate_script", lambda state: (
        captured.append(state) or original_export(state)))
    at = AppTest.from_file(PAGE)
    at.session_state["_menu_Uncategorized Features"] = FEATURE
    at.session_state[vw.COLOR_BY_KEY] = ["treatment"]
    at.session_state["vis_encoding_histogram_separate_by"] = "day"
    at.session_state["vis_encoding_histogram_collapse_by"] = "dish"
    at.session_state[vw.COLLAPSE_BY_KEY] = "dish"
    at.run(timeout=90)
    at.session_state["analysis_control_apply_gmm"] = gmm
    at.session_state[f"log_x_hist_{FEATURE}"] = logged
    at.radio[1].set_value("Feature Histogram").run(timeout=90)
    assert not at.exception
    assert observed
    data, options = observed[-1]
    assert len(data) == 36  # 3 categories × 2 treatments × 3 dishes × 2 units
    expected = source.dropna(subset=[FEATURE]).copy()
    if logged:
        expected[FEATURE] = np.log10(expected[FEATURE] + 1e-6)
    pd.testing.assert_frame_equal(data.set_index("cell_id").sort_index(),
                                  expected.set_index("cell_id").sort_index())
    assert options["separate_by"] == "day"
    assert not any("replicate means" in c.value for c in at.caption)
    assert not any("skewness:" in item.value or "-skewed" in item.value
                   or "symmetric" in item.value for item in at.markdown)
    assert "Collapse by" not in {widget.label for widget in at.selectbox}
    state = captured[-1]
    assert "collapse_by" not in state["method_params"]
    assert state["method_params"]["log_x"] is logged
    assert state["separate_by"] == "day"
    assert state["shape_by"] is state["opacity_by"] is state["subcolor_by"] is None
    if gmm:
        assert len(at.expander) >= 3
        assert len([e for e in at.expander if e.label.startswith("GMM details")]) == 3
