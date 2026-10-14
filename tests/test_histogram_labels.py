"""histogram_labels: the one source of the histogram's axis labels and title for app and export."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.feature_labels import format_feature_label
from src.vis.histogram import histogram_labels


def _prepared(**overrides):
    base = {"selected_var": "F1", "apply_gmm": False, "color_by": ["treatment"], "separate_by": None}
    return dict(base, **overrides)


def test_histogram_labels_follow_scale_mode_and_separation():
    pretty = format_feature_label("F1", engine="mpl")
    assert histogram_labels(_prepared(), log_x=False, engine="mpl") == {
        "x": pretty, "y": "Count", "title": f"Frequency histogram of {pretty} by treatment"}
    assert histogram_labels(_prepared(apply_gmm=True, separate_by="day"), log_x=True, engine="mpl") == {
        "x": f"log₁₀({pretty})", "y": "Density", "title": f"Gaussian Mixture Model fit of {pretty} by treatment"}
    assert histogram_labels(_prepared(apply_gmm=True, color_by=[]), log_x=False, engine="mpl")["y"] == "Probability Density"
    assert histogram_labels(_prepared(color_by=[]), log_x=False, engine="mpl")["title"] == f"Frequency histogram of {pretty}"


def test_histogram_labels_escape_colour_names_for_plotly_only():
    prepared = _prepared(color_by=["a<b"])
    assert histogram_labels(prepared, log_x=False, engine="plotly")["title"].endswith(" by a&lt;b")
    assert histogram_labels(prepared, log_x=False, engine="mpl")["title"].endswith(" by a<b")
