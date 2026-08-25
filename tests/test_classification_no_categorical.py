"""Classification must report the real problem when no categorical feature exists.

`classifier_options_widget` filters `categorical_cols` down to columns that are
present in the data, have more than one unique value, and aren't the FOV/identifier
column. When that filtered set is empty, the "Classify by" multiselect is never
rendered at all -- yet the widget used to return "Please select at least one
category for classification.", telling the user to select something that is not
offered. The true cause is that the dataset has no usable categorical feature, and
the message should say so.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _no_categorical_script():
    import pandas as pd
    import streamlit as st
    from src.widgets.classification_widgets import classifier_options_widget

    df = pd.DataFrame({
        "image_name": ["fov1", "fov2", "fov3", "fov4"],
        "f1": [1.0, 2.0, 3.0, 4.0],
        "f2": [4.0, 3.0, 2.0, 1.0],
    })
    error_msg, *_ = classifier_options_widget(
        df,
        categorical_cols=[],  # dataset has NO categorical feature
        fov_name_col="image_name",
        selected_features=["f1", "f2"],
        classifier="Random Forest",
        splits=0.7,
    )
    if error_msg:
        st.error(error_msg)


def _has_categorical_script():
    import pandas as pd
    import streamlit as st
    from src.widgets.classification_widgets import classifier_options_widget

    df = pd.DataFrame({
        "image_name": ["fov1", "fov2", "fov3", "fov4"],
        "treatment": ["ctrl", "drug", "ctrl", "drug"],
        "f1": [1.0, 2.0, 3.0, 4.0],
        "f2": [4.0, 3.0, 2.0, 1.0],
    })
    error_msg, *_ = classifier_options_widget(
        df,
        categorical_cols=["treatment"],
        fov_name_col="image_name",
        selected_features=["f1", "f2"],
        classifier="Random Forest",
        splits=0.7,
    )
    if error_msg:
        st.error(error_msg)


def test_no_categorical_feature_reports_true_cause():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_function(_no_categorical_script).run(timeout=60)

    assert not at.exception, f"widget raised: {[e.value for e in at.exception]}"
    errors = " ".join(e.value.lower() for e in at.error)
    # The message must name the real problem (no categorical feature)...
    assert "categorical" in errors, (
        f"expected a no-categorical-feature message, got: {[e.value for e in at.error]}"
    )
    # ...and must NOT tell the user to select something that is never offered.
    assert "select at least one category" not in errors, (
        f"misleading 'please select' message shown when there is nothing to select: "
        f"{[e.value for e in at.error]}"
    )


def test_available_categorical_feature_does_not_trigger_no_feature_error():
    """A valid categorical column must not fire the no-categorical-feature error."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_function(_has_categorical_script).run(timeout=60)

    assert not at.exception, f"widget raised: {[e.value for e in at.exception]}"
    errors = " ".join(e.value.lower() for e in at.error)
    assert "no categorical feature" not in errors, (
        f"false positive: no-categorical-feature error shown despite a valid "
        f"categorical column: {[e.value for e in at.error]}"
    )
