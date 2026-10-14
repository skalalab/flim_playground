"""log10_columns: the one log transform the app and its exported scripts share."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.vis.helpers import log10_columns


def test_log10_columns_transforms_nonnegative_columns_and_reports_negative_ones():
    df = pd.DataFrame({"a": [1.0, 10.0], "b": [-1.0, 2.0], "c": [0.0, 1.0]})
    out, skipped = log10_columns(df, ["a", "b"])
    assert skipped == ["b"]
    np.testing.assert_allclose(out["a"], np.log10(df["a"] + 1e-6))
    assert out["b"].tolist() == [-1.0, 2.0]
    assert out["c"].tolist() == [0.0, 1.0]
    assert df["a"].tolist() == [1.0, 10.0], "the input frame is left untouched"


def test_log10_columns_with_nothing_to_transform_returns_the_frame_and_no_report():
    df = pd.DataFrame({"a": [1.0, 10.0]})
    out, skipped = log10_columns(df, [])
    assert skipped == []
    pd.testing.assert_frame_equal(out, df)
