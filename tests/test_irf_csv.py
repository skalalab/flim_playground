"""Text IRFs: one column (IRF) or exactly two (time bin, IRF), one row per time bin."""

import numpy as np
import pandas as pd
import pytest

from src.file_io import get_irf


def _irf(tmp_path, rows, name="irf.csv"):
    path = tmp_path / name
    path.write_text("\r\n".join(",".join(map(str, r)) for r in rows) + "\r\n")
    return pd.Series({"ch1_IRF": str(path)})


@pytest.mark.parametrize("name", ["irf.csv", "IRF.CSV"])
@pytest.mark.parametrize("with_axis", [False, True])
def test_one_or_two_columns_load_the_irf(tmp_path, with_axis, name):
    values = np.arange(128) * 1.5
    rows = zip(range(128), values) if with_axis else ((v,) for v in values)
    err, irf = get_irf(_irf(tmp_path, rows, name), "ch1", 128)
    assert err == ""
    np.testing.assert_array_equal(irf, values)


def test_three_columns_are_refused(tmp_path):
    err, irf = get_irf(_irf(tmp_path, [(i, 1, 2) for i in range(128)]), "ch1", 128)
    assert irf is None and "(128, 3)" in err


def test_short_two_column_file_reports_its_value_count(tmp_path):
    err, irf = get_irf(_irf(tmp_path, zip(range(127), [0.0] * 127)), "ch1", 128)
    assert irf is None and "127" in err and "128" in err and "254" not in err


def test_two_value_columns_stay_refused(tmp_path):
    # Two IRFs side by side: the first column is not a bin index.
    rows = [(v, v) for v in [0, 5, 3, 0] * 32]
    err, irf = get_irf(_irf(tmp_path, rows), "ch1", 128)
    assert irf is None and "time-bin axis" in err
