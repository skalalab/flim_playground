"""Feature Comparison copies only the columns it reads, so a wide table costs no more than a narrow one."""
import tracemalloc

import numpy as np
import pandas as pd
import pytest

from src.vis import feature_comparison as fc


def _table(rows, extra):
    rng = np.random.default_rng(0)
    data = {"id": [f"c{i}" for i in range(rows)], "g": rng.choice(["a", "b"], rows),
            "s": rng.choice(["p", "q"], rows), "v": rng.normal(size=rows)}
    data.update({f"x{i}": rng.normal(size=rows) for i in range(extra)})
    return pd.DataFrame(data)


def _figure(frame, **kwargs):
    # JSON, not to_dict: the figure holds arrays, which a dict comparison cannot test for truth.
    return fc.feature_comparison_plot(frame, unique_row_id_col="id", fov_name_col=None, selected_var="v",
                                      color_by=["g"], source_df=frame, source_row_id_col="id", **kwargs).to_json()


@pytest.mark.parametrize("kwargs", [{}, {"separate_by": "s"}, {"shape_by": "s"}, {"subcolor_by": "s"},
                                    {"effect_size_method": "Absolute Cohen's d", "mean_or_median": "Mean",
                                     "statistical_test": "Welch's t-test"}])
def test_unread_columns_do_not_change_the_figure(kwargs):
    assert _figure(_table(400, 0), **kwargs) == _figure(_table(400, 6), **kwargs)


def test_a_wide_table_costs_no_more_memory_than_a_narrow_one():
    def peak(frame):
        tracemalloc.start()
        _figure(frame)
        used = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        return used

    # 60 unread float columns of 50,000 rows are 24 MB when copied.
    assert peak(_table(50_000, 60)) < peak(_table(50_000, 0)) + 8_000_000
