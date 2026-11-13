"""Results whose inputs did not change are reused: the sina spread and the last classifier fit."""
import numpy as np
import pandas as pd
import pytest

from src import classify, table_session
from src.export_script.skeleton import _extract_source
from src.vis import feature_comparison as fc
from src.widgets import classification_widgets


def test_the_cached_spread_equals_the_plain_one():
    values = np.random.default_rng(1).normal(size=5_000)
    np.testing.assert_array_equal(fc._cached_sina_offsets(values, 0.35), fc.sina_offsets(values, 0.35))


def test_the_density_is_computed_once_for_unchanged_values(monkeypatch):
    fc._cached_sina_offsets.clear()
    calls, real = [], fc.sina_offsets
    monkeypatch.setattr(fc, "sina_offsets", lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    values = np.random.default_rng(2).normal(size=2_000)
    fc._cached_sina_offsets(values, 0.35)
    fc._cached_sina_offsets(values.copy(), 0.35)
    assert len(calls) == 1
    fc._cached_sina_offsets(values + 1, 0.35)
    fc._cached_sina_offsets(values, 0.2)
    assert len(calls) == 3


def test_exports_paste_the_plain_spread():
    assert _extract_source(fc.sina_offsets).startswith("def sina_offsets(")


@pytest.fixture
def fits(monkeypatch):
    calls = []
    monkeypatch.setattr(classify, "run_classification",
                        lambda *a, **k: (calls.append((a, k)), ("", {"n": len(calls)}))[1])
    return calls


def _data(shift=0.0):
    return pd.DataFrame({"a": [0.0 + shift, 1.0, 2.0, 3.0], "classes": ["x", "y", "x", "y"]})


BASE = ["Random Forest", 0.7, "None", "None", "None", {"n_estimators": 100}]


def test_an_unchanged_request_does_not_retrain(fits):
    state = {}
    first = classification_widgets.classification_result(state, _data(), *BASE)
    assert classification_widgets.classification_result(state, _data(), *BASE) is first and len(fits) == 1
    assert fits[0][1] == {"classifier_params": {"n_estimators": 100}, "random_state": 42}


@pytest.mark.parametrize("position, value", [(0, "SVM"), (1, 0.8), (2, "Oversampling"), (3, "Balanced"),
                                             (4, "F1 Score"), (5, {"n_estimators": 200})])
def test_any_changed_setting_retrains(fits, position, value):
    state, changed = {}, list(BASE)
    changed[position] = value
    classification_widgets.classification_result(state, _data(), *BASE)
    classification_widgets.classification_result(state, _data(), *changed)
    assert len(fits) == 2


def test_changed_data_retrains(fits):
    state = {}
    classification_widgets.classification_result(state, _data(), *BASE)
    classification_widgets.classification_result(state, _data(1.0), *BASE)
    assert len(fits) == 2


def test_clearing_the_upload_drops_the_fit(fits):
    state = {}
    classification_widgets.classification_result(state, _data(), *BASE)
    table_session.clear_table_session(state)
    assert "_classification_memo" not in state
