from contextlib import nullcontext
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.fov_extraction as fov_module
import src.widgets.lifetime_widgets as lifetime_widgets
from src.fit_helper import irf_shift


@pytest.mark.parametrize("shift", [np.nan, np.inf, -np.inf, "not-a-number", True])
def test_irf_shift_rejects_non_finite_or_non_numeric_shift(shift):
    irf = np.array([0.0, 1.0, 0.0, 0.0])

    with pytest.raises(ValueError, match="finite number"):
        irf_shift(irf, shift)


def test_choose_shift_widget_rejects_any_failed_shift_fit(monkeypatch):
    results = {
        "shift": np.array([0.5, np.nan]),
        "decay_id": ["fov1", "fov2"],
    }
    metadata_df = pd.DataFrame({"image_name": ["fov1", "fov2"]})
    metadata_dict = {
        "duration": 12.5,
        "time_bins": 4,
        "decay_input_type": "Decay (3/4D)",
        "channels_shift": {"ch1": "fit"},
        "fix_shift": True,
        "fitting_algo": "WLS",
        "fitting_mode": "Local",
        "ch1": {
            "num_components": 1,
            "start": 0,
            "end": 4,
            "fixed_lifetimes": {},
        },
    }

    monkeypatch.setattr(
        lifetime_widgets,
        "choose_shift_fit",
        lambda *args, **kwargs: ("", results),
    )
    monkeypatch.setattr(
        lifetime_widgets,
        "display_shift_data_widget",
        lambda *args, **kwargs: pytest.fail("invalid shifts must be rejected before display"),
    )
    monkeypatch.setattr(
        lifetime_widgets.st,
        "number_input",
        lambda *args, value, **kwargs: value,
    )

    error_msg, shift_data = lifetime_widgets.choose_shift_widget(
        metadata_df,
        metadata_dict,
        "image_name",
        "ch1",
    )

    assert "failed" in error_msg.lower()
    assert "fov2" in error_msg
    assert "correct acquisition channel" in error_msg.lower()
    assert shift_data is None


def test_choose_shift_widget_keeps_finite_shift_results(monkeypatch):
    results = {
        "shift": np.array([0.5, 1.5]),
        "decay_id": ["fov1", "fov2"],
    }
    metadata_df = pd.DataFrame({"image_name": ["fov1", "fov2"]})
    metadata_dict = {
        "duration": 12.5,
        "time_bins": 4,
        "decay_input_type": "Decay (3/4D)",
        "channels_shift": {"ch1": "fit"},
        "fix_shift": True,
        "fitting_algo": "WLS",
        "fitting_mode": "Local",
        "ch1": {
            "num_components": 1,
            "start": 0,
            "end": 4,
            "fixed_lifetimes": {},
        },
    }

    monkeypatch.setattr(
        lifetime_widgets,
        "choose_shift_fit",
        lambda *args, **kwargs: ("", results),
    )
    monkeypatch.setattr(lifetime_widgets, "display_shift_data_widget", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        lifetime_widgets.st,
        "number_input",
        lambda *args, value, **kwargs: value,
    )

    error_msg, shift_data = lifetime_widgets.choose_shift_widget(
        metadata_df,
        metadata_dict,
        "image_name",
        "ch1",
    )

    assert error_msg == ""
    assert shift_data == pytest.approx(1.0)


def test_choose_shift_widget_rejects_empty_shift_results(monkeypatch):
    results = {"shift": np.array([]), "decay_id": []}
    metadata_df = pd.DataFrame({"image_name": ["fov1"]})
    metadata_dict = {
        "duration": 12.5,
        "time_bins": 4,
        "decay_input_type": "Decay (3/4D)",
        "channels_shift": {"ch1": "fit"},
        "fix_shift": True,
        "fitting_algo": "WLS",
        "fitting_mode": "Local",
        "ch1": {
            "num_components": 1,
            "start": 0,
            "end": 4,
            "fixed_lifetimes": {},
        },
    }

    monkeypatch.setattr(
        lifetime_widgets,
        "choose_shift_fit",
        lambda *args, **kwargs: ("", results),
    )
    monkeypatch.setattr(
        lifetime_widgets,
        "display_shift_data_widget",
        lambda *args, **kwargs: pytest.fail("empty shifts must be rejected before display"),
    )

    error_msg, shift_data = lifetime_widgets.choose_shift_widget(
        metadata_df,
        metadata_dict,
        "image_name",
        "ch1",
    )

    assert "no decay curves" in error_msg.lower()
    assert shift_data is None


def test_choose_shift_widget_rejects_fov_without_matching_decay(monkeypatch):
    results = {"shift": np.array([0.5]), "decay_id": ["fov1_0"]}
    metadata_df = pd.DataFrame({"image_name": ["fov1", "fov2"]})
    metadata_dict = {
        "duration": 12.5,
        "time_bins": 4,
        "decay_input_type": "Decay (2D)",
        "channels_shift": {"ch1": "fit"},
        "fix_shift": False,
        "fitting_algo": "WLS",
        "fitting_mode": "Local",
        "ch1": {
            "num_components": 1,
            "start": 0,
            "end": 4,
            "fixed_lifetimes": {},
        },
    }

    monkeypatch.setattr(
        lifetime_widgets,
        "choose_shift_fit",
        lambda *args, **kwargs: ("", results),
    )
    monkeypatch.setattr(lifetime_widgets, "display_shift_data_widget", lambda *args, **kwargs: None)

    error_msg, shift_data = lifetime_widgets.choose_shift_widget(
        metadata_df,
        metadata_dict,
        "image_name",
        "ch1",
    )

    assert "fov2" in error_msg
    assert "no matching decay" in error_msg.lower()
    assert shift_data is None


def test_extract_lifetime_features_rejects_nan_metadata_shift(monkeypatch):
    metadata = pd.Series(
        {
            "image_name": "fov1",
            "time_bins": 4,
            "duration": 12.5,
            "laser_rate": 0.08,
            "ch1_shift": np.nan,
        }
    )
    monkeypatch.setattr(
        fov_module,
        "get_decay_curves",
        lambda *args, **kwargs: ("", {"fov1_1": np.ones(4)}),
    )
    monkeypatch.setattr(
        fov_module,
        "get_irf",
        lambda *args, **kwargs: ("", np.array([0.0, 1.0, 0.0, 0.0])),
    )
    monkeypatch.setattr(
        fov_module,
        "extract_fit_free_results",
        lambda *args, **kwargs: ("", {"fov1_1": {"feature": 1.0}}),
    )
    monkeypatch.setattr(fov_module.st, "empty", lambda: _DummyContainer())
    monkeypatch.setattr(fov_module.st, "progress", lambda *args, **kwargs: _DummyProgress())

    error_msg, result = fov_module.extract_lifetime_features(
        metadata,
        channel_name="ch1",
        input_type="Decay (2D)",
        fit=False,
        fit_free=True,
        fov_col_name="image_name",
        calibration_method="IRF",
    )

    assert "invalid shift" in error_msg.lower()
    assert "fov1" in error_msg
    assert result.empty


class _DummyContainer:
    def container(self):
        return nullcontext()

    def empty(self):
        return None


class _DummyProgress:
    def progress(self, value):
        return None


def test_standard_calibration_ignores_stale_shift_column(monkeypatch):
    metadata = pd.Series(
        {
            "image_name": "fov1",
            "time_bins": 4,
            "duration": 12.5,
            "laser_rate": 0.08,
            "ch1_shift": np.nan,
        }
    )
    monkeypatch.setattr(
        fov_module,
        "get_decay_curves",
        lambda *args, **kwargs: ("", {"fov1_1": np.ones(4)}),
    )
    monkeypatch.setattr(
        fov_module,
        "get_irf",
        lambda *args, **kwargs: pytest.fail("standard-only calibration does not need an IRF shift"),
    )
    monkeypatch.setattr(
        fov_module,
        "extract_fit_free_results",
        lambda *args, **kwargs: ("", {"fov1_1": {"feature": 1.0}}),
    )
    monkeypatch.setattr(fov_module.st, "empty", lambda: _DummyContainer())
    monkeypatch.setattr(fov_module.st, "progress", lambda *args, **kwargs: _DummyProgress())

    error_msg, result = fov_module.extract_lifetime_features(
        metadata,
        channel_name="ch1",
        input_type="Decay (2D)",
        fit=False,
        fit_free=True,
        fov_col_name="image_name",
        calibration_method="Fluorescence Lifetime Standard",
        fluorescence_lifetime_standard_image=np.ones((2, 2, 4)),
        fluorescence_lifetime_standard_lifetime=4.0,
        fluorescence_lifetime_standard_time_axis=2,
    )

    assert error_msg == ""
    assert result.loc["fov1_1", "feature"] == pytest.approx(1.0)


def test_prefitted_fit_with_standard_calibration_does_not_require_shift(monkeypatch):
    metadata = pd.Series(
        {
            "image_name": "fov1",
            "time_bins": 4,
            "duration": 12.5,
            "laser_rate": 0.08,
            "ch1_shift": np.nan,
            "ch1_num_components": 1,
        }
    )
    monkeypatch.setattr(
        fov_module,
        "get_decay_curves",
        lambda *args, **kwargs: ("", {"fov1_1": np.ones(4)}),
    )
    monkeypatch.setattr(
        fov_module,
        "get_irf",
        lambda *args, **kwargs: pytest.fail("prefitted results and standard calibration do not need an IRF shift"),
    )
    monkeypatch.setattr(
        fov_module,
        "extract_spcimage_fit_results",
        lambda *args, **kwargs: ("", pd.DataFrame({"fit_feature": [2.0]}, index=["fov1_1"])),
    )
    monkeypatch.setattr(
        fov_module,
        "extract_fit_free_results",
        lambda *args, **kwargs: ("", {"fov1_1": {"fit_free_feature": 1.0}}),
    )
    monkeypatch.setattr(fov_module.st, "empty", lambda: _DummyContainer())
    monkeypatch.setattr(fov_module.st, "progress", lambda *args, **kwargs: _DummyProgress())

    error_msg, result = fov_module.extract_lifetime_features(
        metadata,
        channel_name="ch1",
        input_type="Decay (3/4D) pixel-prefitted",
        fit=True,
        fit_free=True,
        fov_col_name="image_name",
        calibration_method="Fluorescence Lifetime Standard",
        fluorescence_lifetime_standard_image=np.ones((2, 2, 4)),
        fluorescence_lifetime_standard_lifetime=4.0,
        fluorescence_lifetime_standard_time_axis=2,
    )

    assert error_msg == ""
    assert result.loc["fov1_1", "fit_feature"] == pytest.approx(2.0)
    assert result.loc["fov1_1", "fit_free_feature"] == pytest.approx(1.0)
