"""In-memory extraction decisions and their automatically maintained output records."""

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
import json
from numbers import Real
import os
from pathlib import Path
import uuid

import numpy as np
import pandas as pd

from src.metadata import BACKGROUND_KEYS, background_columns, pending_calibration, validate_background_settings


def _output_path(folder, prefix):
    return Path(folder) / f"{prefix}_{datetime.now():%Y%m%d_%H%M%S_%f}.csv"


def atomic_save_csv(frame, path, *, index=False):
    """Replace only after a complete write in the destination directory succeeds."""
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        # os.open applies the process umask like a plain write; mkstemp-based
        # temporary files are 0600 and would leave records unreadable to others.
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            frame.to_csv(stream, index=index)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception as exc:
        return f"Could not save {path}: {exc}"
    finally:
        temporary.unlink(missing_ok=True)
    return ""


def _valid_shift(value, row_count):
    values = np.asarray(value, dtype=object)
    if values.ndim > 1 or (values.ndim == 1 and len(values) != row_count):
        return False
    return all(
        isinstance(v, Real) and not isinstance(v, (bool, np.bool_)) and np.isfinite(v)
        for v in values.flat
    )


@dataclass
class ExtractionSession:
    metadata_df: pd.DataFrame
    settings: dict
    metadata_path: Path
    calibration_confirmed: bool
    choosing_shift: bool = False
    metadata_error: str = ""
    features: pd.DataFrame | None = None
    features_path: Path | None = None
    features_error: str = ""

    @classmethod
    def create(cls, metadata_df, settings, output_folder):
        session = cls(
            metadata_df.copy(deep=True), deepcopy(settings),
            _output_path(output_folder, "fov_metadata"),
            calibration_confirmed=not (settings["channels_shift"] or settings.get("channels_background")),
        )
        session.save_metadata()
        return session

    def _sync_settings(self):
        """Write settings into the existing row interface used by extraction."""
        for key in ("duration", "time_bins", "laser_rate", "fit_free_calibration_method",
                    "fluorescence_lifetime_standard_lifetime", "fitting_algo", "fitting_mode",
                    "fix_shift"):
            if key in self.settings:
                self.metadata_df[key] = self.settings[key]
        self.metadata_df["derived_features"] = json.dumps(self.settings.get("derived_features", []))
        for channel in self.settings["channel_names"]:
            settings = self.settings[channel]
            for key in ("input_type", "imaging_modality", "num_components", "start", "end",
                        "fluorescence_lifetime_standard_time_axis"):
                if key in settings:
                    self.metadata_df[f"{channel}_{key}"] = settings[key]
            for extractor in settings["selected_feature_extractors"]:
                self.metadata_df[f"{channel}_{extractor}"] = True
            if "fixed_lifetimes" in settings:
                for component in ("t1", "t2", "t3"):
                    self.metadata_df[f"{channel}_fixed_{component}"] = settings["fixed_lifetimes"].get(component)
            for key, value in settings.get("qpi", {}).items():
                self.metadata_df[f"{channel}_{key}"] = value
            if "background" in settings:
                for column, key in zip(background_columns(channel), BACKGROUND_KEYS):
                    self.metadata_df[column] = settings["background"][key]

    def save_metadata(self):
        self._sync_settings()
        self.metadata_error = atomic_save_csv(self.metadata_df, self.metadata_path)
        return self.metadata_error

    @property
    def can_extract(self):
        return self.calibration_confirmed and not self.metadata_error and not self._shift_error() and not self._background_error()

    def _shift_error(self):
        for channel in self.settings["channels_shift"]:
            column = f"{channel}_shift"
            if column not in self.metadata_df or not _valid_shift(self.metadata_df[column], len(self.metadata_df)):
                return f"Confirm valid shifts for every required channel before extracting ({channel})."
        return ""

    def _background_error(self):
        for channel in self.settings.get("channels_background", []):
            error, _ = validate_background_settings(channel, self.settings[channel].get("background"))
            if error:
                return error
        return ""

    def invalidate_results(self):
        self.features = None
        self.features_path = None
        self.features_error = ""

    def begin_recalibration(self):
        self.calibration_confirmed = False
        self.choosing_shift = True
        self.metadata_error = ""
        self.invalidate_results()
        columns = [f"{channel}_shift" for channel in self.settings["channels_shift"]]
        for channel in self.settings.get("channels_background", []):
            columns.extend(background_columns(channel))
            self.settings[channel].pop("background", None)
        self.metadata_df = self.metadata_df.drop(columns=columns, errors="ignore")

    def confirm_calibration(self, shifts, backgrounds=None):
        backgrounds = backgrounds or {}
        _, pending_backgrounds = pending_calibration(self.metadata_df, self.settings)
        for channel in self.settings["channels_shift"]:
            value = shifts.get(channel, self.metadata_df.get(f"{channel}_shift"))
            if not _valid_shift(value, len(self.metadata_df)):
                return f"Confirm valid shifts for every required channel before extracting ({channel})."
        recipes = {}
        for channel in self.settings.get("channels_background", []):
            recipe = backgrounds.get(channel)
            if channel not in pending_backgrounds and channel not in backgrounds:
                recipe = self.settings[channel].get("background")
            error, recipes[channel] = validate_background_settings(channel, recipe)
            if error:
                return error
        for channel in self.settings["channels_shift"]:
            if channel in shifts:
                shift = np.asarray(shifts[channel], dtype=float)
                self.metadata_df[f"{channel}_shift"] = float(shift) if shift.ndim == 0 else shift
        for channel, recipe in recipes.items():
            self.settings[channel]["background"] = recipe
            self.settings[channel]["background_defaults"] = dict(recipe)
        self.calibration_confirmed = True
        self.choosing_shift = False
        self.invalidate_results()
        return self.save_metadata()

    def change_mode(self, mode):
        if mode not in ("Hybrid", "Local"):
            return f"Unsupported fitting mode: {mode}"
        if mode == self.settings.get("fitting_mode"):
            return self.metadata_error
        self.settings["fitting_mode"] = mode
        self.invalidate_results()
        # Recalibration edits become durable only on confirmation.
        return self.save_metadata() if self.calibration_confirmed else ""

    def before_extraction(self):
        if not self.calibration_confirmed:
            return "Confirm calibration before extracting."
        error = self._shift_error() or self._background_error()
        if error:
            return error
        return self.save_metadata()

    def export_features(self, features):
        self.features = features.copy(deep=True)
        self.features_path = _output_path(self.metadata_path.parent, "single_cell_features")
        return self.save_features()

    def save_features(self):
        self.features_error = atomic_save_csv(self.features, self.features_path, index=True)
        return self.features_error
