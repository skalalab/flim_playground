"""Validate acquisition sources and prepare explicit extraction settings."""

from copy import deepcopy
import math
from os import PathLike
from pathlib import Path

from src.file_io import validate_ptu_reference_timing
from src.qpi import BACKGROUND_METHODS, OPD_UNITS


_STANDARD = "Fluorescence Lifetime Standard"
BACKGROUND_KEYS = ("method", "degree", "expand_pct")


def background_columns(channel_name):
    """The four output columns recording a channel's confirmed QPI recipe."""
    return [f"{channel_name}_bg_{key}" for key in BACKGROUND_KEYS]


def pending_calibration(metadata_df, settings):
    """Return unconfirmed FLIM and QPI channels in configured order."""
    shifts = [ch for ch in settings.get("channels_shift", {})
              if f"{ch}_shift" not in metadata_df]
    backgrounds = [ch for ch in settings.get("channels_background", [])
                   if "background" not in settings[ch]
                   or not all(col in metadata_df for col in background_columns(ch))]
    return shifts, backgrounds


def validate_background_settings(channel_name, recipe):
    """Validate one complete, explicit calibration recipe before saving it."""
    if not isinstance(recipe, dict) or any(key not in recipe for key in BACKGROUND_KEYS):
        return f"Confirm background correction settings for {channel_name} before extracting.", None
    method = recipe["method"]
    if method not in BACKGROUND_METHODS:
        return f"Background method for {channel_name} must be one of {', '.join(BACKGROUND_METHODS)}.", None
    degree = _number(recipe["degree"], integer=True) if method == "polynomial" else None
    expand = _number(recipe["expand_pct"])
    if (method == "polynomial" and degree not in (2, 4, 6)) or expand is None or expand > 100:
        return f"Background settings for {channel_name} are invalid.", None
    return "", {"method": method, "degree": degree, "expand_pct": expand}


def _single_value(fov_df, column):
    if column not in fov_df.columns:
        return f"Column `{column}` not found in the FOV table.", None
    values = fov_df[column]
    if values.isna().any() or values.nunique() != 1:
        return f"Column `{column}` is not consistent or contains missing values.", None
    return "", values.iloc[0]


def _number(value, *, minimum=0, integer=False):
    """Return a finite number, or None for an invalid acquisition setting."""
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number < minimum:
        return None
    if integer:
        return int(number) if number.is_integer() else None
    return number


def _validate_paths(fov_df, channel_name, file_types):
    for file_type in file_types:
        column = f"{channel_name}_{file_type}"
        if column not in fov_df.columns:
            return f"File paths column `{column}` not found in the FOV table."
        paths = []
        for value in fov_df[column]:
            if not isinstance(value, (str, PathLike)) or not str(value).strip():
                return f"File path {value} for {column} is not valid."
            path = Path(value)
            if not path.is_file():
                return f"File path {value} for {column} is not valid."
            paths.append(path.resolve())
        if file_type not in ("IRF", _STANDARD) and len(set(paths)) != len(paths):
            return f"File paths for {column} are not unique."
    return ""


def prepare_extraction(
    fov_df, channels, *, fov_name_col, unique_cell_id_col,
    derived_features=(), laser_rate=None, fit_free_calibration_method=None,
    fluorescence_lifetime_standard_lifetime=None,
):
    """Return ``(error_msg, settings_dict)`` without modifying the FOV table.

    ``channels`` maps display names to the input type, imaging modality, selected
    extractors, and fitting definitions. These settings are never inferred from
    table columns or loaded from configuration. The table supplies source paths,
    channel numbers, acquisition timing, and a standard's validated time axis.
    Raw image geometry is validated by the folder setup before this function.
    """
    if fov_name_col not in fov_df.columns:
        return f"Column of field of view names `{fov_name_col}` not found in the FOV table.", None
    if fov_df.empty:
        return "No fields of view are available for extraction.", None
    if fov_df[fov_name_col].isna().any():
        return f"Field of view names are missing in column `{fov_name_col}`.", None
    if fov_df[fov_name_col].duplicated().any():
        return f"Field of view names are not unique. Check the column `{fov_name_col}`.", None
    if not channels:
        return "No channels selected for extraction.", None

    settings = {
        "channel_names": [], "channels_shift": {}, "channels_background": [],
        "fov_name_col": fov_name_col, "unique_cell_id_col": unique_cell_id_col,
        "derived_features": deepcopy(list(derived_features)),
        "fix_shift": True,
    }
    needs_timing = False
    for channel_name, definition in channels.items():
        input_type = definition["input_type"]
        imaging_modality = definition["imaging_modality"]
        extractors = list(definition["selected_feature_extractors"])
        if not extractors:
            return f"No feature extractors selected for channel {channel_name}.", None
        channel = {
            "input_type": input_type, "imaging_modality": imaging_modality,
            "selected_feature_extractors": deepcopy(extractors),
        }
        settings[channel_name] = channel
        settings["channel_names"].append(channel_name)
        if imaging_modality == "QPI":
            constants = definition.get("qpi", {})
            channel["qpi"] = {}
            for key in ("pixel_size_um", "alpha_um3_per_pg"):
                value = _number(constants.get(key))
                if value is None or value <= 0:
                    return f"QPI {key} for {channel_name} must be positive and finite.", None
                channel["qpi"][key] = value
            if constants.get("opd_unit") not in OPD_UNITS:
                return f"QPI opd_unit for {channel_name} must be one of {', '.join(OPD_UNITS)}.", None
            channel["qpi"]["opd_unit"] = constants["opd_unit"]
            if {"Dry-mass statistics", "Spatial texture"} & set(extractors):
                settings["channels_background"].append(channel_name)
        for extractor in extractors:
            settings.setdefault(extractor, []).append(channel_name)

        fit = "Lifetime fit" in extractors
        fit_free = "Lifetime fit free" in extractors
        prefitted = "prefitted" in input_type
        raw_fit = fit and not prefitted
        prefitted_only = prefitted and extractors == ["Lifetime fit"]
        if imaging_modality == "FLIM":
            if settings.setdefault("decay_input_type", input_type) != input_type:
                return "Decay input type should be consistent across all channels.", None
            needs_timing |= not prefitted_only

        if fit:
            components = _number(definition.get("num_components"), minimum=1, integer=True)
            if components is None or components > 3:
                return f"Number of components for {channel_name} must be an integer from 1 to 3.", None
            channel["num_components"] = components
            channel["fixed_lifetimes"] = {}
            for component, value in (definition.get("fixed_lifetimes") or {}).items():
                try:
                    lifetime = float(value) if value is not None else math.nan
                except (TypeError, ValueError):
                    lifetime = -1
                if math.isnan(lifetime) or lifetime == 0:
                    channel["fixed_lifetimes"][component] = None
                elif math.isfinite(lifetime) and lifetime > 0:
                    channel["fixed_lifetimes"][component] = lifetime
                else:
                    return f"Fixed lifetime {component} for {channel_name} must be positive and finite, or empty/zero for a free lifetime.", None
        if raw_fit:
            settings.update(fitting_algo="MLE", fitting_mode="Hybrid")
            settings["channels_shift"][channel_name] = "fit"
        elif fit_free and fit_free_calibration_method != _STANDARD:
            settings["channels_shift"][channel_name] = "fit free"

        if "Decay (3/4D)" in input_type and not prefitted_only:
            column = f"{channel_name}_channel"
            err, channel_no = _single_value(fov_df, column)
            if err:
                return err, None
            channel_no = _number(channel_no, minimum=-1, integer=True)
            if channel_no is None:
                return f"Channel number in `{column}` must be an integer of -1 or greater.", None
            channel["channel_no"] = channel_no

        if input_type in ("Intensity (2D)", "QPI (2D)"):
            file_types = [input_type, "Mask"]
        else:
            file_types = [] if input_type == "Decay (2D)" else ["Mask"]
            if not prefitted_only:
                file_types.append("Decay")
        if prefitted and fit:
            file_types.append("SPCImage t1")
            if channel["num_components"] >= 2:
                file_types.extend(["a1", "t2"])
            if channel["num_components"] == 3:
                file_types.extend(["a2", "t3"])
        if channel_name in settings["channels_shift"]:
            file_types.append("IRF")
        if fit_free and fit_free_calibration_method == _STANDARD:
            file_types.append(_STANDARD)
        err = _validate_paths(fov_df, channel_name, file_types)
        if err:
            return err, None

        if fit_free and fit_free_calibration_method == _STANDARD:
            err, reference = _single_value(fov_df, f"{channel_name}_{_STANDARD}")
            if err:
                return err, None
            channel["fluorescence_lifetime_standard_file"] = reference
            column = f"{channel_name}_fluorescence_lifetime_standard_time_axis"
            err, time_axis = _single_value(fov_df, column)
            if err:
                return err, None
            time_axis = _number(time_axis, integer=True)
            if time_axis not in (0, 1, 2):
                return f"Time axis in `{column}` must be 0, 1, or 2.", None
            channel["fluorescence_lifetime_standard_time_axis"] = time_axis

    if "Lifetime fit free" in settings:
        rate = _number(laser_rate)
        if rate is None or rate <= 0:
            return "Laser rate must be a positive finite number for fit-free extraction.", None
        if fit_free_calibration_method not in ("IRF", _STANDARD):
            return "Fit-free calibration method must be IRF or Fluorescence Lifetime Standard.", None
        settings["laser_rate"] = rate
        settings["fit_free_calibration_method"] = fit_free_calibration_method
        if fit_free_calibration_method == _STANDARD:
            lifetime = _number(fluorescence_lifetime_standard_lifetime)
            if lifetime is None or lifetime <= 0:
                return "Fluorescence lifetime standard lifetime must be a positive finite number.", None
            settings["fluorescence_lifetime_standard_lifetime"] = lifetime

    if needs_timing:
        for column in ("time_bins", "duration"):
            err, value = _single_value(fov_df, column)
            if err:
                return err, None
            value = _number(value, integer=column == "time_bins")
            if value is None or value <= 0:
                return f"Acquisition `{column}` must be a positive finite {'integer' if column == 'time_bins' else 'number'}.", None
            settings[column] = value
        for channel_name in settings.get("Lifetime fit", []):
            channel = settings[channel_name]
            if "prefitted" not in channel["input_type"]:
                channel.update(start=0, end=settings["time_bins"])

        # Header checks do not decode photons or mutate the acquisition table.
        # A supplied laser rate is authoritative over repeated settings columns.
        timing = fov_df[["time_bins", "duration"]].copy()
        if laser_rate is not None:
            timing["laser_rate"] = laser_rate
        for channel_name in settings["channel_names"]:
            reference = settings[channel_name].get("fluorescence_lifetime_standard_file")
            if reference is not None and Path(reference).suffix.lower() == ".ptu":
                err = validate_ptu_reference_timing(
                    reference, timing, settings["time_bins"],
                    f"Fluorescence lifetime standard for {channel_name}",
                )
                if err:
                    return err, None
            if channel_name in settings["channels_shift"]:
                column = f"{channel_name}_IRF"
                for reference, rows in fov_df.groupby(column, sort=False):
                    if Path(reference).suffix.lower() == ".ptu":
                        err = validate_ptu_reference_timing(
                            reference, timing.loc[rows.index], settings["time_bins"],
                            f"IRF for {channel_name}",
                        )
                        if err:
                            return err, None
    return "", settings
