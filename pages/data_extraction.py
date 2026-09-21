import json
import os
from dataclasses import asdict, dataclass

import pandas as pd
import streamlit as st

from src.celebrate import celebrate
from src.config import (
    get_channel_names,
    get_config_mtime,
    get_current_profile_name,
    get_decay_input_type,
    get_derived_features,
    get_fit_free_calibration_method,
    get_fixed_lifetimes,
    get_fov_name_col,
    get_imaging_modality,
    get_input_types,
    get_num_components,
    get_qpi_constants,
    get_reference_file_suffixes,
    get_selected_feature_extractors,
    get_unique_cell_id_col,
)
from src.config_watch import notify_on_config_change
from src.emojis import happy_emoji, sad_emoji
from src.extraction_session import ExtractionSession
from src.file_io import get_lifetime_standard
from src.metadata import pending_calibration, prepare_extraction
from src.navigation import render_top_menu
from src.widgets.analysis_widget_state import (
    control_default,
    number_input_default,
    preserve_analysis_controls,
)
from src.widgets.category_widgets import (
    check_and_merge_df_widget,
    find_available_dfs_widget,
    map_categories_to_labels_widget,
)
from src.widgets.lifetime_widgets import choose_shift_widget, fit_options_widget
from src.widgets.metadata_widgets import (
    check_assign_channel_widget,
    clear_folder_scan_caches,
    lifetime_data_config_widget,
    load_data_suffix_widget,
    load_list_data_from_folder_widget,
    preview_metadata_widget,
)
from src.widgets.numeric_extraction_widgets import fov_extraction_widget
from src.widgets.qpi_widgets import choose_background_widget

# Shared labels for the workflow selector and dispatch.
STEP_NUMERIC = "**Numerical** (e.g. lifetime, morphology)"
STEP_CATEGORICAL = "**Categorical** (e.g. treatment, day)"
STEPS = [STEP_NUMERIC, STEP_CATEGORICAL]


# --- Cross-step context ----------------------------------------------------
@dataclass
class ExtractionContext:
    """Config values shared by all workflow steps.

    Resolve once from the active profile and pass to step renderers and helpers.
    """
    channel_names: dict
    input_types: dict
    imaging_modalities: dict
    has_flim: bool
    decay_input_type: str
    ch_num_components: dict
    selected_ch_feature_extractors: dict
    fov_name_col: str
    fit_free_calibration_method: object
    fluorescence_lifetime_standard_lifetime: object
    fixed_lifetimes: dict
    derived_features: list
    unique_cell_id_col: str
    qpi_constants: dict


def build_context():
    """Resolve all config-derived, step-independent values from the active profile."""
    channel_names = get_channel_names()
    input_types = get_input_types(channel_names.keys())
    imaging_modalities = get_imaging_modality(channel_names.keys())
    has_flim = "FLIM" in imaging_modalities.values()
    decay_input_type = get_decay_input_type()
    ch_num_components = get_num_components(input_types, channel_names.keys())
    selected_ch_feature_extractors = get_selected_feature_extractors(input_types, channel_names.keys())
    fov_name_col = get_fov_name_col()
    fit_free_calibration_method, fluorescence_lifetime_standard_lifetime = get_fit_free_calibration_method(decay_input_type)
    return ExtractionContext(
        channel_names=channel_names,
        input_types=input_types,
        imaging_modalities=imaging_modalities,
        has_flim=has_flim,
        decay_input_type=decay_input_type,
        ch_num_components=ch_num_components,
        selected_ch_feature_extractors=selected_ch_feature_extractors,
        fov_name_col=fov_name_col,
        fit_free_calibration_method=fit_free_calibration_method,
        fluorescence_lifetime_standard_lifetime=fluorescence_lifetime_standard_lifetime,
        fixed_lifetimes={key: get_fixed_lifetimes(key, input_types[key]) for key in channel_names},
        derived_features=get_derived_features(),
        unique_cell_id_col=get_unique_cell_id_col(),
        qpi_constants={key: get_qpi_constants(key, input_types[key])
                       for key in channel_names if imaging_modalities[key] == "QPI"},
    )


def invalidate_preparation():
    """Discard decisions and display state; previous output files remain records."""
    st.session_state["prepared_extraction"] = None
    st.session_state.pop("extraction_source", None)
    st.session_state.pop("autostart_extraction", None)


def preserve_source_controls(ctx):
    """Keep source-folder controls alive across steps and visible after remounting.

    Reassigning a key while its widget is hidden prevents Streamlit's cleanup; the
    same reassignment in the run that recreates the widget is what sends the value
    to the browser, which otherwise remounts the widget with its constructor
    default and reports that default on the next rerun. Runs before any source
    widget on every page run, like the analysis page's control preservation.
    Fitting controls need no key: they rebuild from the prepared settings.
    """
    keys = {
        "fov_metadata_folder_path", "2D_decay_duration", "2D_decay_time_bins",
        "2D_decay_laser_rate_mhz", "laser_rate_mhz",
        "fluorescence_lifetime_standard_lifetime",
    }
    for key, name in ctx.channel_names.items():
        keys.update((f"has_channel_{key}", f"num_component_{name}", f"{name}_channel_selectbox"))
        prefix = f"{name}_{ctx.input_types[key]}_"
        keys.update(k for k in st.session_state if k.startswith(prefix) and k.endswith("_suffix"))
    preserve_analysis_controls(st.session_state, keys)


# --- Source folder preparation helpers ---------------------------------------
def validate_folder_path(folder_path):
    """Validate folder path and return appropriate error message"""
    if folder_path == "":
        st.info("Please provide a folder path.")
        return False
    if not os.path.isdir(folder_path):
        st.error(f"Folder not found! Please check the path. {sad_emoji}")
        return False
    return True


def load_and_validate_fovs(folder_path, actual_file_suffix, reference_suffixes=()):
    """Load FOVs from folder and validate"""
    fovs = load_list_data_from_folder_widget(folder_path, file_suffix=actual_file_suffix,
                                            reference_suffixes=reference_suffixes)
    if len(fovs) == 0:
        st.warning("No data found in the folder. Please check the path and the file suffixes.")
        return None

    st.success(f"Fields of View with ✅ are loaded successfully {happy_emoji}. FOVs with ❌ (if any) will **not** be recorded. Here is the preview of the FOVs and metadata recorded:")
    return fovs


def prepare_fov_dataframe(fovs, selected_channels, selected_ch_num_components, ctx):
    """Prepare FOV dataframe with channel information"""
    fov_df = pd.DataFrame.from_dict(fovs, orient="index")

    # Set index name and reset to column
    fov_df.index.name = ctx.fov_name_col
    fov_df.reset_index(inplace=True)

    # Add channel information
    for channel_key, channel_name in selected_channels.items():
        fov_df[f"{channel_name}_input_type"] = ctx.input_types[channel_key]
        fov_df[f"{channel_name}_imaging_modality"] = ctx.imaging_modalities[channel_key]
        for feature_extractor in ctx.selected_ch_feature_extractors[channel_key]:
            fov_df[f"{channel_name}_{feature_extractor}"] = True
        if ctx.has_flim and channel_name in selected_ch_num_components:
            fov_df[f"{channel_name}_num_components"] = selected_ch_num_components[channel_name]
        # Write fixed-lifetime columns from config defaults (Step 1)
        if "Lifetime fit" in ctx.selected_ch_feature_extractors.get(channel_key, []):
            fixed_lts = ctx.fixed_lifetimes[channel_key]
            for t_key in ["t1", "t2", "t3"]:
                val = fixed_lts.get(t_key)  # None or float
                fov_df[f"{channel_name}_fixed_{t_key}"] = val
        for name, value in ctx.qpi_constants.get(channel_key, {}).items():
            fov_df[f"{channel_name}_{name}"] = value

    # Metadata is an output record; extraction uses these definitions in memory.
    fov_df["derived_features"] = json.dumps(ctx.derived_features)

    return fov_df


def validate_fluorescence_lifetime_standard_per_channel(fov_df, selected_channels, fit_free_calibration_method, time_bins, fluorescence_lifetime_standard_lifetime):
    """Validate and add per-channel fluorescence lifetime standard info if needed"""
    fov_df["fit_free_calibration_method"] = fit_free_calibration_method
    if fit_free_calibration_method != "Fluorescence Lifetime Standard":
        return "", fov_df

    # lifetime is shared across channels
    fov_df["fluorescence_lifetime_standard_lifetime"] = fluorescence_lifetime_standard_lifetime

    for channel_name in selected_channels.values():
        ref_col = f"{channel_name}_Fluorescence Lifetime Standard"
        if ref_col not in fov_df.columns:
            # Channel may not use fit free; skip
            continue
        unique_paths = fov_df[ref_col].dropna().unique().tolist()
        if len(unique_paths) != 1:
            return f"Fluorescence lifetime standard file path column {ref_col} is not consistent across FOVs.", fov_df
        error_msg, reference = get_lifetime_standard(fov_df, channel_name, time_bins)
        if error_msg:
            return error_msg, fov_df
        _, time_axis = reference
        fov_df[f"{channel_name}_fluorescence_lifetime_standard_time_axis"] = time_axis

    return "", fov_df


def finalize_fov_processing(fov_df, selected_channels, decay_input_type, imaging_modalities, duration, time_bins, folder_path, selected_ch_feature_extractors, fit_free_calibration_method=None, fluorescence_lifetime_standard_lifetime=None):
    """Assign channels, validate standards, then preview the FOV table.

    Follows the ``(error_msg, result)`` convention: returns ``("", fov_df)`` on
    success, or ``(error_msg, fov_df)`` at the first failing step. Rendering the
    error message is the caller's responsibility.
    """
    # Check and assign channels
    error_msg, fov_df = check_assign_channel_widget(
        fov_df, selected_channels,
        flim_decay_input_type=decay_input_type,
        imaging_modalities=imaging_modalities,
        selected_ch_feature_extractors=selected_ch_feature_extractors,
        duration=duration, time_bins=time_bins
    )
    if error_msg != "":
        return error_msg, fov_df

    # Validate fluorescence lifetime standard file per channel after channel assignment
    if fit_free_calibration_method is not None:
        time_bins = fov_df["time_bins"].iloc[0]
        error_msg, fov_df = validate_fluorescence_lifetime_standard_per_channel(fov_df, selected_channels, fit_free_calibration_method, time_bins, fluorescence_lifetime_standard_lifetime)
        if error_msg != "":
            return error_msg, fov_df

    preview_metadata_widget(fov_df)
    return "", fov_df


def _render_metadata_record(prepared):
    """Where the Start button stood: a failed save until it is retried, otherwise
    the saved path only in the view that follows the click, before calibration
    moves on to shift finding or extraction."""
    if prepared.metadata_error:
        st.error(prepared.metadata_error)
        if st.button("Retry saving metadata"):
            prepared.save_metadata()
            st.rerun()
    elif not prepared.choosing_shift and (
        not prepared.calibration_confirmed
        or not (prepared.settings["channels_shift"] or prepared.settings.get("channels_background"))
    ):
        st.success(f"Metadata is saved automatically to {prepared.metadata_path}")


def render_source_controls(ctx):
    """Render the metadata settings and return ``(error_msg, source)``.

    ``source`` collects every input the folder scan and preparation read, so a
    prepared session can be checked against the settings still on screen.
    """
    error_msg = ""
    actual_file_suffix = None
    selected_channels = {}
    selected_ch_num_components = {}
    duration = None
    time_bins = None
    laser_rate = None
    folder_path = ""
    fit_free_calibration_method = ctx.fit_free_calibration_method
    fluorescence_lifetime_standard_lifetime = ctx.fluorescence_lifetime_standard_lifetime

    checkbox_cols = st.columns(len(ctx.channel_names))

    for index, (channel_key, channel_name) in enumerate(ctx.channel_names.items()):
        with checkbox_cols[index]:
            # A False default suppresses the duplication warning once the key is restored.
            has_channel = st.checkbox(f"has {channel_name}", value=bool(control_default(st.session_state, f"has_channel_{channel_key}", True)), key=f"has_channel_{channel_key}")
            if has_channel:
                with st.expander(f"{channel_name}: {ctx.imaging_modalities[channel_key]}", expanded=False):
                    st.write(f"Feature extractors: {', '.join(ctx.selected_ch_feature_extractors[channel_key])}")
                selected_channels[channel_key] = channel_name
                if ctx.ch_num_components[channel_key] != 0 and "prefitted" in ctx.input_types[channel_key]:  # Prefitted component count determines required output files.
                    selected_ch_num_components[channel_name] = st.number_input("No. component", value=number_input_default(st.session_state, f"num_component_{channel_name}", ctx.ch_num_components[channel_key]), min_value=1, max_value=3, help="Number of components for the lifetime fit/fit free analysis" if index == 0 else None, key=f"num_component_{channel_name}")
                elif ctx.ch_num_components[channel_key] != 0:  # Configure raw-data components in the fitting step.
                    selected_ch_num_components[channel_name] = ctx.ch_num_components[channel_key]
    if len(selected_channels) == 0:
        error_msg = "Please check at least one of the channels"
        st.error(f"{error_msg} {sad_emoji}")
    else:
        if any(ctx.imaging_modalities[key] == "FLIM" for key in selected_channels):
            selected_extractors = {key: ctx.selected_ch_feature_extractors[key] for key in selected_channels}
            duration, time_bins, laser_rate = lifetime_data_config_widget(selected_extractors, ctx.decay_input_type)
        else:
            duration, time_bins, laser_rate = None, None, None
        if laser_rate is None:
            fit_free_calibration_method = None
        if fit_free_calibration_method == "Fluorescence Lifetime Standard":
            # Fluorescence lifetime standard file is per-channel and collected via suffixes; only lifetime is shared
            fluorescence_lifetime_standard_lifetime = st.number_input("Fluorescence lifetime standard's lifetime in **ns**", value=number_input_default(st.session_state, "fluorescence_lifetime_standard_lifetime", fluorescence_lifetime_standard_lifetime), min_value=0.1, max_value=20.0, step=0.1, key="fluorescence_lifetime_standard_lifetime")

        actual_file_suffix, error_msg = load_data_suffix_widget(ctx.input_types, selected_channels, selected_ch_num_components, ctx.selected_ch_feature_extractors)
        if error_msg != "":
            st.error(error_msg)
        else:
            folder_path = st.text_input("Copy the folder path here", help="The folder should contain all the raw data that is needed for the selected data extraction type.", key="fov_metadata_folder_path")
            if folder_path and st.button("Rescan folder", help="Re-read files from disk, ignoring cached results"):
                invalidate_preparation()
                clear_folder_scan_caches()
                st.rerun()

    return error_msg, {
        "selected_channels": selected_channels,
        "selected_ch_num_components": selected_ch_num_components,
        "duration": duration, "time_bins": time_bins, "laser_rate": laser_rate,
        "fit_free_calibration_method": fit_free_calibration_method,
        "fluorescence_lifetime_standard_lifetime": fluorescence_lifetime_standard_lifetime,
        "actual_file_suffix": actual_file_suffix, "folder_path": folder_path,
    }


def source_fingerprint(ctx, source):
    """Identify the profile and metadata settings a session was prepared from."""
    return json.dumps({"context": asdict(ctx), **source}, sort_keys=True, default=str)


def render_preparation(col1, col2, ctx):
    """Metadata view: review the source folder and create one validated session."""
    with col1:
        error_msg, source = render_source_controls(ctx)
    if error_msg:
        return
    selected_channels = source["selected_channels"]
    folder_path = source["folder_path"]
    laser_rate = source["laser_rate"]
    with col2:
        if not validate_folder_path(folder_path):
            return  # Error already displayed in function
        # Hidden calibration fields still identify references. Active
        # fields use the suffix currently entered by the user instead.
        reference_suffixes = tuple(
            suffix for key, channel in selected_channels.items()
            for kind, suffix in get_reference_file_suffixes(key, ctx.input_types[key]).items()
            if f"{channel}_{kind}" not in source["actual_file_suffix"]
        )
        fovs = load_and_validate_fovs(folder_path, source["actual_file_suffix"], reference_suffixes)
        if fovs is None:
            return  # Error already displayed in function
        fov_df = prepare_fov_dataframe(fovs, selected_channels, source["selected_ch_num_components"], ctx)
        if laser_rate is not None:
            fov_df["laser_rate"] = laser_rate

        # Validate channel assignments and calibration compatibility before preparation.
        error_msg, fov_df = finalize_fov_processing(
            fov_df, selected_channels, ctx.decay_input_type, ctx.imaging_modalities,
            source["duration"], source["time_bins"], folder_path, ctx.selected_ch_feature_extractors,
            source["fit_free_calibration_method"], source["fluorescence_lifetime_standard_lifetime"],
        )
        if error_msg != "":
            st.error(f"{error_msg} {sad_emoji}")
            return
        channels = {
            name: {
                "input_type": ctx.input_types[key],
                "imaging_modality": ctx.imaging_modalities[key],
                "selected_feature_extractors": ctx.selected_ch_feature_extractors[key],
                "num_components": source["selected_ch_num_components"].get(name, 0),
                "fixed_lifetimes": ctx.fixed_lifetimes[key],
                "qpi": ctx.qpi_constants.get(key, {}),
            }
            for key, name in selected_channels.items()
        }
        error_msg, settings = prepare_extraction(
            fov_df, channels, fov_name_col=ctx.fov_name_col,
            unique_cell_id_col=ctx.unique_cell_id_col,
            derived_features=ctx.derived_features, laser_rate=laser_rate,
            fit_free_calibration_method=source["fit_free_calibration_method"],
            fluorescence_lifetime_standard_lifetime=source["fluorescence_lifetime_standard_lifetime"],
        )
        if error_msg:
            st.error(error_msg)
            return
        label = "Start calibration" if settings["channels_shift"] or settings["channels_background"] else "Start extraction"
        if st.button(label, key="prepare_extraction_button", type="primary"):
            prepared = ExtractionSession.create(fov_df, settings, folder_path)
            st.session_state["prepared_extraction"] = prepared
            st.session_state["extraction_source"] = source_fingerprint(ctx, source)
            # Without calibration this click is the extraction start itself; a
            # failed metadata save still holds it until the retry succeeds.
            st.session_state["autostart_extraction"] = prepared.can_extract
            # Rerun so this button leaves the screen in the same interaction.
            st.rerun()


# --- Calibration and extraction -------------------------------------------
def _calibration_verb(shifts, backgrounds):
    if shifts and backgrounds:
        return "Calibrate channels", "recalibrate"
    if backgrounds:
        return "Correct background", "correct background"
    return "Optimize for Shifts", "find shift"


def _render_shift_controls(prepared):
    settings = prepared.settings
    if not prepared.calibration_confirmed:
        pending_shift, pending_background = pending_calibration(prepared.metadata_df, settings)
        if pending_shift:
            if any(settings["channels_shift"][channel] == "fit" for channel in pending_shift):
                fit_options_widget(settings)
            settings["fix_shift"] = st.checkbox(
                "Fix the Shift", value=settings.get("fix_shift", True), key="fix_shift_checkbox",
                help="Use one shift for all FOVs, or estimate a shift for each FOV.",
            )
            label, _ = _calibration_verb(pending_shift, pending_background)
            if st.button(label):
                prepared.choosing_shift = True
        elif pending_background:
            # QPI correction is cached and needs no shift-optimization gate.
            prepared.choosing_shift = True
        return False

    if "fitting_mode" in settings:
        modes = ["Hybrid", "Local"]
        mode = st.selectbox(
            "Fitting Mode", modes, index=modes.index(settings["fitting_mode"]),
            key="fitting_mode_update",
            help="Hybrid: global search then local refinement. Local: warm-start on mean decay then local fit per cell.",
        )
        prepared.change_mode(mode)
    start_col, back_col = st.columns(2)
    with start_col:
        start = st.button("Start extraction", width="stretch", disabled=not prepared.can_extract)
    with back_col:
        if settings["channels_shift"] or settings.get("channels_background"):
            _, verb = _calibration_verb(settings["channels_shift"], settings.get("channels_background"))
            if st.button(f"Go back and {verb}", width="stretch"):
                prepared.begin_recalibration()
                st.rerun()
    return start


def _render_choose_shift(prepared, ctx):
    pending_shift, pending_background = pending_calibration(prepared.metadata_df, prepared.settings)
    # One open expander per channel: a channel already inspected can be collapsed to
    # make room for the others. Pending order is stable, so a collapsed state survives
    # reruns, and the single Confirm below stays outside every block.
    channel_shifts = {}
    for channel in pending_shift:
        with st.expander(f"{channel}: shift calibration", expanded=True):
            error, shifts = choose_shift_widget(prepared.metadata_df, prepared.settings, ctx.fov_name_col, channel_name=channel)
            if error:
                st.error(f"{error} {sad_emoji}")
            else:
                channel_shifts[channel] = shifts
    channel_backgrounds = {}
    for channel in pending_background:
        with st.expander(f"{channel}: background correction", expanded=True):
            error, recipe = choose_background_widget(prepared.metadata_df, prepared.settings, ctx.fov_name_col, channel)
            if error:
                st.error(f"{error} {sad_emoji}")
            else:
                channel_backgrounds[channel] = recipe
    label = "Confirm calibration for each channel"
    if st.button(label):
        error = prepared.confirm_calibration(channel_shifts, channel_backgrounds)
        if prepared.calibration_confirmed:
            st.rerun()
        elif error:
            st.error(error)


def render_numeric_step(col1, col2, ctx):
    with col1:
        # Notify when another tab changes the config, above whichever view shows.
        notify_on_config_change()
    prepared = st.session_state.get("prepared_extraction")
    if prepared is None:
        render_preparation(col1, col2, ctx)
        return
    # A session exists: the calibration view, then the extraction view, take both
    # columns. The metadata settings stay editable in a collapsed expander; an
    # edit, a rescan, or a profile change discards the session and restores the
    # metadata view.
    with col1.expander("Metadata settings", expanded=False):
        error_msg, source = render_source_controls(ctx)
    if error_msg or source_fingerprint(ctx, source) != st.session_state.get("extraction_source"):
        invalidate_preparation()
        st.rerun()
    autostart = st.session_state.pop("autostart_extraction", False)
    with col1:
        start = _render_shift_controls(prepared) or autostart
    with col2:
        # Where the metadata view stood. Rendered after the calibration column so
        # a save that fails in this run is reported at once.
        _render_metadata_record(prepared)
        if prepared.choosing_shift:
            _render_choose_shift(prepared, ctx)
        elif start:
            # This is the final save gate. A failure cannot invoke extraction.
            error = prepared.before_extraction()
            if error:
                st.error(error)
            else:
                prepared.invalidate_results()
                features = fov_extraction_widget(prepared.metadata_df, prepared.settings)
                if not features.empty:
                    prepared.export_features(features)
                    if not prepared.features_error:
                        celebrate()
        if prepared.features is not None:
            st.write(prepared.features.head())
            if prepared.features_error:
                st.error(prepared.features_error)
                if st.button("Retry exporting features"):
                    prepared.save_features()
                    st.rerun()
            else:
                st.success(f"Single cell features exported successfully to {prepared.features_path} {happy_emoji}")


def render_categorical_step(col1, col2, ctx):
    """Step 2: scan a folder of CSVs (col1), then merge + assign categorical labels (col2)."""
    df_folder_path = ""
    delimiter = "_"
    available_dfs = []
    with col1:
        # Categorical features extraction
        df_folder_path = st.text_input("Copy the folder path here", help="The folder should contain all the csv files that you want to assign categories to.")
        delimiter = st.text_input("Field of View Name Delimiter", "_", max_chars=2, help="The delimiter used to split the fov_name column.")
        if df_folder_path != "":
            available_dfs = find_available_dfs_widget(df_folder_path, delimiter)
            if len(available_dfs) > 0:
                st.write(f"Found {len(available_dfs)} available csv files ready to be assigned categories {happy_emoji}:")
                st.write(available_dfs)
            else:
                st.error(f"No available csv files found at {df_folder_path} {sad_emoji}")

    with col2:
        if df_folder_path != "" and len(available_dfs) > 0:
            combined_df, available_categories = check_and_merge_df_widget(available_dfs)
            map_categories_to_labels_widget(available_categories, combined_df, delimiter, df_folder_path)


# --- Page controller -------------------------------------------------------
STEP_RENDERERS = {
    STEP_NUMERIC: render_numeric_step,
    STEP_CATEGORICAL: render_categorical_step,
}

st.set_page_config(layout="wide", page_icon="🔬")
render_top_menu(space_below="0.5rem")
st.session_state.setdefault("prepared_extraction", None)

ctx = build_context()
preserve_source_controls(ctx)
config_identity = (get_current_profile_name(), get_config_mtime())
if st.session_state.get("extraction_config_identity") != config_identity:
    invalidate_preparation()
    st.session_state["extraction_config_identity"] = config_identity
# An unsaved profile can have no channels; stop before creating channel columns.
if not ctx.channel_names:
    st.warning(
        f"The active configuration profile **'{get_current_profile_name()}'** has not "
        "been configured yet. Please go to **Home / Configuration** page, "
        "configure this profile, then click "
        "**Update Configuration** — and come back here."
    )
    st.stop()

col1, col2 = st.columns([0.4, 1])
with col1:
    # first select the step to perform
    selected_step = st.radio(
        "Select a step to extract single-object features",
        STEPS,
        index=0,
        help=(
            "**Numerical**: prepare a source folder, calibrate, and extract per-object "
            "measurements (e.g. lifetime, morphology); the features are exported "
            "automatically. **Categorical**: combine numerical datasets and label the "
            "exported objects (e.g. treatment, day)."
        ),
    )

STEP_RENDERERS[selected_step](col1, col2, ctx)
