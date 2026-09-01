"""Derived features: features built from arithmetic over existing extracted ones.

Covers the four pieces of the feature:
  1. compute_derived_features()  — safe AST evaluation, divide-by-zero, skips.
  2. predict_feature_columns()   — operand-schema prediction incl. fit expansion.
  3. analysis grouping + labels  — single "Derived Features" group + clean label.
  4. metadata round-trip         — the JSON column bakes definitions into the CSV
                                    so a replayed metadata CSV is self-contained.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.derived_features import compute_derived_features, evaluate_expression, alias_names, is_single_operand
from src.feature_schema import (
    predict_feature_columns,
    predict_feature_columns_from_cfg,
    predict_uncategorized_columns,
)
from src.feature_labels import format_feature_label
import src.dataset_io as dataset_io
from src.dataset_io import get_feature_groups_data_extraction
import src.metadata as metadata_mod
from src.metadata import parse_metadata_file


# --------------------------------------------------------------------------- #
# 1. compute_derived_features
# --------------------------------------------------------------------------- #

def _redox_df():
    return pd.DataFrame(
        {
            "Lifetime fit_fad: a1": [10.0, 20.0, 0.0],
            "Lifetime fit_nadh: a1": [30.0, 0.0, 0.0],
        },
        index=["fov1_1", "fov1_2", "fov1_3"],
    )


def test_arithmetic_with_parentheses():
    defs = [{
        "name": "redox_ratio",
        "expression": "A/(A+B)",
        "operands": ["Lifetime fit_fad: a1", "Lifetime fit_nadh: a1"],
    }]
    out, warnings = compute_derived_features(_redox_df(), defs)
    assert warnings == []
    # 10/(10+30)=0.25 ; 20/(20+0)=1.0 ; 0/0 -> NaN
    assert out["Derived: redox_ratio"].iloc[0] == pytest.approx(0.25)
    assert out["Derived: redox_ratio"].iloc[1] == pytest.approx(1.0)
    assert np.isnan(out["Derived: redox_ratio"].iloc[2])
    # index (cell_id) is preserved through the arithmetic + assignment
    assert list(out.index) == ["fov1_1", "fov1_2", "fov1_3"]


@pytest.mark.parametrize(
    "expression, expected",
    [
        ("A+B", [40.0, 20.0, 0.0]),
        ("A-B", [-20.0, 20.0, 0.0]),
        ("A*B", [300.0, 0.0, 0.0]),
        ("-A", [-10.0, -20.0, 0.0]),
        ("A*2+B", [50.0, 40.0, 0.0]),
    ],
)
def test_operators(expression, expected):
    defs = [{
        "name": "x",
        "expression": expression,
        "operands": ["Lifetime fit_fad: a1", "Lifetime fit_nadh: a1"],
    }]
    out, warnings = compute_derived_features(_redox_df(), defs)
    assert warnings == []
    assert out["Derived: x"].tolist() == expected


def test_divide_by_zero_is_nan_not_inf():
    df = pd.DataFrame({"Lifetime fit_fad: a1": [5.0], "Lifetime fit_nadh: a1": [0.0]})
    defs = [{"name": "r", "expression": "A/B",
             "operands": ["Lifetime fit_fad: a1", "Lifetime fit_nadh: a1"]}]
    out, _ = compute_derived_features(df, defs)
    assert np.isnan(out["Derived: r"].iloc[0])
    assert not np.isinf(out["Derived: r"].iloc[0])


def test_missing_operand_is_skipped_with_warning():
    defs = [{"name": "bad", "expression": "A/B", "operands": ["nope_a", "nope_b"]}]
    out, warnings = compute_derived_features(_redox_df(), defs)
    assert "Derived: bad" not in out.columns
    assert len(warnings) == 1
    assert "nope_a" in warnings[0] and "not found" in warnings[0]


def test_referenced_missing_operand_reports_column():
    # A exists, B's column is missing and IS referenced -> skip, report B's column.
    defs = [{"name": "r", "expression": "A/B",
             "operands": ["Lifetime fit_fad: a1", "missing_col"]}]
    out, warnings = compute_derived_features(_redox_df(), defs)
    assert "Derived: r" not in out.columns
    assert "missing_col" in warnings[0]


def test_unused_missing_operand_does_not_block():
    # C's column is missing but the expression never references it -> still computed.
    defs = [{"name": "sum_ab", "expression": "A+B",
             "operands": ["Lifetime fit_fad: a1", "Lifetime fit_nadh: a1", "missing_col"]}]
    out, warnings = compute_derived_features(_redox_df(), defs)
    assert "Derived: sum_ab" in out.columns
    assert warnings == []


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('echo hi')",  # Call + Attribute
        "foo(A)",                              # Call
        "(lambda: A)()",                       # Lambda + Call
        "A.__class__",                         # Attribute
        "A[0]",                                # Subscript
        "A ** B",                              # Pow — not whitelisted
        "A % B",                               # Mod — not whitelisted
        "A > B",                               # Compare
        "A and B",                             # BoolOp
        "A if B else 0",                       # conditional
        "True",                                # bare bool Constant
        "C",                                   # references an alias with no operand
    ],
)
def test_unsafe_or_invalid_expression_rejected(expression):
    defs = [{"name": "u", "expression": expression,
             "operands": ["Lifetime fit_fad: a1", "Lifetime fit_nadh: a1"]}]
    out, warnings = compute_derived_features(_redox_df(), defs)
    assert "Derived: u" not in out.columns
    assert len(warnings) == 1


def test_evaluate_expression_rejects_unsafe_nodes_directly():
    """Directly exercise the AST evaluator (not via compute) for a few node types."""
    series = {"A": pd.Series([1.0, 2.0]), "B": pd.Series([3.0, 4.0])}
    for bad in ("foo(A)", "A.real", "A[0]", "A ** B", "A > B", "A and B", "__import__('os')"):
        with pytest.raises(ValueError):
            evaluate_expression(bad, series)


def test_is_single_operand_flags_bare_aliases_only():
    """A lone operand (incl. parenthesized) is trivial; real formulas are not."""
    assert is_single_operand("A")
    assert is_single_operand("B")
    assert is_single_operand("(C)")
    assert is_single_operand("  A  ")  # surrounding whitespace is ignored by the parser
    for real in ("A/B", "A+B", "A-B", "A*2", "A/(A+B)", "-A", "A+A"):
        assert not is_single_operand(real)
    assert not is_single_operand("")          # unparseable -> not a bare operand
    assert not is_single_operand("A +")       # syntax error -> False, not a crash


def test_evaluate_expression_precedence_and_nesting():
    """AST honours operator precedence, associativity, unary +/-, nested parens."""
    a = pd.Series([12.0, 8.0])
    b = pd.Series([4.0, 2.0])
    series = {"A": a, "B": b}
    # * binds tighter than + ; associativity of - and / is left-to-right
    assert evaluate_expression("A + B * 2", series).tolist() == [20.0, 12.0]
    assert evaluate_expression("A - B - 1", series).tolist() == [7.0, 5.0]
    assert evaluate_expression("A / B / 2", series).tolist() == [1.5, 2.0]
    # nested parentheses + unary minus + float constant
    assert evaluate_expression("-(A - (B + 0.5))", series).tolist() == [-7.5, -5.5]
    assert evaluate_expression("+A", series).tolist() == [12.0, 8.0]


def test_empty_name_or_expression_skipped():
    out, warnings = compute_derived_features(
        _redox_df(),
        [{"name": "", "expression": "A", "operands": ["Lifetime fit_fad: a1"]}],
    )
    assert len(warnings) == 1
    assert not any(c.startswith("Derived:") for c in out.columns)


def test_no_defs_returns_frame_unchanged():
    df = _redox_df()
    out, warnings = compute_derived_features(df, [])
    assert warnings == []
    assert list(out.columns) == list(df.columns)


def test_evaluate_expression_raises_valueerror_on_syntax_error():
    with pytest.raises(ValueError):
        evaluate_expression("A +", {"A": pd.Series([1.0])})


def test_alias_names():
    assert alias_names(0) == []
    assert alias_names(3) == ["A", "B", "C"]


# --------------------------------------------------------------------------- #
# 2. predict_feature_columns
# --------------------------------------------------------------------------- #

def test_predict_fit_phasor_texture_profile():
    cols = predict_feature_columns(
        ["nadh", "fad"],
        {"nadh": ["Lifetime fit", "Lifetime fit free"], "fad": ["Lifetime fit", "Intensity texture"]},
        {"nadh": 2, "fad": 2},
    )
    # cross-channel a1 operands (the redox-ratio use case)
    assert "Lifetime fit_nadh: a1" in cols
    assert "Lifetime fit_fad: a1" in cols
    # phasor coordinates
    assert "Lifetime fit free_nadh: G(1st)" in cols
    assert "Lifetime fit free_nadh: Tau_mod" in cols
    # texture
    assert "Intensity texture_fad: granularity_3" in cols
    assert "Intensity texture_fad: mass_displacement" in cols
    # no duplicates
    assert len(cols) == len(set(cols))
    # uncategorized bookkeeping columns are NOT offered as operands
    assert not any(c.endswith(": amp1") for c in cols)
    assert not any("reduced_chi_square" in c for c in cols)


@pytest.mark.parametrize(
    "n, expected_suffixes",
    [
        (1, ["t1"]),
        (2, ["t1", "t2", "a1", "tm", "tm_iw"]),
        (3, ["t1", "t2", "t3", "a1", "a2", "tm", "tm_iw"]),
    ],
)
def test_fit_component_expansion(n, expected_suffixes):
    cols = predict_feature_columns(["nadh"], {"nadh": ["Lifetime fit"]}, {"nadh": n})
    got = [c.split(": ", 1)[1] for c in cols]
    assert got == expected_suffixes


def test_predict_empty_when_no_extractors():
    assert predict_feature_columns(["nadh"], {"nadh": []}, {"nadh": 2}) == []


def test_predict_texture_2d_is_intensity_sum_only():
    # Decay (2D) has no image/mask, so "Intensity texture" emits ONLY intensity_sum
    # (the decay-curve sum). Mirrors extract_intensity_features' 2D branch.
    cols = predict_feature_columns(
        ["cyto"],
        {"cyto": ["Lifetime fit free", "Intensity texture"]},
        {"cyto": 1},
        {"cyto": "Decay (2D)"},
    )
    assert "Intensity texture_cyto: intensity_sum" in cols
    # image-only texture features must NOT be predicted for 2D
    assert not any("granularity" in c for c in cols)
    assert not any("radial_distribution" in c for c in cols)
    assert not any("mass_displacement" in c for c in cols)
    # phasor coordinates from the co-selected fit-free extractor are unaffected
    assert "Lifetime fit free_cyto: G(1st)" in cols


def test_predict_texture_2d_absent_without_texture_extractor():
    # A 2D channel selecting only fit-free must NOT predict intensity_sum — it is no
    # longer auto-tucked into the fit-free path; it requires "Intensity texture".
    cols = predict_feature_columns(
        ["cyto"],
        {"cyto": ["Lifetime fit free"]},
        {"cyto": 1},
        {"cyto": "Decay (2D)"},
    )
    assert not any("intensity_sum" in c for c in cols)


def test_predict_texture_non_2d_keeps_full_list():
    # Non-2D (e.g. 3/4D image-based) texture still predicts the full suffix list.
    cols = predict_feature_columns(
        ["cyto"],
        {"cyto": ["Intensity texture"]},
        {"cyto": 1},
        {"cyto": "Decay (3/4D)"},
    )
    assert "Intensity texture_cyto: intensity_sum" in cols
    assert "Intensity texture_cyto: granularity_3" in cols
    assert "Intensity texture_cyto: mass_displacement" in cols


# --------------------------------------------------------------------------- #
# 2b. predict_uncategorized_columns
#
# The bookkeeping columns fov_extraction emits WITHOUT an extractor prefix
# ("{channel}_{suffix}"). They are ordinary per-cell numbers, so they are offered
# as operands too — background-correcting an intensity needs the fit's offset.
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "n, expected",
    [
        (1, ["amp1", "offset", "reduced_chi_square"]),
        (2, ["amp1", "offset", "reduced_chi_square", "amp2", "a2"]),
        (3, ["amp1", "offset", "reduced_chi_square", "amp2", "amp3", "a3"]),
    ],
)
def test_uncategorized_fit_columns_track_component_count(n, expected):
    # Mirrors extract_fit_results: amp1/offset/reduced_chi_square always, one amp
    # per extra component, and the a{n} remainder that completes the percentages.
    cols = predict_uncategorized_columns(["nadh"], {"nadh": ["Lifetime fit"]}, {"nadh": n})
    assert cols == [f"nadh_{s}" for s in expected]


@pytest.mark.parametrize("n, expected", [(1, []), (2, ["nadh_a2"]), (3, ["nadh_a3"])])
def test_uncategorized_prefitted_fit_is_remainder_only(n, expected):
    # extract_spcimage_fit_results re-reads someone else's per-pixel fit, so there
    # is no amplitude, offset or chi-square of our own to report.
    cols = predict_uncategorized_columns(
        ["nadh"], {"nadh": ["Lifetime fit"]}, {"nadh": n},
        {"nadh": "Decay (3/4D) pixel-prefitted"},
    )
    assert cols == expected


def test_uncategorized_morphology_is_the_centroids():
    cols = predict_uncategorized_columns(
        ["nadh"], {"nadh": ["Intensity morphology"]}, {"nadh": 1})
    assert cols == ["nadh_centroid_x", "nadh_centroid_y"]


def test_uncategorized_empty_for_texture_and_phasor():
    # Neither extractor emits an unprefixed column.
    assert predict_uncategorized_columns(
        ["nadh"], {"nadh": ["Intensity texture", "Lifetime fit free"]}, {"nadh": 1}) == []


def test_uncategorized_dedupes_across_extractors():
    cols = predict_uncategorized_columns(
        ["nadh", "fad"],
        {"nadh": ["Lifetime fit", "Intensity morphology"], "fad": ["Intensity morphology"]},
        {"nadh": 2, "fad": 2},
    )
    assert cols == [
        "nadh_amp1", "nadh_offset", "nadh_reduced_chi_square", "nadh_amp2", "nadh_a2",
        "nadh_centroid_x", "nadh_centroid_y",
        "fad_centroid_x", "fad_centroid_y",
    ]
    assert len(cols) == len(set(cols))


_UNCAT_CFG = {
    "num_channels": 1,
    "ch1": {
        "channel_name": "nadh",
        "input_type": "Decay (3/4D)",
        "Decay (3/4D)": {
            "selected_feature_extractors": ["Lifetime fit", "Intensity texture"],
            "num_components": 2,
        },
    },
}


def test_from_cfg_excludes_uncategorized_by_default():
    cols = predict_feature_columns_from_cfg(_UNCAT_CFG)
    assert "Lifetime fit_nadh: t1" in cols
    assert "nadh_offset" not in cols


def test_from_cfg_appends_uncategorized_after_the_measurements():
    cols = predict_feature_columns_from_cfg(_UNCAT_CFG, include_uncategorized=True)
    assert "nadh_offset" in cols
    # the background-correction use case: intensity_sum - offset * time_bins
    assert "Intensity texture_nadh: intensity_sum" in cols
    # measurements stay at the top of the picker
    assert cols.index("Intensity texture_nadh: intensity_sum") < cols.index("nadh_offset")
    assert cols[: len(predict_feature_columns_from_cfg(_UNCAT_CFG))] == \
        predict_feature_columns_from_cfg(_UNCAT_CFG)


def test_uncategorized_operand_evaluates_end_to_end():
    # The evaluator never cared about the prefix — only the picker did. A literal
    # (the time-bin count) is already allowed, which is what makes this formula work.
    df = pd.DataFrame({
        "Intensity texture_nadh: intensity_sum": [1000.0, 2000.0],
        "nadh_offset": [1.0, 2.0],
    })
    defs = [{
        "name": "intensity_bg_corrected",
        "expression": "A - B*256",
        "operands": ["Intensity texture_nadh: intensity_sum", "nadh_offset"],
    }]
    out, warnings = compute_derived_features(df, defs)
    assert warnings == []
    assert out["Derived: intensity_bg_corrected"].tolist() == [1000 - 256, 2000 - 512]


# --------------------------------------------------------------------------- #
# 2c. predictor <-> emitter cross-check
#
# feature_schema PREDICTS, from config alone, what fov_extraction EMITS — two
# lists that drift apart silently. A column the emitter adds but the schema
# forgets is an operand the picker never offers; one the schema invents is an
# operand whose derived feature is skipped at extraction time with "operand
# column(s) not found". These run the REAL emitters on synthetic input and
# compare the unprefixed columns they produce against the prediction, so the
# coupling is checked by measurement rather than by a hand-copied list.
# --------------------------------------------------------------------------- #

CH = "nadh"
_IRF = np.zeros(8)
_IRF[0] = 1.0
_TIME_AXIS = np.arange(8, dtype=float)
_DECAY = np.array([100.0, 50.0, 25.0, 12.0, 6.0, 3.0, 2.0, 1.0])
_FIT_RESULTS = {
    "amp1": [80.0, 60.0], "amp2": [20.0, 40.0], "amp3": [10.0, 5.0],
    "t1": [0.4, 0.5], "t2": [2.5, 2.2], "t3": [4.0, 3.8],
    "offset": [5.0, 7.0],
}
_CELLS = ["fov_1_1", "fov_1_2"]


def _unprefixed(emitted):
    """The columns with no ``"{Extractor}_{channel}: "`` prefix — the uncategorized ones."""
    columns = emitted.columns if isinstance(emitted, pd.DataFrame) else {
        col for per_cell in emitted.values() for col in per_cell
    }
    return sorted(c for c in columns if ": " not in c)


def _two_cell_mask():
    mask = np.zeros((12, 12), dtype=int)
    mask[2:6, 2:6] = 1
    mask[7:11, 7:11] = 2
    return mask


@pytest.mark.parametrize("n", [1, 2, 3])
def test_fit_emitter_matches_predicted_uncategorized(n):
    from src.fov_extraction import extract_fit_results

    _, emitted = extract_fit_results(
        channel_name=CH, decay_curves={c: _DECAY for c in _CELLS}, results=_FIT_RESULTS,
        num_components=n, shifted_irf=_IRF, time_axis=_TIME_AXIS,
        start=0, end=8, fixed_lifetimes=None)
    assert _unprefixed(emitted) == sorted(predict_uncategorized_columns(
        [CH], {CH: ["Lifetime fit"]}, {CH: n}, {CH: "Decay (3/4D)"}))


@pytest.mark.parametrize("n", [1, 2, 3])
def test_spcimage_emitter_matches_predicted_uncategorized(n, monkeypatch):
    from src.fov_extraction import extract_spcimage_fit_results

    images = {
        "m": np.array([[1, 1], [2, 2]]),
        "t1": np.full((2, 2), 0.4), "a1": np.full((2, 2), 80.0),
        "t2": np.full((2, 2), 2.5), "a2": np.full((2, 2), 15.0), "t3": np.full((2, 2), 4.0),
    }
    monkeypatch.setattr("src.fov_extraction.load_image", lambda path: images[path])
    metadata = {f"{CH}_Mask": "m", f"{CH}_SPCImage t1": "t1", f"{CH}_a1": "a1",
                f"{CH}_t2": "t2", f"{CH}_a2": "a2", f"{CH}_t3": "t3", "fov": "fov_1"}

    err, emitted = extract_spcimage_fit_results(metadata, CH, n, "fov")
    assert err == ""
    assert _unprefixed(emitted) == sorted(predict_uncategorized_columns(
        [CH], {CH: ["Lifetime fit"]}, {CH: n}, {CH: "Decay (3/4D) pixel-prefitted"}))


def test_morphology_emitter_matches_predicted_uncategorized():
    from src.fov_extraction import get_intensity_morphology_features

    err, emitted = get_intensity_morphology_features(
        {"fov": "fov_1"}, CH, "fov", _two_cell_mask())
    assert err == ""
    assert _unprefixed(emitted) == sorted(predict_uncategorized_columns(
        [CH], {CH: ["Intensity morphology"]}, {CH: 1}))


def test_texture_and_phasor_emit_nothing_uncategorized():
    """Both are fully prefixed — the prediction must not invent operands for them."""
    from src.fov_extraction import (
        extract_fit_free_results,
        extract_intensity_sum_2d,
        extract_texture_features_from_arrays,
    )

    image = np.random.default_rng(0).random((12, 12)) * 100
    texture = extract_texture_features_from_arrays(
        image, _two_cell_mask(), "fov_1", f"Intensity texture_{CH}: ")
    assert _unprefixed(texture) == []

    texture_2d = extract_intensity_sum_2d(CH, {c: _DECAY for c in _CELLS},
                                          {c: {} for c in _CELLS})
    assert _unprefixed(texture_2d) == []

    err, phasor = extract_fit_free_results(
        CH, {c: _DECAY for c in _CELLS}, laser_rate=0.08, duration=8.0,
        calibration_method="IRF", shifted_irf=_IRF)
    assert err == ""
    assert _unprefixed(phasor) == []
    assert predict_uncategorized_columns(
        [CH], {CH: ["Intensity texture", "Lifetime fit free"]}, {CH: 1}) == []


def _extracted_frame():
    """A frame assembled the way ``fov_extraction`` assembles one: real emitter
    output for fit + morphology + texture, concatenated, plus the FOV column."""
    from src.fov_extraction import (
        extract_fit_results,
        extract_texture_features_from_arrays,
        get_intensity_morphology_features,
    )

    _, fit_feats = extract_fit_results(
        channel_name=CH, decay_curves={c: _DECAY for c in _CELLS}, results=_FIT_RESULTS,
        num_components=2, shifted_irf=_IRF, time_axis=_TIME_AXIS,
        start=0, end=8, fixed_lifetimes=None)
    _, morph = get_intensity_morphology_features({"fov": "fov_1"}, CH, "fov", _two_cell_mask())
    image = np.random.default_rng(0).random((12, 12)) * 100
    texture = extract_texture_features_from_arrays(
        image, _two_cell_mask(), "fov_1", f"Intensity texture_{CH}: ")

    df = pd.concat([pd.DataFrame.from_dict(fit_feats, orient="index"), morph, texture], axis=1)
    df["image_name"] = "fov_1"  # added by fov_extraction; an identifier, never an operand
    return df


def test_every_uncategorized_operand_is_present_and_computable():
    """Not just offered — each one has to survive compute_derived_features on a real
    extracted frame. A predicted-but-absent column would be skipped there with a
    warning instead of producing a column."""
    df = _extracted_frame()
    uncategorized = predict_uncategorized_columns(
        [CH], {CH: ["Lifetime fit", "Intensity morphology", "Intensity texture"]},
        {CH: 2}, {CH: "Decay (3/4D)"})
    assert uncategorized, "no uncategorized operands predicted for this profile"

    for col in uncategorized:
        assert col in df.columns, f"{col} is offered as an operand but never extracted"
        out, warnings = compute_derived_features(
            df.copy(), [{"name": "chk", "expression": "A*2", "operands": [col]}])
        assert warnings == [], f"{col}: {warnings}"
        assert np.allclose(out["Derived: chk"].astype(float),
                           df[col].astype(float) * 2, equal_nan=True)


def test_background_corrected_intensity_over_real_extractor_output():
    """The motivating formula, on columns two different extractors actually produced:
    intensity_sum (texture) minus offset (fit) times the time-bin count."""
    df = _extracted_frame()
    intensity, offset = f"Intensity texture_{CH}: intensity_sum", f"{CH}_offset"
    out, warnings = compute_derived_features(df.copy(), [{
        "name": "intensity_bg", "expression": "A - B*256", "operands": [intensity, offset],
    }])
    assert warnings == []
    assert np.allclose(out["Derived: intensity_bg"], df[intensity] - df[offset] * 256)


# --------------------------------------------------------------------------- #
# 3. analysis grouping + labels
# --------------------------------------------------------------------------- #

def test_derived_columns_form_single_group(monkeypatch):
    monkeypatch.setattr(
        dataset_io, "get_all_feature_extractors",
        lambda: ["Lifetime fit", "Lifetime fit free", "Intensity morphology", "Intensity texture"],
    )
    cols = [
        "Lifetime fit_nadh: a1",
        "Lifetime fit_fad: a1",
        "Derived: redox_ratio",
        "Derived: fad_over_nadh",
        "some_random_col",
    ]
    groups = get_feature_groups_data_extraction(cols)
    assert groups["Derived Features"] == ["Derived: redox_ratio", "Derived: fad_over_nadh"]
    # the per-channel fit group is untouched by the derived branch
    assert groups["Lifetime fit_nadh"] == ["Lifetime fit_nadh: a1"]
    # non-derived, non-extractor column still lands in Uncategorized
    assert "some_random_col" in groups["Uncategorized Features"]


def test_format_feature_label_derived():
    assert format_feature_label("Derived: redox_ratio") == "redox_ratio"
    # name containing spaces / underscores is returned verbatim (no unit)
    assert format_feature_label("Derived: fad over nadh") == "fad over nadh"


# --------------------------------------------------------------------------- #
# 4. metadata round-trip (self-contained, replayable CSV)
# --------------------------------------------------------------------------- #

def _patch_metadata_config(monkeypatch):
    monkeypatch.setattr(
        metadata_mod, "get_available_feature_extractors",
        lambda input_type: ["Lifetime fit", "Lifetime fit free", "Intensity morphology", "Intensity texture"],
    )
    monkeypatch.setattr(metadata_mod, "get_fov_name_col", lambda: "image_name")
    monkeypatch.setattr(metadata_mod, "get_unique_cell_id_col", lambda: "cell_id")
    # Intensity-only + no file-type columns => no file-existence checks needed.
    monkeypatch.setattr(metadata_mod, "get_file_types", lambda input_type: [])


def _intensity_only_metadata(extra_cols=None):
    data = {
        "image_name": ["fov1"],
        "ch1_input_type": ["Intensity (2D)"],
        "ch1_imaging_modality": ["Intensity-only"],
        "ch1_Intensity morphology": [True],
    }
    if extra_cols:
        data.update(extra_cols)
    return pd.DataFrame(data)


def test_derived_features_round_trip(monkeypatch, tmp_path):
    _patch_metadata_config(monkeypatch)
    defs = [
        {"name": "redox_ratio", "expression": "A/(A+B)",
         "operands": ["Lifetime fit_fad: a1", "Lifetime fit_nadh: a1"]},
        {"name": "diff", "expression": "A-B",
         "operands": ["Lifetime fit_fad: t1", "Lifetime fit_nadh: t1"]},
    ]
    df = _intensity_only_metadata({"derived_features": [json.dumps(defs)]})

    # Round-trips through CSV (JSON cell has commas/colons/quotes; pandas quotes it).
    csv_path = tmp_path / "meta.csv"
    df.to_csv(csv_path, index=False)
    reloaded = pd.read_csv(csv_path, index_col=False, low_memory=False)

    err, md = parse_metadata_file(reloaded, "image_name")
    assert err == ""
    assert md["derived_features"] == defs


def test_missing_derived_features_column_defaults_empty(monkeypatch):
    _patch_metadata_config(monkeypatch)
    err, md = parse_metadata_file(_intensity_only_metadata(), "image_name")
    assert err == ""
    assert md["derived_features"] == []


def test_json_definition_dump_parse_is_lossless():
    """json.dumps <-> json.loads preserves definitions exactly, including operand
    names that contain ': ', '(', ')' and the list commas the CSV cell must hold."""
    defs = [
        {"name": "redox_ratio", "expression": "A/(A+B)",
         "operands": ["Lifetime fit_fad: a1", "Lifetime fit_nadh: a1"]},
        {"name": "g_minus_s", "expression": "A-B",
         "operands": ["Lifetime fit free_nadh: G(1st)", "Lifetime fit free_nadh: S(1st)"]},
    ]
    assert json.loads(json.dumps(defs)) == defs
    assert json.loads(json.dumps([])) == []


def test_json_column_survives_csv_quoting(tmp_path):
    """The repeated-per-row JSON column round-trips through pandas to_csv/read_csv
    despite commas/colons inside the JSON (this is how the metadata CSV stores it)."""
    defs = [{"name": "r", "expression": "A/(A+B)",
             "operands": ["Lifetime fit_fad: a1", "Lifetime fit_nadh: a1"]}]
    df = pd.DataFrame({
        "image_name": ["fov1", "fov2"],
        "derived_features": [json.dumps(defs)] * 2,
    })
    path = tmp_path / "meta.csv"
    df.to_csv(path, index=False)
    back = pd.read_csv(path, index_col=False, low_memory=False)
    assert back["derived_features"].nunique() == 1  # global column, identical per row
    assert json.loads(back["derived_features"].iloc[0]) == defs


def test_parse_tolerates_malformed_json(monkeypatch):
    """A hand-corrupted derived_features cell degrades to [] rather than crashing."""
    _patch_metadata_config(monkeypatch)
    df = _intensity_only_metadata({"derived_features": ["{not valid json"]})
    err, md = parse_metadata_file(df, "image_name")
    assert err == ""
    assert md["derived_features"] == []
