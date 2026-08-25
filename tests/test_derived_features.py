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
from src.feature_schema import predict_feature_columns
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
