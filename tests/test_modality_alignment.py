import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Skip when the optional modality-alignment module is absent from this checkout.
pytest.importorskip(
    "src.modality_alignment",
    reason="src/modality_alignment.py is only on the modality-alignment branch",
)

from src.modality_alignment import (
    apply_filter_to_alignment_data,
    classify_alignment_prep_issue,
    infer_alignment_id_columns,
    infer_shared_categorical_columns,
    prepare_alignment_data,
    run_scot_alignment,
)


def _build_input_frames():
    df_a = pd.DataFrame(
        {
            "cell_id": ["a-2", "a-1", "a-3"],
            "cell_type": ["T", "T", "B"],
            "treatment": ["ctrl", "ctrl", "drug"],
            "feat_a1": [2.0, 1.0, 3.0],
            "feat_a2": [20.0, 10.0, 30.0],
        }
    )
    df_b = pd.DataFrame(
        {
            "sample_id": ["b-1", "b-2", "b-3"],
            "cell_type": ["T", "T", "B"],
            "treatment": ["ctrl", "ctrl", "drug"],
            "feat_b1": [100.0, 200.0, 300.0],
            "feat_b2": [1000.0, 2000.0, 3000.0],
        }
    )
    return df_a, df_b


def test_infer_alignment_id_columns_picks_best_unique_non_numeric_column_per_csv():
    df_a = pd.DataFrame(
        {
            "row_num": [1, 2, 3],
            "cell_id": ["a1", "a2", "a3"],
            "cell_type": ["T", "T", "B"],
        }
    )
    df_b = pd.DataFrame(
        {
            "row_num": [11, 12, 13],
            "sample_id": ["b1", "b2", "b3"],
            "cell_type": ["T", "T", "B"],
        }
    )

    inferred_a, inferred_b = infer_alignment_id_columns(df_a, df_b)

    assert inferred_a == "cell_id"
    assert inferred_b == "sample_id"


def test_infer_alignment_id_columns_accepts_unpaired_modalities():
    df_a = pd.DataFrame(
        {
            "cell_id": ["A549_FCCP_A1_N1_1", "A549_FCCP_A1_N1_2", "A549_FCCP_A1_N1_3"],
            "cell_type": ["epithelial", "epithelial", "epithelial"],
        }
    )
    df_b = pd.DataFrame(
        {
            "cell_id": ["A549_CCCP_02_100", "A549_CCCP_02_105", "A549_CCCP_02_109"],
            "cell_type": ["epithelial", "epithelial", "epithelial"],
        }
    )

    inferred_a, inferred_b = infer_alignment_id_columns(df_a, df_b)

    assert inferred_a == "cell_id"
    assert inferred_b == "cell_id"


def test_infer_alignment_id_columns_rejects_when_csv_lacks_unique_non_numeric_column():
    df_a = pd.DataFrame(
        {
            "cell_id": ["a1", "a2", "a3"],
            "cell_type": ["T", "T", "B"],
        }
    )
    df_b = pd.DataFrame(
        {
            "sample_id": ["b1", "b1", "b2"],
            "row_num": [1, 2, 3],
        }
    )

    with pytest.raises(ValueError, match="CSV B does not have a unique non-numeric column"):
        infer_alignment_id_columns(df_a, df_b)


def test_infer_shared_categorical_columns_uses_all_shared_non_numeric_columns():
    df_a = pd.DataFrame(
        {
            "cell_id": ["c1", "c2"],
            "cell_type": ["T", "B"],
            "treatment": ["ctrl", "drug"],
            "shared_numeric": [1.0, 2.0],
            "feat_only_a": [10.0, 11.0],
        }
    )
    df_b = pd.DataFrame(
        {
            "sample_id": ["z1", "z2"],
            "cell_type": ["T", "B"],
            "treatment": ["ctrl", "drug"],
            "shared_numeric": [5.0, 6.0],
            "feat_only_b": [20.0, 21.0],
        }
    )

    inferred = infer_shared_categorical_columns(df_a, df_b, "cell_id", "sample_id")

    assert inferred == ["cell_type", "treatment"]


def test_classify_alignment_prep_issue_keeps_validation_failures_as_errors():
    severity, message = classify_alignment_prep_issue(
        ValueError("At least two rows must remain in each modality after filtering.")
    )

    assert severity == "error"
    assert "At least two rows must remain in each modality" in message


def test_prepare_alignment_data_keeps_modalities_unpaired_and_builds_filter_df():
    df_a, df_b = _build_input_frames()

    alignment_data = prepare_alignment_data(
        df_a,
        df_b,
        id_col_a="cell_id",
        id_col_b="sample_id",
        categorical_cols=["cell_type", "treatment"],
        features_a=["feat_a1", "feat_a2"],
        features_b=["feat_b1", "feat_b2"],
    )

    assert alignment_data.cell_ids_a == ["a-2", "a-1", "a-3"]
    assert alignment_data.cell_ids_b == ["b-1", "b-2", "b-3"]
    assert alignment_data.features_a.index.tolist() == alignment_data.cell_ids_a
    assert alignment_data.features_b.index.tolist() == alignment_data.cell_ids_b
    assert set(alignment_data.filter_df.columns) == {
        "cell_id",
        "modality",
        "cell_type",
        "treatment",
        "A__feat_a1",
        "A__feat_a2",
        "B__feat_b1",
        "B__feat_b2",
    }
    assert alignment_data.filter_df["modality"].tolist() == [
        "Modality A",
        "Modality A",
        "Modality A",
        "Modality B",
        "Modality B",
        "Modality B",
    ]
    assert alignment_data.dropped_na_rows_a == 0
    assert alignment_data.dropped_na_rows_b == 0


def test_prepare_alignment_data_normalizes_id_columns_independently():
    df_a = pd.DataFrame(
        {
            "cell_id": [1, 2, 3],
            "cell_type": ["T", "T", "B"],
            "feat_a1": [2.0, 1.0, 3.0],
            "feat_a2": [20.0, 10.0, 30.0],
        }
    )
    df_b = pd.DataFrame(
        {
            "sample_id": [" b1 ", "b2", "b3 "],
            "cell_type": ["T", "T", "B"],
            "feat_b1": [100.0, 200.0, 300.0],
            "feat_b2": [1000.0, 2000.0, 3000.0],
        }
    )

    alignment_data = prepare_alignment_data(
        df_a,
        df_b,
        id_col_a="cell_id",
        id_col_b="sample_id",
        categorical_cols=["cell_type"],
        features_a=["feat_a1", "feat_a2"],
        features_b=["feat_b1", "feat_b2"],
    )

    assert alignment_data.cell_ids_a == ["1", "2", "3"]
    assert alignment_data.cell_ids_b == ["b1", "b2", "b3"]


def test_prepare_alignment_data_drops_nan_rows_per_modality():
    df_a, df_b = _build_input_frames()
    df_a.loc[df_a["cell_id"] == "a-1", "feat_a1"] = np.nan
    df_b.loc[df_b["sample_id"] == "b-3", "feat_b2"] = np.nan

    alignment_data = prepare_alignment_data(
        df_a,
        df_b,
        id_col_a="cell_id",
        id_col_b="sample_id",
        categorical_cols=["cell_type"],
        features_a=["feat_a1", "feat_a2"],
        features_b=["feat_b1", "feat_b2"],
    )

    assert alignment_data.cell_ids_a == ["a-2", "a-3"]
    assert alignment_data.cell_ids_b == ["b-1", "b-2"]
    assert alignment_data.dropped_na_rows_a == 1
    assert alignment_data.dropped_na_rows_b == 1


def test_prepare_alignment_data_requires_two_rows_per_modality_after_cleanup():
    df_a, df_b = _build_input_frames()
    df_a.loc[df_a["cell_id"] != "a-2", "feat_a1"] = np.nan

    with pytest.raises(ValueError, match="At least two rows must remain in each modality"):
        prepare_alignment_data(
            df_a,
            df_b,
            id_col_a="cell_id",
            id_col_b="sample_id",
            categorical_cols=["cell_type"],
            features_a=["feat_a1", "feat_a2"],
            features_b=["feat_b1", "feat_b2"],
        )


def test_apply_filter_to_alignment_data_filters_each_modality_independently():
    df_a, df_b = _build_input_frames()
    alignment_data = prepare_alignment_data(
        df_a,
        df_b,
        id_col_a="cell_id",
        id_col_b="sample_id",
        categorical_cols=["cell_type", "treatment"],
        features_a=["feat_a1", "feat_a2"],
        features_b=["feat_b1", "feat_b2"],
    )

    filtered_df = alignment_data.filter_df[alignment_data.filter_df["treatment"] == "ctrl"]
    filtered_alignment = apply_filter_to_alignment_data(alignment_data, filtered_df)

    assert filtered_alignment.cell_ids_a == ["a-2", "a-1"]
    assert filtered_alignment.cell_ids_b == ["b-1", "b-2"]
    assert filtered_alignment.features_a.shape == (2, 2)
    assert filtered_alignment.features_b.shape == (2, 2)


def test_apply_filter_to_alignment_data_requires_two_rows_per_modality_after_filtering():
    df_a, df_b = _build_input_frames()
    alignment_data = prepare_alignment_data(
        df_a,
        df_b,
        id_col_a="cell_id",
        id_col_b="sample_id",
        categorical_cols=["cell_type", "treatment"],
        features_a=["feat_a1", "feat_a2"],
        features_b=["feat_b1", "feat_b2"],
    )

    filtered_df = alignment_data.filter_df[alignment_data.filter_df["modality"] == "Modality A"]

    with pytest.raises(ValueError, match="At least two rows must remain in each modality after filtering"):
        apply_filter_to_alignment_data(alignment_data, filtered_df)


class FakeSCOTv2:
    last_kwargs = None
    call_count = 0

    def __init__(self, data):
        self.data = data

    def align(self, **kwargs):
        type(self).call_count += 1
        type(self).last_kwargs = kwargs
        rows_a = self.data[0].shape[0]
        rows_b = self.data[1].shape[0]
        dims = kwargs["out_dim"]
        aligned_a = np.arange(rows_a * dims, dtype=float).reshape(rows_a, dims)
        aligned_b = np.arange(rows_b * dims, dtype=float).reshape(rows_b, dims) + 100.0
        return [aligned_a, aligned_b]


def test_run_scot_alignment_combines_modalities_and_clamps_k():
    df_a, df_b = _build_input_frames()
    alignment_data = prepare_alignment_data(
        df_a,
        df_b,
        id_col_a="cell_id",
        id_col_b="sample_id",
        categorical_cols=["cell_type", "treatment"],
        features_a=["feat_a1", "feat_a2"],
        features_b=["feat_b1", "feat_b2"],
    )

    combined_df = run_scot_alignment(
        alignment_data,
        {
            "k": 20,
            "eps": 0.005,
            "rho": 0.1,
            "out_dim": 4,
            "projection_method": "embedding",
        },
        aligner_cls=FakeSCOTv2,
    )

    assert FakeSCOTv2.last_kwargs["k"] == 2
    assert FakeSCOTv2.last_kwargs["eps"] == 0.005
    assert FakeSCOTv2.last_kwargs["rho"] == 0.1
    assert FakeSCOTv2.last_kwargs["projMethod"] == "embedding"
    assert combined_df.shape == (6, 8)
    assert combined_df.columns.tolist() == [
        "SCOT1",
        "SCOT2",
        "SCOT3",
        "SCOT4",
        "cell_id",
        "cell_type",
        "treatment",
        "modality",
    ]
    assert combined_df["modality"].tolist() == [
        "Modality A",
        "Modality A",
        "Modality A",
        "Modality B",
        "Modality B",
        "Modality B",
    ]
    assert combined_df["cell_id"].tolist() == ["a-2", "a-1", "a-3", "b-1", "b-2", "b-3"]


def test_run_scot_alignment_caches_default_alignment_result(monkeypatch):
    import streamlit as st
    from src.vendor.scot import scotv2 as scotv2_module

    FakeSCOTv2.call_count = 0
    st.cache_data.clear()
    monkeypatch.setattr(scotv2_module, "SCOTv2", FakeSCOTv2)

    df_a, df_b = _build_input_frames()
    alignment_data = prepare_alignment_data(
        df_a,
        df_b,
        id_col_a="cell_id",
        id_col_b="sample_id",
        categorical_cols=["cell_type", "treatment"],
        features_a=["feat_a1", "feat_a2"],
        features_b=["feat_b1", "feat_b2"],
    )
    scot_params = {
        "k": 20,
        "eps": 0.005,
        "rho": 0.1,
        "out_dim": 4,
        "projection_method": "embedding",
    }

    first = run_scot_alignment(alignment_data, scot_params)
    second = run_scot_alignment(alignment_data, scot_params)

    assert FakeSCOTv2.call_count == 1
    pd.testing.assert_frame_equal(first, second)


def test_data_analysis_source_mentions_modality_alignment():
    source = Path("pages/data_analysis.py").read_text()
    assert "Modality Alignment" in source
    assert "_render_compact_file_uploader_style()" in source
    assert 'section[data-testid="stFileUploaderDropzone"]' in source
    assert "Shared categorical columns" not in source
    assert "Automatically using shared metadata columns for filters and validation" not in source
    assert "Modality A unique ID column" not in source
    assert "Modality B unique ID column" not in source
    assert "matches the two CSVs by unique cell ID" not in source
    assert "per-upload schema configuration and aligns the two CSVs as separate modalities" not in source
    assert "SCOT aligns the two modalities into a shared latent space first." not in source
    assert "**2D Visualization (after Alignment)**" in source
    assert 'st.expander("SCOT Hyperparameters", expanded=False)' in source
    assert 'alignment_filter_cols = ["modality"] +' not in source
    assert 'alignment_plot_categorical_cols = ["modality"] +' in source
    assert 'st.status("Running modality alignment..."' in source


def test_visualization_widgets_source_mentions_scot_widget():
    source = Path("src/widgets/visualization_widgets.py").read_text()
    module = ast.parse(source)
    assert any(
        isinstance(node, ast.FunctionDef) and node.name == "scot_hyperParams_widget"
        for node in module.body
    )
