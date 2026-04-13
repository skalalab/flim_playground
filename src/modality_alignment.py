from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
import json
from typing import Iterable

import numpy as np
import pandas as pd
import streamlit as st


_LOCAL_SCOT_ALIGNMENT_CACHE: dict[tuple[object, ...], pd.DataFrame] = {}


@dataclass(frozen=True)
class AlignmentData:
    cell_ids_a: list[str]
    cell_ids_b: list[str]
    metadata_a: pd.DataFrame
    metadata_b: pd.DataFrame
    features_a: pd.DataFrame
    features_b: pd.DataFrame
    filter_df: pd.DataFrame
    dropped_na_rows_a: int
    dropped_na_rows_b: int


def infer_default_id_column(columns: Iterable[str]) -> str | None:
    candidates = list(columns)
    if not candidates:
        return None

    normalized = {
        col: col.lower().replace(" ", "_").replace("-", "_")
        for col in candidates
    }
    priorities = (
        "cell_id",
        "cell_ids",
        "sample_id",
        "sample_ids",
        "id",
    )
    for target in priorities:
        for col, normalized_col in normalized.items():
            if normalized_col == target:
                return col
    for col, normalized_col in normalized.items():
        if normalized_col.endswith("_id") or normalized_col == "id":
            return col
    return candidates[0]


def _normalized_column_name(col: str) -> str:
    return col.lower().replace(" ", "_").replace("-", "_")


def load_alignment_csv(uploaded_csv) -> tuple[pd.DataFrame | None, list[str]]:
    warnings: list[str] = []
    if uploaded_csv is None:
        return None, warnings

    raw = uploaded_csv.getvalue() if hasattr(uploaded_csv, "getvalue") else uploaded_csv.read()
    df = pd.read_csv(BytesIO(raw), index_col=False, low_memory=False)
    df = df.reset_index(drop=True)

    empty_cols = df.columns[df.isnull().all()].tolist()
    if empty_cols:
        df = df.drop(columns=empty_cols)
        warnings.append(
            f"Removed {len(empty_cols)} empty column(s): {', '.join(empty_cols[:5])}"
        )

    duplicate_cols = df.columns[df.columns.duplicated()].tolist()
    if duplicate_cols:
        df = df.loc[:, ~df.columns.duplicated(keep="first")]
        warnings.append(
            f"Dropped duplicate column(s), keeping first occurrence: {', '.join(duplicate_cols[:5])}"
        )

    return df, warnings


def get_numeric_feature_options(
    df: pd.DataFrame,
    id_col: str,
    categorical_cols: list[str],
) -> list[str]:
    excluded = {id_col, *categorical_cols}
    numeric_cols: list[str] = []
    for col in df.columns:
        if col in excluded:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        if series.notna().any():
            numeric_cols.append(col)
    return numeric_cols


def _looks_numeric(series: pd.Series) -> bool:
    non_null = series.dropna()
    if non_null.empty:
        return False
    if pd.api.types.is_numeric_dtype(non_null):
        return True
    coerced = pd.to_numeric(non_null, errors="coerce")
    return bool(coerced.notna().all())


def infer_shared_categorical_columns(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    id_col_a: str,
    id_col_b: str,
) -> list[str]:
    shared_cols = sorted(
        col for col in set(df_a.columns).intersection(df_b.columns) if col not in {id_col_a, id_col_b}
    )
    return [
        col
        for col in shared_cols
        if not _looks_numeric(df_a[col]) and not _looks_numeric(df_b[col])
    ]


def classify_alignment_prep_issue(exc: Exception) -> tuple[str, str]:
    return "error", str(exc)


def _normalize_id_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        non_null = series.dropna()
        if not non_null.empty and np.isfinite(non_null).all() and np.allclose(non_null, np.round(non_null)):
            normalized = series.astype("Int64").astype("string")
        else:
            normalized = series.astype("string")
    else:
        normalized = series.astype("string")
    return normalized.fillna("").str.strip()


def _candidate_id_columns(df: pd.DataFrame) -> dict[str, pd.Series]:
    candidates: dict[str, pd.Series] = {}
    for col in df.columns:
        if _looks_numeric(df[col]):
            continue
        normalized = _normalize_id_series(df[col])
        if (normalized == "").any():
            continue
        if normalized.nunique(dropna=False) != len(df):
            continue
        candidates[col] = normalized
    return candidates


def _id_name_score(col: str) -> int:
    normalized = _normalized_column_name(col)
    priorities = {
        "cell_id": 50,
        "cell_ids": 45,
        "sample_id": 40,
        "sample_ids": 35,
        "id": 30,
    }
    if normalized in priorities:
        return priorities[normalized]
    if normalized.endswith("_id"):
        return 20
    if "id" in normalized:
        return 10
    return 0


def _infer_best_id_column(df: pd.DataFrame, csv_label: str) -> str:
    candidates = _candidate_id_columns(df)
    if not candidates:
        raise ValueError(
            f"Could not infer matching ID columns. {csv_label} does not have a unique non-numeric "
            "column that can serve as a row identifier."
        )

    best_col = None
    best_score = None
    for col in df.columns:
        if col not in candidates:
            continue
        score = (_id_name_score(col),)
        if best_score is None or score > best_score:
            best_score = score
            best_col = col

    if best_col is None:
        raise ValueError(
            f"Could not infer matching ID columns. {csv_label} does not have a usable row identifier."
        )
    return best_col


def infer_alignment_id_columns(df_a: pd.DataFrame, df_b: pd.DataFrame) -> tuple[str, str]:
    return _infer_best_id_column(df_a, "CSV A"), _infer_best_id_column(df_b, "CSV B")


def _prepare_frame(
    df: pd.DataFrame,
    id_col: str,
    categorical_cols: list[str],
    selected_features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if id_col not in df.columns:
        raise ValueError(f"ID column '{id_col}' is missing.")
    if len(selected_features) < 2:
        raise ValueError("Select at least two numeric features for each modality.")

    missing_features = [feature for feature in selected_features if feature not in df.columns]
    if missing_features:
        raise ValueError(
            f"Selected feature columns are missing: {', '.join(missing_features)}"
        )

    working_df = df.copy()
    working_df[id_col] = _normalize_id_series(working_df[id_col])
    if working_df[id_col].duplicated().any():
        raise ValueError(f"Duplicate values found in '{id_col}'.")

    missing_categorical = [col for col in categorical_cols if col not in working_df.columns]
    if missing_categorical:
        raise ValueError(
            f"Shared categorical columns are missing: {', '.join(missing_categorical)}"
        )

    metadata_cols = [id_col] + categorical_cols
    metadata_df = working_df[metadata_cols].copy()
    for col in categorical_cols:
        metadata_df[col] = metadata_df[col].fillna("N/A").astype(str)

    features_df = working_df[[id_col] + selected_features].copy()
    for feature in selected_features:
        features_df[feature] = pd.to_numeric(features_df[feature], errors="coerce")
    features_df = features_df.set_index(id_col)

    metadata_df = metadata_df.set_index(id_col)
    return metadata_df, features_df


def _build_modality_filter_frame(
    metadata: pd.DataFrame,
    features: pd.DataFrame,
    modality_label: str,
    feature_prefix: str,
    own_features: list[str],
    other_feature_prefix: str,
    other_features: list[str],
) -> pd.DataFrame:
    id_col = metadata.index.name or "cell_id"
    filter_df = metadata.reset_index().rename(columns={id_col: "cell_id"})
    filter_df.insert(1, "modality", modality_label)
    for feature in own_features:
        filter_df[f"{feature_prefix}__{feature}"] = features[feature].to_numpy()
    for feature in other_features:
        filter_df[f"{other_feature_prefix}__{feature}"] = np.nan
    return filter_df


def prepare_alignment_data(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    id_col_a: str,
    id_col_b: str,
    categorical_cols: list[str],
    features_a: list[str],
    features_b: list[str],
) -> AlignmentData:
    metadata_a, features_df_a = _prepare_frame(df_a, id_col_a, categorical_cols, features_a)
    metadata_b, features_df_b = _prepare_frame(df_b, id_col_b, categorical_cols, features_b)

    keep_mask_a = features_df_a[features_a].notna().all(axis=1)
    keep_mask_b = features_df_b[features_b].notna().all(axis=1)
    dropped_na_rows_a = int((~keep_mask_a).sum())
    dropped_na_rows_b = int((~keep_mask_b).sum())

    metadata_a = metadata_a.loc[keep_mask_a].copy()
    metadata_b = metadata_b.loc[keep_mask_b].copy()
    features_df_a = features_df_a.loc[keep_mask_a, features_a].copy()
    features_df_b = features_df_b.loc[keep_mask_b, features_b].copy()

    if len(metadata_a) < 2 or len(metadata_b) < 2:
        raise ValueError("At least two rows must remain in each modality after feature cleanup.")

    metadata_a_out = metadata_a.reset_index().rename(columns={id_col_a: "cell_id"})
    metadata_b_out = metadata_b.reset_index().rename(columns={id_col_b: "cell_id"})

    filter_df_a = _build_modality_filter_frame(
        metadata_a,
        features_df_a,
        modality_label="Modality A",
        feature_prefix="A",
        own_features=features_a,
        other_feature_prefix="B",
        other_features=features_b,
    )
    filter_df_b = _build_modality_filter_frame(
        metadata_b,
        features_df_b,
        modality_label="Modality B",
        feature_prefix="B",
        own_features=features_b,
        other_feature_prefix="A",
        other_features=features_a,
    )
    filter_df = pd.concat([filter_df_a, filter_df_b], ignore_index=True)

    ordered_columns = [
        "cell_id",
        "modality",
        *categorical_cols,
        *[f"A__{feature}" for feature in features_a],
        *[f"B__{feature}" for feature in features_b],
    ]
    filter_df = filter_df[ordered_columns]

    return AlignmentData(
        cell_ids_a=metadata_a_out["cell_id"].tolist(),
        cell_ids_b=metadata_b_out["cell_id"].tolist(),
        metadata_a=metadata_a_out,
        metadata_b=metadata_b_out,
        features_a=features_df_a,
        features_b=features_df_b,
        filter_df=filter_df,
        dropped_na_rows_a=dropped_na_rows_a,
        dropped_na_rows_b=dropped_na_rows_b,
    )


def apply_filter_to_alignment_data(
    alignment_data: AlignmentData,
    filtered_df: pd.DataFrame,
) -> AlignmentData:
    if "cell_id" not in filtered_df.columns:
        raise ValueError("Filtered dataframe must contain a 'cell_id' column.")
    if "modality" not in filtered_df.columns:
        raise ValueError("Filtered dataframe must contain a 'modality' column.")

    filtered_ids_a = (
        filtered_df.loc[filtered_df["modality"] == "Modality A", "cell_id"].astype(str).tolist()
    )
    filtered_ids_b = (
        filtered_df.loc[filtered_df["modality"] == "Modality B", "cell_id"].astype(str).tolist()
    )
    if len(filtered_ids_a) < 2 or len(filtered_ids_b) < 2:
        raise ValueError("At least two rows must remain in each modality after filtering.")

    metadata_a = alignment_data.metadata_a.set_index("cell_id").loc[filtered_ids_a].reset_index()
    metadata_b = alignment_data.metadata_b.set_index("cell_id").loc[filtered_ids_b].reset_index()
    features_a = alignment_data.features_a.loc[filtered_ids_a]
    features_b = alignment_data.features_b.loc[filtered_ids_b]

    return AlignmentData(
        cell_ids_a=filtered_ids_a,
        cell_ids_b=filtered_ids_b,
        metadata_a=metadata_a,
        metadata_b=metadata_b,
        features_a=features_a,
        features_b=features_b,
        filter_df=filtered_df.reset_index(drop=True).copy(),
        dropped_na_rows_a=alignment_data.dropped_na_rows_a,
        dropped_na_rows_b=alignment_data.dropped_na_rows_b,
    )


def _clamp_k(requested_k: int, n_rows_a: int, n_rows_b: int) -> int:
    return max(1, min(int(requested_k), n_rows_a - 1, n_rows_b - 1))


def _serialize_dataframe(df: pd.DataFrame) -> str:
    normalized_df = df.copy().where(pd.notna(df), None)
    payload = {
        "columns": list(normalized_df.columns),
        "index": normalized_df.index.tolist(),
        "index_name": normalized_df.index.name,
        "data": normalized_df.values.tolist(),
        "dtypes": [str(dtype) for dtype in normalized_df.dtypes],
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _deserialize_dataframe(payload: str) -> pd.DataFrame:
    data = json.loads(payload)
    df = pd.DataFrame(data["data"], columns=data["columns"])
    df.index = pd.Index(data["index"], name=data["index_name"])
    for column, dtype in zip(data["columns"], data["dtypes"]):
        df[column] = df[column].astype(dtype)
    return df


@lru_cache(maxsize=32)
@st.cache_data(show_spinner=False)
def _run_scot_alignment_cached(
    features_a_payload: str,
    features_b_payload: str,
    metadata_a_payload: str,
    metadata_b_payload: str,
    *,
    k: int,
    eps: float,
    rho: float,
    out_dim: int,
    projection_method: str,
) -> pd.DataFrame:
    from src.scot.scotv2 import SCOTv2

    features_a = _deserialize_dataframe(features_a_payload)
    features_b = _deserialize_dataframe(features_b_payload)
    metadata_a = _deserialize_dataframe(metadata_a_payload)
    metadata_b = _deserialize_dataframe(metadata_b_payload)

    aligner = SCOTv2(
        [
            features_a.to_numpy(dtype=np.float32),
            features_b.to_numpy(dtype=np.float32),
        ]
    )
    aligned = aligner.align(
        normalize=True,
        k=k,
        eps=eps,
        rho=rho,
        projMethod=projection_method,
        out_dim=out_dim,
    )

    combined = np.vstack(aligned)
    dim_cols = [f"SCOT{i + 1}" for i in range(combined.shape[1])]
    dims_df = pd.DataFrame(combined, columns=dim_cols)
    metadata = pd.concat(
        [
            metadata_a.assign(modality="Modality A"),
            metadata_b.assign(modality="Modality B"),
        ],
        ignore_index=True,
    )
    return pd.concat([dims_df, metadata], axis=1)


def run_scot_alignment(
    alignment_data: AlignmentData,
    scot_params: dict,
    aligner_cls=None,
) -> pd.DataFrame:
    n_rows_a = len(alignment_data.cell_ids_a)
    n_rows_b = len(alignment_data.cell_ids_b)
    if n_rows_a < 2 or n_rows_b < 2:
        raise ValueError("At least two rows must remain in each modality after filtering.")

    projection_method = scot_params.get("projection_method", "embedding")
    out_dim = int(scot_params.get("out_dim", 10))
    k = _clamp_k(scot_params.get("k", 20), n_rows_a, n_rows_b)
    eps = float(scot_params.get("eps", 0.005))
    rho = float(scot_params.get("rho", 0.1))

    if aligner_cls is None:
        features_a_payload = _serialize_dataframe(alignment_data.features_a)
        features_b_payload = _serialize_dataframe(alignment_data.features_b)
        metadata_a_payload = _serialize_dataframe(alignment_data.metadata_a)
        metadata_b_payload = _serialize_dataframe(alignment_data.metadata_b)
        cache_key = (
            features_a_payload,
            features_b_payload,
            metadata_a_payload,
            metadata_b_payload,
            k,
            eps,
            rho,
            out_dim,
            projection_method,
        )
        if cache_key not in _LOCAL_SCOT_ALIGNMENT_CACHE:
            _LOCAL_SCOT_ALIGNMENT_CACHE[cache_key] = _run_scot_alignment_cached(
                features_a_payload,
                features_b_payload,
                metadata_a_payload,
                metadata_b_payload,
                k=k,
                eps=eps,
                rho=rho,
                out_dim=out_dim,
                projection_method=projection_method,
            ).copy(deep=True)
        return _LOCAL_SCOT_ALIGNMENT_CACHE[cache_key].copy(deep=True)

    from src.scot.scotv2 import SCOTv2

    aligner_cls = SCOTv2 if aligner_cls is None else aligner_cls
    aligner = aligner_cls(
        [
            alignment_data.features_a.to_numpy(dtype=np.float32),
            alignment_data.features_b.to_numpy(dtype=np.float32),
        ]
    )
    aligned = aligner.align(
        normalize=True,
        k=k,
        eps=eps,
        rho=rho,
        projMethod=projection_method,
        out_dim=out_dim,
    )

    combined = np.vstack(aligned)
    dim_cols = [f"SCOT{i + 1}" for i in range(combined.shape[1])]
    dims_df = pd.DataFrame(combined, columns=dim_cols)

    metadata = pd.concat(
        [
            alignment_data.metadata_a.assign(modality="Modality A"),
            alignment_data.metadata_b.assign(modality="Modality B"),
        ],
        ignore_index=True,
    )
    return pd.concat([dims_df, metadata], axis=1)
