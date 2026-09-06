"""Names for derived categorical exports, independent of plotting and Streamlit."""
import pandas as pd


EXPORT_METHODS = {
    "Feature Histogram": ("GMM_group", "gmm_grouped_data.csv", "Download GMM Grouped Data"),
    "2D Feature Distribution": ("2D_GMM_group", "2D_gmm_data.csv", "Download 2D GMM data"),
}


def format_export_group_labels(components, color_group, *, separate_by=None,
                               category=None, color_by=None):
    """Name zero-based GMM components with one shared export format."""
    prefix = color_group
    if separate_by:
        prefix = f"{category}::{color_group}" if color_by else category
    return [f"{prefix}_group{int(component) + 1}" for component in components]


def available_label_column(columns, default):
    """Reserve an unused result name without overwriting an earlier annotation."""
    name, suffix = default, 2
    while name in columns:
        name = f"{default}_{suffix}"
        suffix += 1
    return name


def normalize_export_labels(df, source_column, settings=None):
    """Validate and trim export names; repeated value names intentionally combine categories."""
    settings = settings or {}
    column_name = settings.get("column_name", source_column)
    if not isinstance(column_name, str) or not column_name.strip():
        raise ValueError("Enter a non-empty exported column name.")
    column_name = column_name.strip()
    if column_name != source_column and column_name in df.columns:
        raise ValueError(f"The column {column_name!r} already exists. Choose a different exported column name.")
    value_names = {}
    for original, name in settings.get("value_names", {}).items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"Enter a non-empty exported value name for {original!r}.")
        value_names[original] = name.strip()
    return {"column_name": column_name, "value_names": value_names}


def apply_export_labels(df, source_column, settings=None):
    """Rename only an export copy, preserving missing and unmapped assignments."""
    result = df.copy()
    if source_column not in result.columns:
        return result
    names = normalize_export_labels(result, source_column, settings)
    if names["value_names"]:
        result[source_column] = result[source_column].map(
            lambda value: names["value_names"].get(value, value) if pd.notna(value) else value)
    return result.rename(columns={source_column: names["column_name"]})
