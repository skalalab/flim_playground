"""Shared naming panel for derived GMM categorical CSV exports."""
from contextlib import nullcontext
import hashlib
import json

import pandas as pd
import streamlit as st

from src.column_roles import code_span
from src.export_labels import normalize_export_labels
from src.vis.helpers import natural_tuple_sort
from src.widgets.analysis_widget_state import control_default
from src.widgets.gmm_table_widgets import gmm_component_names_editor


def export_context_fingerprint(df, context):
    """Include analyzed values and fit settings, excluding display-only controls."""
    digest = hashlib.sha256(json.dumps(context, sort_keys=True, default=str).encode())
    digest.update(repr(list(zip(df.columns, map(str, df.dtypes)))).encode())
    digest.update(pd.util.hash_pandas_object(df, index=True).to_numpy().tobytes())
    return digest.hexdigest()


def export_labels_entry(df, source_column, *, method, context):
    """Return fit-scoped names shared by the export controls and results tables."""
    signature = export_context_fingerprint(df, context)
    store = st.session_state.setdefault("_derived_export_labels", {})
    entry = store.get(method)
    if entry is None or entry["signature"] != signature:
        entry = {"signature": signature, "generation": entry["generation"] + 1 if entry else 0,
                 "column_name": source_column, "value_names": {}}
        store[method] = entry
    return entry


def export_labels_widget(df, source_column, *, method, context, component_tables=None,
                         component_table_layout=None, column_name_container=None):
    """Return (normalized settings, valid); retain names when inputs are hidden.

    An optional layout receives the shared table editor and returns its name edits.
    Without assignments it receives None to render read-only fit details instead.
    """
    entry = export_labels_entry(df, source_column, method=method, context=context)
    if source_column not in df or not df[source_column].notna().any():
        if component_table_layout is not None:
            component_table_layout(None)
        reason = ("GMM assigns subpopulation labels only to fits with more than one component. "
                  "Groups with insufficient data or no valid fit also remain unassigned.")
        st.info(f"No group labels were generated for the current analysis. {reason}")
        return None, True

    suffix = f"{method}_{entry['generation']}"
    column_key = f"export_label_column_{suffix}"
    with column_name_container if column_name_container is not None else nullcontext():
        column_name = st.text_input(
            "Exported column name", value=control_default(st.session_state, column_key, entry["column_name"]),
            key=column_key)
    labels = natural_tuple_sort(df[source_column].dropna().unique())
    features = list(dict.fromkeys(context.get("features", []))) if component_tables is None else []
    # The result frame already contains analyzed (possibly log-transformed)
    # values. Summarize those assigned rows without applying a transform twice.
    means = df.groupby(source_column, observed=True)[features].mean() if features else None
    feature_labels = []
    for index, feature in enumerate(features):
        log_scale = context.get("fit", {}).get("log_x" if index == 0 else "log_y")
        feature_labels.append(code_span(feature) + (" (log₁₀)" if log_scale else ""))
    value_names = {label: entry["value_names"].get(label, label) for label in labels}
    if component_tables is not None:
        if component_tables:
            st.markdown("**✎ Double-click a subpopulation name to rename it.**")
        # Retain edits to hidden categories and components without assigned rows.
        value_names = {**entry["value_names"], **value_names}

        def editor(tables):
            return gmm_component_names_editor(tables, entry, method=method, context=context)

        value_names.update(component_table_layout(editor) if component_table_layout is not None
                           else editor(component_tables))
    else:
        for start in range(0, len(labels), 2):
            columns = st.columns(min(2, len(labels)))
            for offset, (column, original) in enumerate(zip(columns, labels[start:start + 2])):
                key = f"export_label_value_{suffix}_{start + offset}"
                help_text = None
                if features:
                    help_text = "Mean of assigned rows\n\n" + "\n\n".join(
                        f"{label}: {means.loc[original, feature]:.4g}"
                        for feature, label in zip(features, feature_labels))
                with column:
                    value_names[original] = st.text_input(
                        f"New name for {code_span(original)}",
                        value=control_default(st.session_state, key, value_names[original]),
                        key=key, help=help_text)
    entry["column_name"] = column_name
    entry["value_names"] = value_names
    try:
        settings = normalize_export_labels(df, source_column, entry)
    except ValueError as error:
        st.error(code_span(str(error)))
        return None, False
    counts = pd.Series(list(settings["value_names"].values())).value_counts()
    shared = counts[counts > 1].index.tolist()
    if shared:
        st.info("These exported values combine multiple generated groups: " +
                ", ".join(code_span(name) for name in shared))
    return settings, True
