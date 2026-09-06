"""Edit subpopulation names beside fitted 1D or 2D GMM statistics."""
from contextlib import nullcontext
import hashlib
import json
from html import escape

import pandas as pd
import streamlit as st


def gmm_component_names_editor(tables, entry, *, method, context):
    """Return edits keyed by original labels; model statistics stay read-only."""
    names = {}
    per_row = 2 if len(tables) > 1 and len(tables[0]["features"]) == 1 else 1
    for start in range(0, len(tables), per_row):
        columns = st.columns(2, gap="small") if per_row == 2 else [nullcontext()]
        for table, column in zip(tables[start:start + per_row], columns):
            with column:
                names.update(_component_names_table(table, entry, method=method, context=context))
    return names


def _component_names_table(table, entry, *, method, context):
    """Render one editor with stable inputs, independent of its layout position."""
    is_1d = len(table["features"]) == 1
    number_width, name_width = (24 if is_1d else 28), 177
    mean_width, weight_width = (104, 60) if is_1d else (120, 70)
    name_label = "✎ Name"
    bases = entry.setdefault("table_bases", {})
    rows = table["rows"]
    h_index = table.get("h_index")
    mean_columns = ["Mean ± SD"] if is_1d else ["X (mean ± SD)", "Y (mean ± SD)"]
    labels = [row["source_label"] for row in rows]
    # A changed column layout remounts from saved names, not old input baselines.
    identity = json.dumps([table["category"], table["group"], labels, mean_columns,
                           name_label, number_width, name_width, mean_width, weight_width], default=str)
    digest = hashlib.sha256(identity.encode()).hexdigest()[:16]
    key = f"gmm_table_names_{method}_{entry['generation']}_{digest}"
    # data_editor includes its input bytes in widget identity. Keep its input
    # stable while mounted so successive edits accumulate. After hiding/remounting,
    # seed from saved names rather than assigning protected editor widget state.
    if key not in st.session_state or key not in bases:
        bases[key] = [entry["value_names"].get(label, label) for label in labels]
    data = pd.DataFrame({
        "#": [row["component"] for row in rows], "Name": bases[key],
        **{column: [row[field] for row in rows]
           for column, field in zip(mean_columns, ("x_mean_sd", "y_mean_sd"))},
        "Weight": [row["weight"] for row in rows],
    }, index=labels)
    features = [str(feature) + (" (log₁₀)" if context.get("fit", {}).get(axis) else "")
                for feature, axis in zip(table["features"], ("log_x", "log_y"))]
    fit_label = f" (H-index: {h_index:.3f})" if h_index is not None else ""
    st.markdown(
        f"<p><strong>{escape(str(table['group']))}{fit_label}</strong></p>",
        unsafe_allow_html=True,
    )
    edited = st.data_editor(
        data, key=key, hide_index=True, num_rows="fixed", width="stretch", height="content",
        disabled=[column for column in data if column != "Name"],
        column_config={
            "#": st.column_config.NumberColumn("#", width=number_width, format="%d"),
            "Name": st.column_config.TextColumn(
                name_label, width=name_width, required=True,
                help="Double-click a name to edit the exported subpopulation label."),
            **{column: st.column_config.TextColumn(width=mean_width, help=feature)
               for column, feature in zip(mean_columns, features)},
            "Weight": st.column_config.NumberColumn(width=weight_width, format="%.2f"),
        })
    return edited["Name"].to_dict()
