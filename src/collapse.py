"""Average single-cell measurements within replicate and plot-slot groups.

Keep collapse_rows self-contained apart from pandas: export_script.py embeds it
with inspect.getsource(), without module state or project helpers. Tunable values
must therefore be arguments with defaults.
"""
import pandas as pd


def collapse_rows(df, collapse_by, slot_cols, row_id_col, count_label="n"):
    """Return one row per replicate and plot slot, averaging numeric measurements.

    Group by `collapse_by` and valid, distinct `slot_cols`, preserving separate
    treatments for paired replicates. An absent `collapse_by` leaves df unchanged.
    Retain other nonnumeric columns only if constant within every group, counting
    NaN as a distinct value. Replace the nonnumeric row identifier with a group label.

    Returns ``(collapsed, label_col, varied)``:

    - ``collapsed`` -- surviving columns in input order, then ``label_col``;
      groups follow first appearance for reproducible plot jitter.
    - ``label_col`` -- a unique identifier-column name, with values such as
      ``"D1 (n=1874)"`` that include each group's row count.
    - ``varied`` -- nonnumeric columns dropped for varying within a group.
    """
    if not collapse_by or collapse_by not in df.columns:
        return df, row_id_col, []

    key_cols, in_key = [], set()
    for col in [collapse_by, *(slot_cols or [])]:
        if col and col in df.columns and col not in in_key:
            in_key.add(col)
            key_cols.append(col)

    # dropna=False so a missing value in a key column cannot silently drop rows.
    grouped = df.groupby(key_cols, sort=False, observed=True, dropna=False)

    numeric, others = [], []
    for col in df.columns:
        if col in in_key:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric.append(col)
        elif col != row_id_col:
            others.append(col)

    kept, varied = [], []
    if others:
        levels = grouped[others].nunique(dropna=False).max()
        for col in others:
            count = levels[col]
            # With no groups, a NaN count does not indicate variation.
            (kept if pd.isna(count) or count <= 1 else varied).append(col)

    # Built from the sizes rather than the means, so a frame with no numeric column
    # still yields one row per group.
    out = grouped.size().rename("__collapse_count__").to_frame()
    if numeric:
        out = out.join(grouped[numeric].mean())
    if kept:
        out = out.join(grouped[kept].first())
    out = out.reset_index()

    # Choose a label name that does not overwrite an input column.
    label_col = f"{collapse_by} ({count_label})"
    suffix = 0
    while label_col in df.columns:
        suffix += 1
        label_col = f"{collapse_by} ({count_label}).{suffix}"
    out[label_col] = (out[collapse_by].astype(str)
                      + f" ({count_label}="
                      + out["__collapse_count__"].astype(str) + ")")

    ordered = [col for col in df.columns if col in out.columns] + [label_col]
    return out[ordered], label_col, varied
