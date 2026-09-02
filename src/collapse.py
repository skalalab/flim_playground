"""Collapse single-cell rows to one row per replicate, holding the arithmetic mean.

Cells are not independent samples. They come from dishes, patients and images, so
treating 300 cells drawn from 3 dishes as n=300 measures cell count rather than
biological difference. `collapse_rows` lets the caller name the real replicate and
reduces the frame to one row per replicate *within each x slot*, after which the box,
the mean line, the significance stars and the effect sizes all read replicates.

**Pure pandas, one public function, no helpers of its own.** `export_script.py`
copies this source verbatim through `inspect.getsource()` and resolves nothing it
calls, so a helper here -- or an import of this project's own modules, or of the UI
framework -- becomes a NameError inside a script advertised as behaving identically to
the app, at run time and on someone else's machine. Every tunable is a default
argument for the same reason: `getsource` copies defaults but no module state.
"""
import pandas as pd


def collapse_rows(df, collapse_by, slot_cols, row_id_col, count_label="n"):
    """One row per (`collapse_by` x x-slot), each numeric column its arithmetic mean.

    `slot_cols` is the page's `[*color_by, separate_by]` passed raw -- Nones,
    duplicates and absent names are dropped here -- so the group key is the replicate
    column plus whatever fixes a point's x position. Keeping the slot columns in the
    key is what preserves a paired design: a dish measured under two treatments stays
    two rows, one in each slot.

    One rule decides every column's fate: **a column survives iff it holds a single
    value in every group.** Numeric columns are averaged. Any other column is kept
    with its value when it is constant within every group (`day` is one value per
    dish) and dropped when it varies inside any group (`image_name` is several per
    dish) -- a row that summarises two values of a column has no single value to
    carry, so guessing one with `.first()` would paint a misleading label silently.
    Dropping instead makes a caller's mistake raise rather than mislead.

    `nunique(dropna=False)` is load-bearing: a column holding "A" in three rows and
    NaN in the fourth is *not* constant.

    Returns ``(collapsed, label_col, varied)``:

    - ``collapsed`` -- the surviving columns in the input's own order, then
      ``label_col``. Row order follows first appearance, so it is reproducible; the
      sina jitter is seeded per group and indexes by row position.
    - ``label_col`` -- the name of the identifier column that replaces `row_id_col`,
      valued like ``"D1 (n=1874)"``. The count rides inside the value rather than in a
      column of its own so the hover needs no second field.
    - ``varied`` -- the columns dropped for varying. The page turns these into the
      visual channels it must switch off, and the reason it shows for doing so.
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
            # An empty frame has no groups, so nunique is NaN rather than 0 -- that
            # is "nothing contradicts it", not "it varies".
            (kept if pd.isna(count) or count <= 1 else varied).append(col)

    # Built from the sizes rather than the means, so a frame with no numeric column
    # still yields one row per group.
    out = grouped.size().rename("__collapse_count__").to_frame()
    if numeric:
        out = out.join(grouped[numeric].mean())
    if kept:
        out = out.join(grouped[kept].first())
    out = out.reset_index()

    # Never overwrite a real column, the way resolve_row_id_col guards "Row number".
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
