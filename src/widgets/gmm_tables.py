"""Compact, responsive component tables shared by 1D and 2D GMM results."""
from html import escape


_TABLE_STYLES = """
<style>
.flim-gmm-results {
    container-type: inline-size;
}
.flim-gmm-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 1rem;
}
.flim-gmm-table {
    overflow-x: auto;
}
.flim-gmm-table table {
    width: 100%;
    margin: 0;
    font-size: 0.875rem;
    font-variant-numeric: tabular-nums;
}
.flim-gmm-table caption {
    text-align: left;
    color: inherit;
    font-weight: 600;
    padding-bottom: 0.35rem;
}
.flim-gmm-table th, .flim-gmm-table td {
    padding: 0.35rem 0.5rem;
    text-align: right;
    white-space: nowrap;
}
.flim-gmm-table th:first-child, .flim-gmm-table td:first-child {
    width: 1%;
    text-align: left;
}
@container (max-width: 44rem) {
    .flim-gmm-grid {
        grid-template-columns: minmax(0, 1fr);
    }
}
</style>
"""


def gmm_component_table(group_name, rows, feature_names, *, h_index=None):
    """Format rows of (component, mean ± SD per feature, weight) as one table."""
    labels = ["Mean ± SD"] if len(feature_names) == 1 else ["X (mean ± SD)", "Y (mean ± SD)"]
    headers = ''.join(
        f'<th scope="col" title="{escape(str(feature))}">{label}</th>'
        for feature, label in zip(feature_names, labels)
    )
    fit_label = f" (H-index: {h_index:.3f})" if h_index is not None else ""
    body = ''.join(
        '<tr>' + ''.join(f'<td>{escape(str(value))}</td>' for value in row) + '</tr>'
        for row in rows
    )
    return (
        '<div class="flim-gmm-table"><table>'
        f'<caption>{escape(str(group_name))}{fit_label}</caption>'
        '<thead><tr><th scope="col" title="Component">#</th>'
        + headers + '<th scope="col">Weight</th></tr></thead><tbody>'
        + body + '</tbody></table></div>'
    )


def gmm_tables_html(tables):
    """Arrange component tables in pairs, stacking when the panel is narrow."""
    if not tables:
        return ""
    return (
        _TABLE_STYLES + '\n<div class="flim-gmm-results"><div class="flim-gmm-grid">'
        + ''.join(tables) + '</div></div>'
    )
