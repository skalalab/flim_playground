"""Column roles, grouping rules, and review-table previews.

Keep this module free of internal imports: both dataset_io and the profile
widgets use it, and dataset_io already imports those widgets.
"""
from collections import Counter

import pandas as pd

# Each column has one role. Ignore remains recorded so profile matching can
# distinguish a dismissed column from an unseen one. FOV columns are categoricals.
ROLE_ROW_ID = "row_id"
ROLE_CATEGORICAL = "categorical"
ROLE_NUMERICAL = "numerical"
ROLE_IGNORE = "ignore"

# Highest precedence first when a column appears in multiple stored role lists.
# profile_column_roles walks this in reverse so higher roles overwrite lower ones.
ROLES = (ROLE_ROW_ID, ROLE_CATEGORICAL, ROLE_NUMERICAL, ROLE_IGNORE)

# Labels shared by the review table and its validation messages.
ROLE_LABELS = {
    ROLE_ROW_ID: "Row ID",
    ROLE_CATEGORICAL: "Categorical",
    ROLE_NUMERICAL: "Numerical",
    ROLE_IGNORE: "Ignore",
}
LABEL_ROLES = {label: role for role, label in ROLE_LABELS.items()}

# Roles that may be assigned to at most one column.
_SOLE_ROLES = (ROLE_ROW_ID,)

NO_GROUP = "—"
# Display label for an ungrouped measurement. build_working_copy merges groups
# with this reserved name into the ungrouped slot; NO_GROUP is the widget value.
UNGROUPED_LABEL = "Uncategorized"


def _is_whole_number_column(series):
    """Whether every value is whole, including integers stored as floats.

    Modulo rejects fractional and infinite values; unsupported types return False.
    """
    try:
        return bool(((series % 1) == 0).all())
    except TypeError:
        return False


def _is_row_id_candidate(series):
    """Whether the values qualify as a row-ID candidate, regardless of name or position.

    Values must be non-null and distinct. Booleans are excluded; other numeric
    values must be whole numbers, and nonnumeric values must stay unique as text.
    A unique integer measurement can qualify, so the review table exposes the guess.
    """
    if series.isna().any() or series.nunique(dropna=False) != len(series):
        return False
    if pd.api.types.is_bool_dtype(series):
        return False        # Booleans remain categorical even when unique.
    if not pd.api.types.is_numeric_dtype(series):
        return series.astype(str).is_unique
    return _is_whole_number_column(series)


def detect_column_roles(df, guess_row_id=True):
    """Guess `{column: role}` from values and dtypes, without cardinality cutoffs.

    Empty columns are Ignore. The leftmost row-ID candidate takes Row ID unless
    `guess_row_id=False`. Remaining numeric columns are Numerical, except booleans;
    all others are Categorical. `dataset_io.detect_roles` applies numeric coercion
    before calling this function.
    """
    roles = {}
    row_id_taken = not guess_row_id
    for col in df.columns:
        series = df[col]
        if series.isna().all():
            # Empty columns are dropped by the loader but still belong to the
            # profile's known columns. Their float dtype must not imply a feature.
            roles[col] = ROLE_IGNORE
        elif not row_id_taken and _is_row_id_candidate(series):
            roles[col] = ROLE_ROW_ID
            row_id_taken = True
        elif pd.api.types.is_bool_dtype(series):
            # pandas considers bool numeric; the analysis treats it as a category.
            roles[col] = ROLE_CATEGORICAL
        elif pd.api.types.is_numeric_dtype(series):
            roles[col] = ROLE_NUMERICAL
        else:
            roles[col] = ROLE_CATEGORICAL
    return roles


# The earliest separator in a name wins. Hyphens stay within names such as
# "E-cadherin" and "anti-PD1_dose", whose prefix is "anti-PD1".
_GROUP_SEPARATORS = (": ", "_", ".")


def _prefix(name):
    """Return a nonempty prefix before a supported separator, or None."""
    found = [(name.find(sep), sep) for sep in _GROUP_SEPARATORS]
    found = [(at, sep) for at, sep in found if at > 0]
    if not found:
        return None
    return name[:min(found)[0]]


def sibling_groups(keys_and_groups):
    """Return `{key: group}` where all known columns with that key agree.

    This lets new columns follow their siblings into renamed groups. Skip None
    keys and ambiguous keys whose columns belong to multiple groups.
    """
    by_key = {}
    for key, group in keys_and_groups:
        if key is None:
            continue
        by_key.setdefault(key, set()).add(group)
    return {key: next(iter(groups))
            for key, groups in by_key.items() if len(groups) == 1}


def detect_column_groups(columns, existing_groups=None, known_groups=None):
    """Guess `{column: group}` for new columns, omitting ungrouped columns.

    Pass only columns the profile does not know, preserving saved assignments.
    In order, each prefix follows its known siblings' shared group, joins an
    existing group of the same name, or forms a group with another new column
    sharing that prefix.

    `existing_groups` supplies group names, including empty groups; a mapping or
    iterable is accepted. `known_groups` maps saved column names to group names.
    """
    existing = set(existing_groups or ())
    siblings = sibling_groups(
        (_prefix(col), group) for col, group in (known_groups or {}).items())
    groups = {}
    rest = []
    for col in columns:
        prefix = _prefix(col)
        if prefix is None:
            continue
        if prefix in siblings:
            groups[col] = siblings[prefix]
        elif prefix in existing:
            groups[col] = prefix
        else:
            rest.append((col, prefix))

    shared = Counter(prefix for _col, prefix in rest)
    groups.update({col: prefix for col, prefix in rest if shared[prefix] > 1})
    return groups


def code_span(text):
    """Wrap a name or value as literal text in a Markdown code span.

    Apply this to interpolated values, leaving the message's own Markdown intact.
    The fence exceeds every backtick run in the text; padding protects backticks
    at either edge under CommonMark's code-span rules.
    """
    text = str(text) or " "
    longest = run = 0
    for char in text:
        run = run + 1 if char == "`" else 0
        longest = max(longest, run)
    fence = "`" * (longest + 1)
    pad = " " if text.startswith("`") or text.endswith("`") else ""
    return f"{fence}{pad}{text}{pad}{fence}"


def _number(value):
    """A measurement as the review table shows it: short, and never 7.0 for 7."""
    return f"{value:g}"


def column_preview(series, numeric=None):
    """Summarize a column as a numeric range or a sample value and level count.

    `numeric` supplies the analysis' coercion result while `series` stays raw;
    None falls back to its dtype. Booleans and columns with no convertible values
    use the categorical form so the preview agrees with their contents.
    """
    values = series.dropna()
    if values.empty:
        # Normalization drops all-empty columns after review.
        return "empty — will be dropped"
    # The caller's numeric set may include bool because pandas treats it as numeric.
    reads_numeric = (pd.api.types.is_numeric_dtype(series) if numeric is None else numeric)
    if reads_numeric and not pd.api.types.is_bool_dtype(series):
        # Use the same conversion as the analysis when previewing raw text numbers.
        numbers = pd.to_numeric(values, errors="coerce").dropna()
        if not numbers.empty:
            low, high = numbers.min(), numbers.max()
            return _number(low) if low == high else f"{_number(low)} – {_number(high)}"
    levels = values.nunique()
    return f"{values.iloc[0]} ({levels} level{'' if levels == 1 else 's'})"


def enforce_role_invariants(roles, groups, numeric_cols=(), previous_roles=None):
    """Return `(roles, groups, notices)` with at most one Row ID and numerical groups.

    The last newly assigned Row ID wins, or the first holder when no edit identifies
    a winner. Demoted holders become Numerical if in `numeric_cols`, Categorical
    otherwise. Inputs are not mutated, and having no Row ID is valid.
    """
    roles = dict(roles)
    previous = previous_roles or {}
    notices = []
    for role in _SOLE_ROLES:
        holders = [col for col, held in roles.items() if held == role]
        if len(holders) < 2:
            continue
        # With no new claimant, preserve the first holder in column order.
        claimants = [] if previous_roles is None else [
            col for col in holders if previous.get(col) != role]
        keeps = claimants[-1] if claimants else holders[0]
        for col in holders:
            if col == keeps:
                continue
            roles[col] = ROLE_NUMERICAL if col in numeric_cols else ROLE_CATEGORICAL
            notices.append(
                f"Only one column can be the {ROLE_LABELS[role]}: {code_span(keeps)} took "
                f"it, so {code_span(col)} is {ROLE_LABELS[roles[col]]} again.")
    groups = {col: group for col, group in groups.items()
              if group and group != NO_GROUP and roles.get(col) == ROLE_NUMERICAL}
    return roles, groups, notices


def validate_roles(roles):
    """Return an error if no column is marked Numerical, otherwise ""."""
    if not any(role == ROLE_NUMERICAL for role in roles.values()):
        return ("No column is marked Numerical, so there would be nothing to plot. "
                "Mark at least one measurement column Numerical.")
    return ""


def row_id_notice(roles):
    """Explain generated row numbers, or return "" when a Row ID is assigned."""
    if any(role == ROLE_ROW_ID for role in roles.values()):
        return ""
    return "Rows will be identified by row number."
