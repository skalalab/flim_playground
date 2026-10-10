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


def detect_column_roles(df, guess_row_id=True, id_hints=(),
                        categorical_hints=(), assigned_roles=None):
    """Infer only unknown columns using exact name hints, then emptiness and dtype.

    Preserve every present assignment, including a cleared former Row ID. With no
    assigned Row ID, choose the leftmost unassigned name in id_hints before reading
    values. Its raw values are validated later, never used to select a replacement.
    Numeric coercion belongs to dataset_io.detect_roles, not this pure helper.
    """
    assigned = {col: role for col, role in (assigned_roles or {}).items()
                if col in df.columns}
    row_id = None
    if guess_row_id and ROLE_ROW_ID not in assigned.values():
        ids = set(id_hints)
        row_id = next((col for col in df.columns
                       if col not in assigned and col in ids), None)
    categories = set(categorical_hints)
    roles = {}
    for col in df.columns:
        if col in assigned:
            roles[col] = assigned[col]
            continue
        if col == row_id:
            roles[col] = ROLE_ROW_ID
            continue
        series = df[col]
        if series.isna().all():
            roles[col] = ROLE_IGNORE
        elif col in categories or pd.api.types.is_bool_dtype(series):
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


def _measurement_name(name, extractors):
    """Return (full extractor/channel key, channel), or None for other names."""
    head, separator, feature = name.partition(": ")
    if not separator or not feature:
        return None
    for extractor in extractors:
        prefix = extractor + "_"
        if head.startswith(prefix) and len(head) > len(prefix):
            return head, head[len(prefix):]
    return None


def recognized_channel_names(columns, extractor_hints=(), channel_hints=()):
    """Recognize channels from name hints and full measurement names, in order.

    Column roles do not affect channel recognition. This lets the working-copy
    caller include all present and saved names without supplying group assignments
    from nonnumerical columns.
    """
    extractors = sorted(set(extractor_hints), key=len, reverse=True)
    channels = list(dict.fromkeys(channel_hints))
    for col in columns:
        measurement = _measurement_name(col, extractors)
        if measurement and measurement[1] not in channels:
            channels.append(measurement[1])
    return channels


def detect_column_groups(columns, existing_groups=None, known_groups=None,
                         extractor_hints=(), channel_hints=()):
    """Infer groups for NEW Numerical columns from explicit naming hints.

    Saved sibling choices take priority, including None for an ungrouped sibling.
    Recognized measurements use the full extractor/channel name even as singletons;
    Derived columns use Derived Features. Recognized channel bookkeeping can join
    saved bookkeeping siblings or a group named that channel, but creates no group.
    Other prefixes join saved siblings, an existing name, or 2+ new siblings.
    Caller-owned existing names include empty groups and retain their own order.
    """
    columns = list(columns)
    known = known_groups or {}
    existing = set(existing_groups or ())
    extractors = sorted(set(extractor_hints), key=len, reverse=True)
    channels = sorted(recognized_channel_names(
        [*columns, *known], extractor_hints=extractors,
        channel_hints=channel_hints), key=len, reverse=True)

    def identity(col):
        measurement = _measurement_name(col, extractors)
        if measurement:
            return "measurement", measurement[0]
        if col.startswith("Derived: "):
            return "derived", "Derived Features"
        for channel in channels:
            if col.startswith(channel + "_"):
                return "bookkeeping", channel
        return "generic", _prefix(col)

    def saved_group(group):
        return None if not group or group in (NO_GROUP, UNGROUPED_LABEL) else group

    siblings = sibling_groups((identity(col), saved_group(group))
                              for col, group in known.items())
    identities = {col: identity(col) for col in columns}
    shared = Counter(key for kind, key in identities.values()
                     if kind == "generic" and key is not None)
    groups = {}
    for col, (kind, key) in identities.items():
        if key is None:
            continue
        if (kind, key) in siblings:
            group = siblings[kind, key]
        elif kind in ("measurement", "derived"):
            group = key
        elif key in existing:
            group = key
        elif kind == "generic" and shared[key] >= 2:
            group = key
        else:
            group = None
        if group and group != NO_GROUP:
            groups[col] = group
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
