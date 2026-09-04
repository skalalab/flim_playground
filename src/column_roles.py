"""What role each column of a table plays, and a guess at it for an unseen file.

Deliberately free of internal imports, like `emojis.py` and for the same reason:
`dataset_io` imports from `widgets/analysis_config_widgets`, so anything both of
them need must live somewhere neither imports. The role constants are needed by
the profile (to read stored lists as a role map) and by the reader (to guess roles
for a file no profile has seen), which is exactly that shape.
"""
from collections import Counter

import pandas as pd

# A column plays exactly one of these. "ignore" is a real, recorded answer -- it
# means "seen and dismissed", which the profile must keep in order to tell that
# apart from a column it has never seen at all.
#
# There is deliberately no FOV role. A user's table may well carry a field-of-view
# column, but nothing distinguishes it from any other categorical, so it is treated as
# one -- filterable, colourable, groupable like the rest. A designated FOV column
# survives only on the extraction branch, where config.toml names it and extraction
# always emits it; here a dataset may or may not have such a column and neither case
# needs a role of its own.
ROLE_ROW_ID = "row_id"
ROLE_CATEGORICAL = "categorical"
ROLE_NUMERICAL = "numerical"
ROLE_IGNORE = "ignore"

# Highest precedence first, and that order is load-bearing, not cosmetic: the profile
# stores roles as parallel lists, and one column can still land in two of them -- a
# hand-edited config, or one written while the FOV role still existed, which put its
# FOV column in categorical_cols as well. Reading them back as a role map needs a total
# order to be a function at all. analysis_config_widgets.profile_column_roles walks
# this in reverse.
ROLES = (ROLE_ROW_ID, ROLE_CATEGORICAL, ROLE_NUMERICAL, ROLE_IGNORE)

# What the review table's dropdown says. Kept beside ROLES so the two cannot drift, and
# free of any import so this module stays cycle-free for the widgets that need it.
ROLE_LABELS = {
    ROLE_ROW_ID: "Row ID",
    ROLE_CATEGORICAL: "Categorical",
    ROLE_NUMERICAL: "Numerical",
    ROLE_IGNORE: "Ignore",
}
LABEL_ROLES = {label: role for role, label in ROLE_LABELS.items()}

# The roles only one column may hold. Optional, and never shared. Still a tuple with
# the row id its only member: enforce_role_invariants walks it, and it has held two.
_SOLE_ROLES = (ROLE_ROW_ID,)

NO_GROUP = "—"


def _is_whole_number_column(series):
    """Whether every value is a whole number -- 1.0, 2.0, 3.0 but not 1.1, 1.2, 1.3.

    Asked of the values rather than the dtype, because the dtype answers the wrong
    question. An integer identifier that came through a spreadsheet, or through
    `coerce_majority_numeric_cols` beside a column that needed a NaN, arrives as
    1.0, 2.0, 3.0 and is still an identifier; refusing every float would miss exactly
    the column the coercion had just made. What rules a column out is a fractional
    part, and that is what this asks.

    `% 1` carries the two numeric edge cases for free: `inf % 1` is NaN, which fails the
    comparison, and complex raises rather than comparing -- neither is a whole number.
    """
    try:
        return bool(((series % 1) == 0).all())
    except TypeError:
        return False


def _is_row_id_candidate(series):
    """Whether a column could be an identifier, from its values alone.

    Two questions, and neither reads the position, the name, or the numeric dtype:

    - every value distinct and non-null, always. The same bijection `_row_id_reason`
      demands of a column the user *names*, so auto-detect can only ever propose a
      column that would survive the gate;
    - a numeric column must additionally hold whole numbers. A non-numeric one
      qualifies outright, since distinct text is not a measurement whatever else it is.

    **Why position is not a third question.** Restricting a numeric candidate to column 0
    protects an all-distinct integer measurement (`npix` over 1204 cells) from taking the
    role -- but only by accident of layout, and it costs every identifier a file happens
    to put elsewhere, which is a rule about position pretending to be a rule about
    uniqueness. Position survives as a **tie-break**, not a filter: `detect_column_roles`
    scans left to right and stops at the first candidate, so a real identifier beside an
    all-distinct whole measurement still wins by being leftmost, which is where
    identifiers usually are.

    **What that costs, accepted deliberately.** In a file carrying no identifier at all,
    an all-distinct whole-numbered measurement takes the role -- a lifetime rounded to
    whole picoseconds, say. One measurement out of the pickers until a dropdown puts it
    back, against a position rule that would instead miss a real identifier outright
    (`tests/test_auto_detect_roles.py` pins the last-column case). Both errors cost one
    dropdown, and this one announces itself: the review table shows the guess beside the
    column's own preview before anything is saved.
    """
    if series.isna().any() or series.nunique(dropna=False) != len(series):
        return False
    if pd.api.types.is_bool_dtype(series):
        return False        # a two-level category, however few rows make it distinct
    if not pd.api.types.is_numeric_dtype(series):
        return True
    return _is_whole_number_column(series)


def detect_column_roles(df, guess_row_id=True):
    """Guess `{column: role}` for a table no profile has seen.

    Threshold-free by construction: every rule is a dtype question or a uniqueness
    question. There is no cardinality cutoff separating a categorical from free text,
    because a cutoff is a number to tune whose mistakes are invisible, whereas a
    free-text column guessed as categorical announces itself in the review table --
    1204 distinct values, one dropdown to correct.

    - Numerical: numeric dtype. Run the analysis' own coercion first if you want the
      1% rule applied -- `dataset_io.detect_roles` is the wrapper that does.
    - Row ID: the leftmost candidate column, at most one. Candidacy reads the values
      alone, never the position or the name -- see _is_row_id_candidate. The scan stops
      at the first hit, which is the whole of "leftmost". `guess_row_id=False` turns
      the rule off for a caller that already has an identifier and is only asking about
      columns it has never seen -- guessing a second one there would pit the guess
      against the profile's own answer, and the leftmost wins.
    - Categorical: everything else, a field-of-view column included -- nothing
      distinguishes one from any other categorical, which is why it has no role.
    - Ignore is guessed for one column only -- see the empty-column branch below.
      Everywhere else it stays a decision rather than an assumption.
    """
    roles = {}
    row_id_taken = not guess_row_id
    for col in df.columns:
        series = df[col]
        if series.isna().all():
            # The one column Ignore is guessed for, and not an assumption about what
            # it means: check_and_fix_df removes an all-empty column before
            # get_features ever sees it, so this only stops the review table from
            # offering it as a measurement -- pandas types an empty CSV column
            # float64, which otherwise reads as one. The column still counts towards
            # the profile's known set, because whether it happens to be blank in this
            # file is a fact about the data and not about the table's shape.
            roles[col] = ROLE_IGNORE
        elif not row_id_taken and _is_row_id_candidate(series):
            roles[col] = ROLE_ROW_ID
            row_id_taken = True
        elif pd.api.types.is_numeric_dtype(series):
            roles[col] = ROLE_NUMERICAL
        else:
            roles[col] = ROLE_CATEGORICAL
    return roles


# Tried in no particular order -- whichever occurs *earliest* in a given name wins, so
# there is no precedence between them to argue about. A name like "nadh_t1.mean" is
# cut at the underscore because that is where its structure starts.
#
# No hyphen, deliberately. These three mark structure; a hyphen usually sits *inside* a
# word -- "E-cadherin", "anti-PD1", "t-SNE", "2026-08-27" -- so cutting there names the
# group "E" or "2026". A junk name costs more than a missing one because it is sticky:
# rule 1 recruits every later column that starts with it, and Save writes it into the
# profile. A hyphen inside a prefix is kept, so "anti-PD1_dose" still groups on
# "anti-PD1".
_GROUP_SEPARATORS = (": ", "_", ".")


def _prefix(name):
    """The part of a column name before its earliest separator, or None.

    `> 0` rather than `>= 0`: a name that *starts* with a separator has an empty
    prefix, which is not a group name.
    """
    found = [(name.find(sep), sep) for sep in _GROUP_SEPARATORS]
    found = [(at, sep) for at, sep in found if at > 0]
    if not found:
        return None
    return name[:min(found)[0]]


def sibling_groups(keys_and_groups):
    """`{key: group}` for the keys whose known columns agree on a single group.

    Rule 1 of detect_column_groups, generalised from *names* to *filings*. A group the
    user renamed no longer matches any column's key -- but the columns already sitting
    in it still point at it, and they are the best evidence of where a new column like
    them belongs. Callers supply their own notion of key, because the key is whatever
    would otherwise have named the group: the prefix here, the `{extractor}_{channel}`
    string in dataset_io.detect_groups.

    A key whose columns sit in two different groups is dropped rather than resolved: the
    user has demonstrated that this key is not their grouping axis, and picking one of
    the two would make the answer depend on an order they cannot see. A `None` key is
    skipped -- a column like "Area" carries nothing to match on.
    """
    by_key = {}
    for key, group in keys_and_groups:
        if key is None:
            continue
        by_key.setdefault(key, set()).add(group)
    return {key: next(iter(groups))
            for key, groups in by_key.items() if len(groups) == 1}


def detect_column_groups(columns, existing_groups=None, known_groups=None):
    """Guess `{column: group}` for columns that have no group yet.

    Only the columns that get one appear in the result; the rest are simply absent and
    fall to Uncategorized Features downstream, as an ungrouped column always has.

    The caller passes **new** columns only. A column the profile already knows keeps
    the group it was stored with, and nothing here may overwrite that.

    Guessing freely is affordable here in a way it is not for roles: a wrong role
    removes a measurement from the analysis, while a wrong group sorts a dropdown
    oddly. Nothing is lost, no plot changes, and the mistake is visible in the same
    table that made it.

    Three rules, in order:

    1. **Follow the siblings** -- a column whose prefix is one a *known* column already
       carries joins that column's group, whatever the group is called. See
       sibling_groups: this is the rule that survives a rename, which is the correction
       the review table invites.
    2. **Join an existing group by name** -- a column whose prefix is the name of a
       group the profile already has joins it, alone if need be. Nothing is invented:
       the group exists because the user made it. Still earns its place beside rule 1,
       for a group whose columns carry no prefix of their own ("Area" in "morphology").
    3. **Form groups from shared prefixes** -- among the rest, a prefix carried by two
       or more columns becomes a group. "Two or more" is not a threshold to tune: a
       group of one is not a group, by definition rather than by choice.

    `existing_groups` may be the profile's `{group: [columns]}` mapping or any iterable
    of group names -- only the names are read. `known_groups` is the other direction,
    `{column: group}`, because rule 1 needs the columns.
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
    """A column name or cell value as a Markdown code span, so it reads as it is spelled.

    Streamlit's status widgets -- `st.error`, `st.warning`, `st.info`, `st.toast` -- render
    their body as Markdown, and these messages interpolate names and values straight out
    of the user's file. A column called `*note*` arrived as *note*: italics, with the
    asterisks eaten, so the message named a column the file does not contain, and a
    repeated value of `__x__` came out as a bold `x`.

    A code span rather than backslash escapes, for two reasons. Inside one, every `*`,
    `_` and `[` is literal by definition, so nothing has to decide per character which
    are dangerous *here* -- and `nadh_t1_mean` is not italic but `_id_` is, a distinction
    no interpolation site should have to make. And a leaked backslash would show, where a
    leaked chip is just monospace: the failure mode is legible either way, which matters
    because nothing checks the rendering. The monospace also says "this is your file's
    own text" without a sentence spent saying so.

    Wrapped at the interpolation point, never over a whole message: the `**...**` in the
    at-the-cap message is ours and deliberate. Same rule `vis/helpers.hover_field` follows
    for hovertemplates.

    The fence is one backtick longer than the longest run inside the text, and content
    that begins or ends with a backtick is padded with the space CommonMark then strips --
    the two rules that make a name containing backticks survive.
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


def column_preview(series):
    """One line describing what is actually in a column, for the review table.

    The whole defence against auto-detect's guesses: a free-text column guessed as
    Categorical announces itself here as 1204 levels, and one dropdown corrects it. That
    is what buys the design its freedom from cardinality thresholds, so this has to
    describe the column rather than summarise it flatteringly.

    A range rather than a first value for measurements, because the mistakes worth
    catching at a glance are of scale -- nanoseconds read as picoseconds, a column of
    zeros -- and a single value shows neither.
    """
    values = series.dropna()
    if values.empty:
        # Said in the future tense on purpose: check_and_fix_df drops the column at
        # step 6 and says so again, which reads as a sequence rather than a repeat.
        return "empty — will be dropped"
    if pd.api.types.is_numeric_dtype(series):
        low, high = values.min(), values.max()
        return _number(low) if low == high else f"{_number(low)} – {_number(high)}"
    levels = values.nunique()
    return f"{values.iloc[0]} ({levels} level{'' if levels == 1 else 's'})"


def enforce_role_invariants(roles, groups, numeric_cols=(), previous_roles=None):
    """Apply the rules a single dropdown cannot express, returning `(roles, groups, notices)`.

    A dropdown constrains one cell at a time and knows nothing about the others, so "at
    most one Row ID" and "a group belongs to a measurement" can only be enforced after the
    edit comes back. Pure so that every combination can be put to it directly, rather than
    through the rerun-and-re-key round trip the table needs to show the repair.

    `previous_roles` decides which column *keeps* a contested role -- the one just
    assigned wins, since that is the click the user made. `numeric_cols` decides where
    the loser lands: back to Numerical if it holds measurements, Categorical otherwise.
    Demoting an integer measurement to Categorical would quietly take it out of the
    analysis, which is the same error auto-detect is written to avoid.
    """
    roles = dict(roles)
    previous = previous_roles or {}
    notices = []
    for role in _SOLE_ROLES:
        holders = [col for col, held in roles.items() if held == role]
        if len(holders) < 2:
            continue
        # With no previous state to compare against -- a working copy just rebuilt, a
        # profile whose lists disagree -- keep the leftmost, so the answer does not
        # depend on a dictionary order the user cannot see.
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
    """Why this table cannot be analysed yet, or "" when it can.

    get_features says the same thing two screens later, as "No feature found in the
    uploaded file". The review table is where it belongs: this is the screen holding the
    dropdown that fixes it.
    """
    if not any(role == ROLE_NUMERICAL for role in roles.values()):
        return ("No column is marked Numerical, so there would be nothing to plot. "
                "Mark at least one measurement column Numerical.")
    return ""


def row_id_notice(roles):
    """What will identify a row when no column does, or "" when one is marked Row ID.

    The other half of the sentence `_row_id_reason` offers on the way out: that one says
    the role may be given to no column, this says what happens when it is. Stated rather
    than warned about -- a table with no identifier of its own is an ordinary table, and
    `resolve_row_id_col` numbers its rows 1..N -- but not left silent, because the
    numbering is invented after the gate closes and first appears in hover text, under a
    column the file never had. Short because it sits directly under the table, which is
    already showing that no row holds the role.
    """
    if any(role == ROLE_ROW_ID for role in roles.values()):
        return ""
    return "Rows will be identified by row number."
