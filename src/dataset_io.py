import csv
import html
import io
import re

import pandas as pd
import streamlit as st
from pandas.errors import EmptyDataError, ParserError

from src.column_roles import (
    ROLE_NUMERICAL,
    ROLE_ROW_ID,
    UNGROUPED_LABEL,
    code_span,
    detect_column_groups,
    detect_column_roles,
    validate_roles,
)
from src.config import get_all_feature_extractors
from src.emojis import happy_emoji, sad_emoji
from src.widgets.analysis_config_widgets import (
    get_fov_name_col_analysis,
    get_unique_row_id_col,
)

# Calamine formats accepted by exported scripts, including formats the uploader omits.
SPREADSHEET_SUFFIXES = (".xlsx", ".xlsm", ".xlsb", ".xls", ".ods")
DELIMITED_SUFFIXES = (".csv", ".tsv", ".txt")
# The uploader offers only formats covered by workbook fixtures.
SUPPORTED_SUFFIXES = DELIMITED_SUFFIXES + (".xlsx", ".xlsm", ".ods")

_CANDIDATE_DELIMITERS = ("\t", ";", "|", ",")
_DELIMITER_NAMES = {"\t": "tab", ";": "semicolon", "|": "pipe", ",": "comma"}
# Detected so the error can name it, but never split on: feature names
# contain colons ("Lifetime fit_ch1: T1"). Space is not a candidate because a
# column of names like "John Smith" would look like two fields.
_UNUSABLE_DELIMITER = (":", "colon")

# Enough to reach several data rows without pulling a large file into memory.
_SAMPLE_BYTES = 64 * 1024
_SAMPLE_ROWS = 20


def _sample_text(uploaded_file):
    """Return sampled text with CRLF and CR line endings normalized to LF."""
    uploaded_file.seek(0)
    chunk = uploaded_file.read(_SAMPLE_BYTES)
    uploaded_file.seek(0)
    if isinstance(chunk, bytes):
        chunk = chunk.decode("utf-8", errors="replace")
    return chunk.replace("\r\n", "\n").replace("\r", "\n")


def _splits_consistently(sample, delimiter):
    """Whether the delimiter gives all sampled rows the same field count above one.

    csv.reader respects separators inside quoted cells, matching the parser.
    """
    rows = _field_counts(sample, delimiter)
    return bool(rows) and len(set(rows)) == 1 and rows[0] > 1


def _field_counts(sample, delimiter):
    """Count fields in sampled non-empty records, up to _SAMPLE_ROWS.

    Unless the row cap is reached, omit a final unterminated record because the
    byte sample may have cut it short.
    """
    rows = []
    for row in csv.reader(io.StringIO(sample), delimiter=delimiter):
        if row:
            rows.append(len(row))
        # One record beyond the cap confirms that counted records are complete.
        if len(rows) > _SAMPLE_ROWS:
            break
    if len(rows) > _SAMPLE_ROWS:
        return rows[:_SAMPLE_ROWS]
    if len(rows) > 1 and not sample.endswith("\n"):
        return rows[:-1]
    return rows


def _first_ragged_line(uploaded_file, delimiter):
    """Return (row_number, fields, expected) for the first uneven CSV record, or None.

    Check the whole file because pandas pads short rows with NaN. Record numbers
    include blank records. Stream the input and leave its buffer open and rewound.
    """
    uploaded_file.seek(0)
    stream = io.TextIOWrapper(uploaded_file, encoding="utf-8", errors="replace")
    expected = None
    try:
        for line_no, row in enumerate(csv.reader(stream, delimiter=delimiter), start=1):
            if not row:
                continue
            if expected is None:
                expected = len(row)
            elif len(row) != expected:
                return line_no, len(row), expected
    finally:
        # detach() leaves the buffer open; without it the wrapper closes it.
        stream.detach()
        uploaded_file.seek(0)
    return None


def _resolve_delimiter(uploaded_file):
    """Return (delimiter, viable, present, unusable) from the sampled file.

    Exactly one candidate must split every sampled row into a consistent field
    count above one. Multiple viable candidates are ambiguous; no unique answer
    uses comma as the parser fallback. _first_ragged_line checks the full file.
    Restrict candidates to avoid splitting spaces inside feature names.
    """
    sample = _sample_text(uploaded_file)
    present = tuple(d for d in _CANDIDATE_DELIMITERS if d in sample)
    viable = tuple(d for d in present if _splits_consistently(sample, d))
    delimiter = viable[0] if len(viable) == 1 else ","
    char, name = _UNUSABLE_DELIMITER
    unusable = name if not viable and _splits_consistently(sample, char) else None
    return delimiter, viable, present, unusable


def drop_unnamed_columns(df):
    """Drop pandas-generated "Unnamed: N" headers unless every header is unnamed.

    Preserve wholly unnamed tables for the caller's missing-header error.
    Keep this helper Streamlit/config-free for embedding in exported scripts.
    """
    unnamed = [col for col in df.columns if str(col).startswith("Unnamed: ")]
    if not unnamed or len(unnamed) == len(df.columns):
        return df
    return df.drop(columns=unnamed)


# Workbook container signatures used to distinguish binary content from text.
_ZIP_MAGIC = b"PK\x03\x04"
_OLE2_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"


def _content_is_spreadsheet(head):
    """Whether the leading bytes match a ZIP or OLE2 container signature."""
    return head.startswith((_ZIP_MAGIC, _OLE2_MAGIC))


def suffix_of_name(name):
    """Lowercased extension of a filename, ".csv" for anything without one."""
    head, sep, ext = (name or "").rpartition(".")
    return f".{ext.lower()}" if sep and head else ".csv"


def suffix_of(uploaded_file):
    """Lowercased extension of an upload, ".csv" for anything without one."""
    return suffix_of_name(getattr(uploaded_file, "name", ""))


@st.cache_data(show_spinner=False)
def _read_table_cached(uploaded_file, suffix):
    """Cache the raw parse and return (df, reader_metadata).

    Metadata records sheets, delimiters, and structural issues for _diagnose_table.
    Rewind upload buffers around reads: Streamlit includes UploadedFile position
    in cache keys. Pass suffix separately because bare BytesIO keys use contents.
    Keep CSV-only parser options out of the spreadsheet branch and stringify its
    headers explicitly.
    """
    uploaded_file.seek(0)
    if suffix in SPREADSHEET_SUFFIXES:
        # One open gives both the sheet names and the first sheet's frame.
        with pd.ExcelFile(uploaded_file, engine="calamine") as workbook:
            sheet_names = [str(name) for name in workbook.sheet_names]
            df = workbook.parse(sheet_name=0)
        df.columns = [str(col) for col in df.columns]
        meta = {"sheets": sheet_names}
    else:
        delimiter, viable, present, unusable = _resolve_delimiter(uploaded_file)
        # Check full-file field counts unless the delimiter is ambiguous.
        ragged = None
        for char in ([delimiter] if len(viable) == 1 else () if viable else present):
            found = _first_ragged_line(uploaded_file, char)
            if found:
                line_no, n_fields, expected = found
                ragged = (_DELIMITER_NAMES[char], expected, line_no, n_fields)
                break
        parse_error = None
        try:
            df = pd.read_csv(uploaded_file, index_col=False, sep=delimiter, low_memory=False)
        except EmptyDataError:
            # Empty text can contain whitespace or a BOM despite having nonzero size.
            df, parse_error = pd.DataFrame(), "empty"
        except ParserError:
            # Odd quote parity suggests an unclosed quoted value; doubled escapes
            # contribute an even count.
            uploaded_file.seek(0)
            odd_quotes = uploaded_file.read().count(b'"') % 2 == 1
            uploaded_file.seek(0)
            df = pd.DataFrame()
            parse_error = "unbalanced_quote" if odd_quotes else "unreadable"
        meta = {"delimiter": delimiter, "viable_delimiters": viable,
                "present_delimiters": present, "unusable_delimiter": unusable,
                "ragged": ragged, "parse_error": parse_error}

    unnamed = [col for col in df.columns if str(col).startswith("Unnamed: ")]
    meta["unnamed_count"] = len(unnamed)
    meta["all_unnamed"] = bool(unnamed) and len(unnamed) == len(df.columns)
    # Record original positions before dropping unnamed columns. Stringifying
    # spreadsheet headers can make distinct headers, such as 1 and "1", collide.
    positions = {}
    for position, col in enumerate(df.columns, start=1):
        positions.setdefault(str(col), []).append(position)
    meta["duplicate_names"] = {name: at for name, at in positions.items() if len(at) > 1}
    return drop_unnamed_columns(df), meta


def _decodes_as_utf8(head):
    """Whether sampled bytes decode as UTF-8.

    Allow failures in the final three bytes of a full sample to tolerate a
    multibyte character cut by the sample boundary.
    """
    try:
        head.decode("utf-8")
    except UnicodeDecodeError as exc:
        return len(head) == _SAMPLE_BYTES and exc.start >= len(head) - 3
    return True


def _name_content_mismatch(uploaded_file, filename):
    """Return a format or encoding error from the file's leading bytes, otherwise "".

    Check for empty content and extension/content mismatches before choosing a parser.
    """
    uploaded_file.seek(0)
    head = uploaded_file.read(_SAMPLE_BYTES)
    uploaded_file.seek(0)
    # Report empty files before classifying their format.
    if not head:
        return (f"Error: '{filename}' is empty (0 bytes). Check that it finished saving or "
                "downloading before uploading it.")

    is_spreadsheet_name = suffix_of(uploaded_file) in SPREADSHEET_SUFFIXES
    is_spreadsheet_content = _content_is_spreadsheet(head)
    if not is_spreadsheet_content and not _decodes_as_utf8(head):
        # Report an actionable encoding error before the parser runs.
        return (f"Error: '{filename}' is not saved as UTF-8 text, so its accented or special "
                "characters cannot be read. In Excel, save it as \"CSV UTF-8 (Comma "
                "delimited)\".")
    if is_spreadsheet_name and not is_spreadsheet_content:
        return (f"Error: '{filename}' is named like a spreadsheet but its contents are plain "
                "text. Rename it to .csv, .tsv or .txt, or open it in Excel and save it as a "
                "real .xlsx workbook.")
    if is_spreadsheet_content and not is_spreadsheet_name:
        return (f"Error: '{filename}' is named like a text file but its contents are a "
                "spreadsheet. Rename it to .xlsx, .xlsm or .ods to match, or open it in "
                "Excel and export it as a CSV. A legacy .xls or .xlsb has to be re-saved "
                "as .xlsx first.")
    return ""


_ALLOWED_SEPARATORS = "Columns can be separated by comma, tab, semicolon or pipe."


def _join_items(items):
    """"2", or "2 and 3", or "2, 3 and 7"."""
    items = [str(item) for item in items]
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


def _join_names(delimiters):
    """"comma", or "comma and semicolon", or "comma, semicolon and pipe"."""
    return _join_items(_DELIMITER_NAMES[d] for d in delimiters)


def _diagnose_table(df, read_meta, filename):
    """Return (warnings, errors) for structural problems in the parsed table.

    Read the first sheet with headers on row 1 and one row per data point.
    Identifier and measurement validation belongs to the interpretation stage.
    """
    warning_msg = error_msg = ""
    sheets = read_meta.get("sheets") or []
    sheet = sheets[0] if sheets else None

    if len(sheets) > 1:
        skipped = ", ".join(repr(name) for name in sheets[1:])
        warning_msg += (
            f"Warning: '{filename}' has {len(sheets)} sheets and only the first one, "
            f"'{sheet}', was read ({skipped} skipped). Move the table you want to "
            "analyse to the first sheet.\n"
        )

    where = f"sheet '{sheet}' of '{filename}'" if sheet else f"'{filename}'"

    # Report parse failures before inspecting the frame.
    parse_error = read_meta.get("parse_error")
    if parse_error == "empty":
        error_msg += (
            f"Error: no table could be read from {where} — it contains no column names and no "
            "rows. Check that it is not blank, and that it was saved with its header row.\n"
        )
        return warning_msg, error_msg
    if parse_error == "unbalanced_quote":
        error_msg += (
            f"Error: {where} has an unclosed quotation mark — one that opens a value and is "
            "never closed, swallowing every row after it.\n"
        )
        return warning_msg, error_msg

    if read_meta.get("all_unnamed"):
        error_msg += (
            f"Error: no column in {where} has a name — row 1 is blank right across the table. "
            "Row 1 is read as the column names and every row below it as one data point, so any "
            "title, blank or merged rows above the table have to be removed.\n"
        )
        return warning_msg, error_msg

    # A text file must have an unambiguous supported delimiter.
    viable = read_meta.get("viable_delimiters")
    if viable is not None:
        if len(viable) > 1:
            error_msg += (
                f"Error: {where} can be read as {_join_names(viable)}-separated, and those give "
                "different columns. Re-export the table using one of them only, so the separator "
                "is unambiguous.\n"
            )
        elif read_meta.get("unusable_delimiter"):
            error_msg += (
                f"Error: {where} appears to be {read_meta['unusable_delimiter']}-separated. "
                f"{_ALLOWED_SEPARATORS}\n"
            )
        elif read_meta.get("ragged"):
            # Blank values still count as fields; this error concerns separators.
            name, expected, line_no, n_fields = read_meta["ragged"]
            error_msg += (
                f"Error: {where} is not rectangular: with {name}, row {line_no} has {n_fields} "
                f"field{'s' if n_fields != 1 else ''} where row 1 has {expected}. Look for a "
                "missing or extra separator there, or a title or comment line above the column "
                "names.\n"
            )
        if error_msg:
            return warning_msg, error_msg

    if parse_error == "unreadable":
        # Use a general parser error when the structural checks cannot be more specific.
        error_msg += (
            f"Error: {where} could not be read as a table. Check that every row has the same "
            "columns as the header, and that quoted values are closed.\n"
        )
        return warning_msg, error_msg

    skipped = read_meta.get("unnamed_count", 0)
    if skipped:
        plural = "s" if skipped > 1 else ""
        warning_msg += (
            f"Warning: {skipped} column{plural} in {where} had no name in row 1 and "
            f"{'were' if skipped > 1 else 'was'} skipped. If the table has title, blank or merged "
            "rows above its column names, remove them.\n"
        )

    # Downstream df[name] access requires a Series. Reject duplicate names
    # before review so profile assignments remain unambiguous.
    duplicates = read_meta.get("duplicate_names") or {}
    if duplicates:
        for name, at in duplicates.items():
            both = "both" if len(at) == 2 else "all"
            error_msg += (
                f"Error: columns {_join_items(at)} of {where} are {both} named '{name}'. "
                "Every column needs a name of its own. Rename all but one of them in "
                "row 1.\n")
        if sheet:
            # pandas suffixes parsed duplicate headers; spreadsheet header
            # stringification can still introduce a collision afterward.
            error_msg += (
                "Two header cells can hold the same characters and still be stored "
                "differently -- the number 1 and the text 1 look identical on screen -- "
                "so check the header row's cell formatting as well as its text.\n")
        return warning_msg, error_msg

    # Diagnose a table with no populated cells without copying the frame.
    if df.isna().all().all():
        error_msg += (f"Error: {where} has column names but no rows beneath them. Check that "
                      "the data rows were included when the file was saved.\n")

    return warning_msg, error_msg


# Bare decimal numbers written with a comma decimal separator.
_COMMA_DECIMAL = re.compile(r"-?\d+,\d+")
_HINT_SAMPLE_ROWS = 200


def _plain_quoted(text):
    """A name or value set apart the way the reader's plain-text messages do it."""
    return f"'{text}'"


def _comma_decimal_hint(df, mark=_plain_quoted):
    """Suggest decimal-point formatting when a text column looks comma-decimal.

    This supplements an existing error; matching values can also be valid labels.
    Every sampled non-null value must match. `mark` formats column names for the
    caller's renderer: plain quotes for HTML-escaped reader messages or code_span
    for Markdown review messages.
    """
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        values = df[col].dropna().head(_HINT_SAMPLE_ROWS)
        if values.empty:
            continue
        if values.astype(str).str.strip().str.fullmatch(_COMMA_DECIMAL.pattern).all():
            example = str(values.iloc[0]).strip()
            return (f"\nNote: values like '{example}' in {mark(col)} "
                    "write the decimal point as a comma, so they are "
                    "read as text rather than numbers. Re-export the table with a full stop as "
                    "the decimal point.")
    return ""


def _as_html(msg):
    """Escape plain-text messages, then replace newline runs with <br>.

    Escape names and cell values here so embedded markup remains visible text.
    """
    return re.sub(r"\n+", "<br>", html.escape(msg.strip()))


def _render_warning(warning_msg):
    """Render accumulated warnings."""
    if warning_msg:
        st.markdown(f"<h5 style='text-align: center; color: orange'>{_as_html(warning_msg)}</h5>",
                    unsafe_allow_html=True)


def _render_reject(error_msg, warning_msg=""):
    """Render a rejection, warnings first — they are context for the error."""
    _render_warning(warning_msg)
    st.markdown(f"<h5 style='text-align: center; color: red'>{_as_html(error_msg)}</h5>",
                unsafe_allow_html=True)
    st.write(f"Therefore, we cannot extract data from your uploaded file {sad_emoji}")


def resolve_effective_fov_col(df, fov_name_col):
    """Return the configured FOV column if present, otherwise None.

    Call after check_and_fix_df, which can remove an all-empty FOV column.
    """
    if df is None or not fov_name_col:
        return None
    return fov_name_col if fov_name_col in df.columns else None


def resolve_row_id_col(df, unique_row_id_col):
    """Return (df, row_id_col), inserting string row numbers when no ID is named.

    A configured ID must already be validated by check_and_fix_df. Otherwise add
    1-based row numbers in place, using a name unique among the remaining columns.
    Call after empty columns are removed so naming is deterministic.

    Keep this helper Streamlit/config-free: inspect.getsource embeds it in
    exported scripts, which must generate the same identifier as the app.
    """
    if unique_row_id_col:
        return df, unique_row_id_col
    # Avoid collisions with existing column names.
    row_id_col = "Row number"
    suffix = 0
    while row_id_col in df.columns:
        suffix += 1
        row_id_col = f"Row number.{suffix}"
    # String labels keep row numbers integral in hover text and CSV exports.
    df.insert(0, row_id_col, [str(i) for i in range(1, len(df) + 1)])
    return df, row_id_col


def read_table(uploaded_file):
    """Read an upload and return (df, metadata, delimiter, warnings, error).

    Validate structure without consulting column roles or profiles, so review can
    inspect the file's headers first. Return messages without rendering them;
    an empty error means the file is structurally usable.
    """
    delimiter = ","
    filename = getattr(uploaded_file, "name", "") or "the uploaded file"
    # Validate content before choosing a parser from the suffix.
    mismatch = _name_content_mismatch(uploaded_file, filename)
    if mismatch != "":
        return None, {}, delimiter, "", mismatch
    suffix = suffix_of(uploaded_file)
    df, read_meta = _read_table_cached(uploaded_file, suffix)
    # Spreadsheet exports use comma as their text delimiter.
    delimiter = read_meta.get("delimiter", ",")
    # Structural errors must be resolved before column roles are applied.
    scope_warning, scope_error = _diagnose_table(df, read_meta, filename)
    if scope_error != "":
        return None, read_meta, delimiter, scope_warning, scope_error
    return df, read_meta, delimiter, scope_warning, ""


def interpret_table(df, categorical_cols, unique_row_id_col, fov_name_col,
                    ignored_cols=None, feature_groups=None, scope_warning="",
                    use_data_extraction=True):
    """Validate and prepare a parsed frame using the supplied column roles and groups.

    Return (df, feature_groups, upload_complete, row_id_col), including the resolved
    identifier name when row numbers are generated. The caller supplies the
    confirmed column decisions independently of the current profile.
    Render structural warnings before feature-level warnings.
    """
    # Kept for the hint below: check_and_fix_df returns None when it fails.
    table = df
    df, warning_msg, error_msg = check_and_fix_df(df, categorical_cols, unique_row_id_col, fov_name_col)
    warning_msg = scope_warning + warning_msg
    if error_msg != "":
        # Include sheet-scope warnings as context for interpretation errors.
        _render_reject(error_msg + _comma_decimal_hint(table), warning_msg)
        return None, None, False, unique_row_id_col

    _render_warning(warning_msg)
    # Resolve after normalization and before feature selection excludes the ID.
    df, row_id_col = resolve_row_id_col(df, unique_row_id_col)
    df, feature_groups_dict, warning_msg, error_msg = get_features(
        df, categorical_cols, use_data_extraction=use_data_extraction,
        unique_row_id_col=row_id_col, ignored_cols=ignored_cols,
        feature_groups=feature_groups)
    if error_msg != "":
        _render_reject(error_msg + _comma_decimal_hint(table))
        return None, None, False, row_id_col

    # Only extraction config designates a FOV column; user tables may omit one.
    if use_data_extraction and fov_name_col and resolve_effective_fov_col(df, fov_name_col) is None:
        warning_msg += (f"Warning: the FOV column '{fov_name_col}' was not found. "
                        "The FOV name is left out of hover text.\n")
    _render_warning(warning_msg)
    st.write(f"Data uploaded successfully {happy_emoji}")
    return df, feature_groups_dict, True, row_id_col


def load_table(uploaded_file, categorical_cols):
    """Load and validate an extraction table using extraction configuration.

    Return (df, feature_groups, upload_complete, delimiter, row_id_col). Exported
    scripts reuse the detected delimiter. User tables call read_table and
    interpret_table separately so the review gate can supply their working copy.
    """
    unique_row_id_col = get_unique_row_id_col(use_data_extraction=True)
    # Rejections return the configured name until an identifier has been resolved.
    row_id_col = unique_row_id_col
    if uploaded_file is None:
        return None, None, False, ",", row_id_col

    df, _read_meta, delimiter, scope_warning, error_msg = read_table(uploaded_file)
    if error_msg != "":
        _render_reject(error_msg, scope_warning)
        return None, None, False, delimiter, row_id_col

    fov_name_col = get_fov_name_col_analysis(use_data_extraction=True)
    df, feature_groups_dict, upload_complete, row_id_col = interpret_table(
        df, categorical_cols, unique_row_id_col, fov_name_col,
        scope_warning=scope_warning)
    return df, feature_groups_dict, upload_complete, delimiter, row_id_col

def get_feature_groups_data_extraction(cols):
    """Group features by extractor/channel names, with a shared Derived Features group.

    Return nonempty {group: [columns]} entries, placing Uncategorized Features last.
    """
    all_feature_extractors = get_all_feature_extractors()
    feature_groups_dict = {}
    feature_groups_dict["Uncategorized Features"] = []
    for col in cols:
        # Extracted names use "extractor_channel: feature". Derived features
        # share a cross-channel group and bypass that parsing.
        if col.startswith("Derived: "):
            feature_groups_dict.setdefault("Derived Features", []).append(col)
            continue
        try:
            extractor_channel, feature = col.split(": ")
        except Exception:
            feature_groups_dict["Uncategorized Features"].append(col)
            continue
        try:
            extractor, channel = extractor_channel.split("_", 1)
        except Exception:
            feature_groups_dict["Uncategorized Features"].append(col)
            continue
        if extractor in all_feature_extractors:
            if extractor_channel not in feature_groups_dict:
                feature_groups_dict[extractor_channel] = []
            feature_groups_dict[extractor_channel].append(col)
        else:
            feature_groups_dict["Uncategorized Features"].append(col)
    # Move "Uncategorized Features" to the end of the dictionary
    if "Uncategorized Features" in feature_groups_dict:
        uncategorized = feature_groups_dict.pop("Uncategorized Features")
        if uncategorized:
            feature_groups_dict["Uncategorized Features"] = uncategorized

    return feature_groups_dict

def get_feature_groups_user_defined(cols, all_feature_groups):
    """Group columns using the caller's {group: [columns]} mapping.

    Use the first matching group for each column. Unmatched columns, including all
    columns when the mapping is empty, go into Uncategorized Features last.
    The caller supplies working-copy groups; this function reads no profile.
    """
    feature_groups_dict = {}
    feature_groups_dict["Uncategorized Features"] = []

    for col in cols:
        found_group = False
        for feature_group in all_feature_groups:
            cols_in_group = all_feature_groups[feature_group]
            if col in cols_in_group:
                if feature_group not in feature_groups_dict:
                    feature_groups_dict[feature_group] = []
                feature_groups_dict[feature_group].append(col)
                found_group = True
                break  # Column found in this group, no need to check other groups

        if not found_group:
            feature_groups_dict["Uncategorized Features"].append(col)

    # Move "Uncategorized Features" to the end of the dictionary
    if "Uncategorized Features" in feature_groups_dict:
        uncategorized = feature_groups_dict.pop("Uncategorized Features")
        if uncategorized:
            feature_groups_dict["Uncategorized Features"] = uncategorized
    return feature_groups_dict

def _as_the_analysis_reads_it(df):
    """Return a copy with the analysis' 1% numeric-coercion rule applied.

    Share this pass between role guessing and numeric-column detection.
    """
    coerced, _warning = coerce_majority_numeric_cols(df.copy(), set())
    return coerced


def _numeric_names(coerced):
    """The numeric column names of an already-coerced frame."""
    return {col for col in coerced.columns
            if pd.api.types.is_numeric_dtype(coerced[col])}


def detect_roles(df, guess_row_id=True):
    """Guess {column: role} after applying the analysis' numeric coercion to a copy.

    Coercion lets mostly numeric text columns receive the same role the loader uses.
    """
    return detect_column_roles(_as_the_analysis_reads_it(df),
                               guess_row_id=guess_row_id)


def numeric_column_names(df):
    """Return numeric column names after the analysis' 1% coercion rule.

    Used for review validation and demoting contested identifiers. build_working_copy
    returns this set from its existing coercion pass to avoid repeating the conversion.
    """
    return _numeric_names(_as_the_analysis_reads_it(df))


def _row_id_reason(df, roles):
    """Return why a present Row ID column is unusable, otherwise "".

    Check for an all-empty column, null values, then duplicate string identifiers.
    Keep these rules aligned with check_and_fix_df, which must remain standalone
    for script export. Review reports the problem beside the role selector.
    """
    # Row ID is optional; removing the role requests generated row numbers.
    exits = ("Give the role to another column, or to none "
             "(rows will be identified by row number).")
    for col, role in roles.items():
        if role != ROLE_ROW_ID or col not in df.columns:
            continue
        series = df[col]
        if series.isna().all():
            return (f"{code_span(col)} is marked Row ID but is empty in every row, so the "
                    f"table would be left with no identifier. {exits}")
        blank = int(series.isna().sum())
        if blank:
            return (f"{code_span(col)} is marked Row ID but is blank in {blank} of "
                    f"{len(series)} rows, so those rows have nothing identifying them. "
                    f"{exits}")
        # Validate the labels the loader actually uses: an Excel number 1 and
        # text "1" are distinct raw values but become the same identifier.
        series = series.astype(str)
        repeats = series[series.duplicated()]
        if len(repeats):
            first = repeats.iloc[0]
            return (f"{code_span(col)} is marked Row ID but does not identify a row on "
                    f"its own: {code_span(first)} appears "
                    f"{int((series == first).sum())} times, and {len(repeats)} of "
                    f"{len(series)} rows would be dropped as duplicates. {exits}")
    return ""


def _unusable_measurements_reason(df, roles):
    """Return an error when Numerical columns contain no usable numeric data.

    One nonempty numeric column is enough. Empty columns do not count even when
    pandas gives them a numeric dtype, because normalization drops them.
    """
    marked = [col for col, role in roles.items()
              if role == ROLE_NUMERICAL and col in df.columns]
    if not marked:
        return ""
    numeric = numeric_column_names(df)
    if any(col in numeric and not df[col].isna().all() for col in marked):
        return ""
    return ("No column marked Numerical holds numbers, so there would be nothing to "
            "plot. Check the Preview column for what those columns actually contain.")


def review_blocking_reason(df, roles):
    """Return the first role or data error that prevents analysis, otherwise "".

    Validate identifier values and require a usable Numerical column. Append the
    comma-decimal hint only to measurement errors, where it can explain the cause.
    """
    bad_id = _row_id_reason(df, roles)
    if bad_id:
        return bad_id
    reason = validate_roles(roles) or _unusable_measurements_reason(df, roles)
    return reason + _comma_decimal_hint(df, mark=code_span) if reason else ""


def build_working_copy(df, profile_roles=None, profile_groups=None,
                       profile_group_names=None):
    """Build (roles, groups, numeric_columns) for the uploaded file.

    Keep saved roles and groups for columns present in the file, guess assignments
    for new columns, and omit absent columns. Only new Numerical columns receive
    group guesses, using stored groups as sibling evidence.

    profile_group_names includes empty groups that the column mapping cannot
    represent. Share one coerced copy between role guessing and numeric detection.
    All profile state comes from the caller; this function reads no configuration.
    """
    profile_roles = profile_roles or {}
    profile_groups = profile_groups or {}
    # Preserve a stored Row ID only when its column is present. Otherwise allow
    # a new candidate, while avoiding a guess that competes with a retained ID.
    keeps_row_id = any(profile_roles.get(col) == ROLE_ROW_ID for col in df.columns)
    # Reuse one coercion pass for both roles and the returned numeric set.
    coerced = _as_the_analysis_reads_it(df)
    detected = detect_column_roles(coerced, guess_row_id=not keeps_row_id)
    roles = {col: profile_roles.get(col, detected[col]) for col in df.columns}
    groups = {col: profile_groups[col] for col in df.columns
              if col in profile_roles and col in profile_groups}
    fresh = [col for col in df.columns
             if col not in profile_roles and roles[col] == ROLE_NUMERICAL]
    existing = set(profile_groups.values()) | set(profile_group_names or ())
    # User tables use prefix grouping independently of extraction configuration.
    groups.update(detect_column_groups(fresh, existing_groups=existing,
                                       known_groups=profile_groups))
    # Merge the reserved display label into the ungrouped slot so the picker
    # cannot show duplicate labels with different stored values.
    return (roles,
            {col: group for col, group in groups.items()
             if roles.get(col) == ROLE_NUMERICAL and group != UNGROUPED_LABEL},
            _numeric_names(coerced))


def coerce_majority_numeric_cols(df, skip_cols):
    """Convert eligible columns when at most 1% of their non-null values are nonnumeric.

    Leave skipped, already numeric, date, duration, and all-null columns unchanged.
    Accepted conversions replace stray text with NaN and add a warning.
    Keep this helper Streamlit/config-free for inspect.getsource script exports.
    """
    warning_msg = ""
    for col in df.columns:
        # Keep dates and durations out of numeric conversion, which would turn
        # their internal time units into apparent measurements.
        if pd.api.types.is_datetime64_any_dtype(df[col]) or pd.api.types.is_timedelta64_dtype(df[col]):
            continue
        if col not in skip_cols and not pd.api.types.is_numeric_dtype(df[col]):
            converted = pd.to_numeric(df[col], errors='coerce')
            non_null_original = int(df[col].notna().sum())
            if non_null_original == 0:
                continue
            num_coerced = non_null_original - int(converted.notna().sum())
            coerced_pct = num_coerced / non_null_original
            if coerced_pct <= 0.01:
                if num_coerced > 0:
                    warning_msg += f"Warning: {num_coerced} non-numeric value{'s' if num_coerced > 1 else ''} in '{col}' were converted to NaN.\n"
                df[col] = converted
    return df, warning_msg

def get_features(df, categorical_cols, use_data_extraction=True, unique_row_id_col=None,
                 ignored_cols=None, feature_groups=None):
    """Coerce and retain numeric features, grouping them for analysis.

    Extraction groups follow feature names; user-table groups come from the caller.
    Return (df, feature_groups, warnings, error), retaining identifiers and present
    categorical columns alongside the features and excluding ignored measurements.

    Pass the resolved row-ID name so generated digit strings stay out of numeric
    coercion. None resolves it from configuration, generating row numbers if needed.
    """
    if unique_row_id_col is None:
        df, unique_row_id_col = resolve_row_id_col(
            df, get_unique_row_id_col(use_data_extraction))
    error_msg = ""

    # Skip conversion for columns the user explicitly ignores.
    ignored = set(ignored_cols or ())
    skip_cols = set([unique_row_id_col] + list(categorical_cols)) | ignored
    df, warning_msg = coerce_majority_numeric_cols(df, skip_cols)

    # Ignore must also exclude columns that already have a numeric dtype.
    numeric_cols = [col for col in df.columns
                    if pd.api.types.is_numeric_dtype(df[col]) and col not in ignored]
    if use_data_extraction:
        feature_groups_dict = get_feature_groups_data_extraction(numeric_cols)
    else:
        # An empty caller-supplied mapping leaves all features ungrouped.
        feature_groups_dict = get_feature_groups_user_defined(
            numeric_cols, feature_groups or {})
    all_numerical_features_cols = []
    for feature_group, cols in feature_groups_dict.items():
        all_numerical_features_cols.extend(cols)

    if len(all_numerical_features_cols) == 0:
        error_msg += "Error: No feature found in the uploaded file.\n"
        # Warning values stay strings so callers can concatenate them.
        return None, None, "", error_msg

    # Keep the identifier once even if extraction settings also list it as
    # categorical; duplicate selection would make df[name] return a DataFrame.
    # A present FOV column is retained through categorical_cols.
    avilable_categorical_cols = [col for col in categorical_cols
                                 if col in df.columns and col != unique_row_id_col]
    columns_to_keep = [unique_row_id_col] + avilable_categorical_cols + all_numerical_features_cols

    # Report unexpected exclusions, omitting columns explicitly marked Ignore.
    columns_to_keep_set = set(columns_to_keep)
    dropped = [col for col in df.columns
               if col not in columns_to_keep_set and col not in ignored]
    if dropped:
        listed = ", ".join(dropped[:5])
        more = f" and {len(dropped) - 5} more" if len(dropped) > 5 else ""
        plural = "s" if len(dropped) > 1 else ""
        was_were = "were" if len(dropped) > 1 else "was"
        warning_msg += (f"Warning: {len(dropped)} column{plural} {was_were} not analysed: "
                        f"{listed}{more}.\n")
        if use_data_extraction:
            # Extraction categorical names must match the file headers exactly.
            warning_msg += ("If one of these is a categorical feature, add it under "
                            "Categorical Features on the Home page — "
                            "the name must match the column exactly.\n")

    df = df[columns_to_keep]

    # Report missing values in the retained analysis columns.
    columns_with_na = df.columns[df.isna().any()].tolist()
    if columns_with_na:
        num_na_columns = len(columns_with_na)
        if num_na_columns <= 5:
            warning_msg += f"Warning: {', '.join(columns_with_na)} column{'s' if num_na_columns > 1 else ''} contain{'s' if num_na_columns == 1 else ''} NaN values.\n"
        else:
            warning_msg += f"Warning: {', '.join(columns_with_na[:5])} and {num_na_columns - 5} more columns contain NaN values.\n"

    return df, feature_groups_dict, warning_msg, error_msg

def check_and_fix_df(df, categorical_cols, unique_row_id_col, fov_name_col):
    """Remove empty columns, validate a named row ID, and normalize categorical labels.

    Return (df, warnings, error), with df=None on rejection. A blank identifier
    name is allowed; resolve_row_id_col generates one after this step.
    Keep this helper Streamlit/config-free for inspect.getsource script exports.
    """
    warning_msg = error_msg = ""
    df = df.reset_index(drop=True)

    # Remove empty columns before validating named identifiers.
    empty_cols = df.columns[df.isnull().all()]
    if len(empty_cols) > 0:
        if len(empty_cols) <= 5:
            warning_msg += f"Warning: {', '.join(empty_cols)} columns are all empty. They were removed.\n"
        else:
            warning_msg += (f"Warning: {', '.join(empty_cols[:5])} columns and "
                            f"{len(empty_cols) - 5} more are all empty. They were removed.\n")
        df.drop(columns=empty_cols, inplace=True)
        if df.empty:
            error_msg += "Error: No data available after removing empty columns.\n"
            return None, warning_msg, error_msg

    # Header uniqueness is handled by the reader: pandas suffixes duplicate
    # headers, and _diagnose_table rejects collisions after stringification.

    # A named identifier must exist. With no name, resolve_row_id_col generates
    # unique row numbers after this validation step.
    if unique_row_id_col:
        if unique_row_id_col not in df.columns:
            error_msg += f"Error: {unique_row_id_col} column is missing in the uploaded file. It is required. \n"
            return None, warning_msg, error_msg

        # Reject missing or repeated identifiers without dropping data rows.
        # Check nulls before converting them into apparently valid text labels.
        blank_ids = int(df[unique_row_id_col].isna().sum())
        if blank_ids:
            error_msg += (f"Error: '{unique_row_id_col}' is the unique identifier and is "
                          f"blank in {blank_ids} of {len(df)} rows, which leaves those rows "
                          "with nothing identifying them. Name a different column as the "
                          "identifier.\n")
            return None, warning_msg, error_msg

        # Check the exact representation used by plots and downloads, after
        # rejecting blanks so they cannot become apparently valid text IDs.
        row_ids = df[unique_row_id_col].astype(str)
        shared = row_ids.duplicated()
        if shared.any():
            first_value = row_ids[shared].iloc[0]
            repeats = int((row_ids == first_value).sum())
            error_msg += (f"Error: '{unique_row_id_col}' is the unique identifier and does "
                          f"not identify a row on its own: '{first_value}' appears {repeats} "
                          f"times, and {int(shared.sum())} of {len(df)} rows share an "
                          "identifier. Name a different column as the identifier.\n")
            return None, warning_msg, error_msg

        df[unique_row_id_col] = row_ids
    # Include a configured FOV in categorical normalization when present.
    # Do this here as well as in callers because exported scripts inline this helper.
    if fov_name_col and fov_name_col not in categorical_cols:
        categorical_cols = list(categorical_cols) + [fov_name_col]

    # Match exact headers and preserve their original spelling.
    for col in df.columns:
        if col in categorical_cols:
            series = df[col]
            # Preserve integer labels such as "1" when nulls caused a float dtype;
            # fractional values retain their decimals.
            if pd.api.types.is_float_dtype(series):
                real = series.dropna()
                if not real.empty and (real % 1 == 0).all():
                    series = series.astype("Int64")
            # Use the null mask so literal labels such as "nan" remain unchanged.
            df[col] = series.astype(str).where(series.notna(), "N/A")

    return df, warning_msg, error_msg
