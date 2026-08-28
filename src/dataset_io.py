import csv
import html
import io
import re

import pandas as pd
import streamlit as st
from pandas.errors import EmptyDataError, ParserError

from src.config import get_all_feature_extractors
from src.emojis import happy_emoji, sad_emoji
from src.widgets.analysis_config_widgets import (
    get_all_feature_groups,
    get_fov_name_col_analysis,
    get_unique_row_id_col,
)

# Read by calamine. Wider than SUPPORTED_SUFFIXES because an exported script's
# DATA_PATH can point at any of them.
SPREADSHEET_SUFFIXES = (".xlsx", ".xlsm", ".xlsb", ".xls", ".ods")
DELIMITED_SUFFIXES = (".csv", ".tsv", ".txt")
# .xlsb and .xls read fine but are not offered in the uploader: we have no way to
# write one, so there is no real fixture to test either reader against.
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
    r"""The start of the file as text, with every row ending turned into \n.

    csv.reader does not treat a bare \r as a row ending, and Excel's
    "CSV (Macintosh)" export still writes them.
    """
    uploaded_file.seek(0)
    chunk = uploaded_file.read(_SAMPLE_BYTES)
    uploaded_file.seek(0)
    if isinstance(chunk, bytes):
        chunk = chunk.decode("utf-8", errors="replace")
    return chunk.replace("\r\n", "\n").replace("\r", "\n")


def _splits_consistently(sample, delimiter):
    """Whether `delimiter` gives every sampled row the same field count, above 1.

    Uses csv.reader rather than str.split, so a separator inside a quoted cell
    ("drug;dose") counts as content, the same way pandas reads it.
    """
    rows = _field_counts(sample, delimiter)
    return bool(rows) and len(set(rows)) == 1 and rows[0] > 1


def _field_counts(sample, delimiter):
    """Field count of each non-empty sampled row, minus a last row cut mid-line."""
    rows = []
    for row in csv.reader(io.StringIO(sample), delimiter=delimiter):
        if row:
            rows.append(len(row))
        # Read one row past the cap. If another row follows, the cap ended the
        # loop rather than the sample running out, so every counted row is whole.
        if len(rows) > _SAMPLE_ROWS:
            break
    if len(rows) > _SAMPLE_ROWS:
        return rows[:_SAMPLE_ROWS]
    if len(rows) > 1 and not sample.endswith("\n"):
        return rows[:-1]
    return rows


def _first_ragged_line(uploaded_file, delimiter):
    """First line whose field count differs from row 1's, else None.

    Returns `(line_no, n_fields, expected)`. Checks the whole file, not just the
    sample, because pandas pads a short row with NaN instead of complaining. It
    pads at the end, so if the gap was in the middle of the row every value after
    it lands one column to the left and the frame looks perfectly normal.

    Line numbers count every line including blank ones, so they match what an
    editor shows. Streams the file rather than decoding it into memory again.
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
    """The one separator that parses this file into a rectangular table.

    Returns `(delimiter, viable, present, unusable)`. Exactly one candidate must
    split every sampled row into the same number of fields, above one; two means
    the file is ambiguous and the caller reports that rather than guessing.
    `present` separates a single-column file (fine) from an unevenly split one.

    Reads the sample only. _first_ragged_line checks the rest of the file.

    pandas' `sep=None` is unusable here: it asks csv.Sniffer, which infers from
    any character and picks the space in "Lifetime fit_ch1: T1".
    """
    sample = _sample_text(uploaded_file)
    present = tuple(d for d in _CANDIDATE_DELIMITERS if d in sample)
    viable = tuple(d for d in present if _splits_consistently(sample, d))
    delimiter = viable[0] if len(viable) == 1 else ","
    char, name = _UNUSABLE_DELIMITER
    unusable = name if not viable and _splits_consistently(sample, char) else None
    return delimiter, viable, present, unusable


def drop_unnamed_columns(df):
    """Drop columns whose header cell was blank, unless *no* column has a name.

    pandas names a blank header cell "Unnamed: N", and nothing in the UI can refer
    to one. A frame where nothing has a name is a missing header, not stray columns, so
    it is left alone for the caller to report.

    Streamlit/config-free: getsource()-d into exported scripts.
    """
    unnamed = [col for col in df.columns if str(col).startswith("Unnamed: ")]
    if not unnamed or len(unnamed) == len(df.columns):
        return df
    return df.drop(columns=unnamed)


# Modern spreadsheets are zip containers, legacy .xls is an OLE2 file, and
# neither sequence can start a text table.
_ZIP_MAGIC = b"PK\x03\x04"
_OLE2_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"


def _content_is_spreadsheet(head):
    """Whether the leading bytes are a workbook, whatever the file is named."""
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
    """Cache the raw parse, which is the expensive part of every rerun.

    Returns `(df, meta)`. meta holds what only the reader can see — sheet names,
    separator findings — which _diagnose_table turns into messages. The cache key
    is the file's content, not the analysis profile, so switching profiles never
    serves a stale frame.

    Two traps:
      - st.cache_data keys an UploadedFile on name, seek position and contents, so
        anything that reads the buffer must rewind or every lookup misses. `suffix`
        is a separate argument because a bare BytesIO is keyed on contents alone.
      - the branches need different parameters. `index_col=False` is read_csv-only
        (the Excel parser tests `is not None` and would eat a header), `low_memory`
        is not an Excel parameter, and only read_csv guarantees str headers.
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
        # The sample picked the separator; check the whole file keeps to it. With
        # two qualifying candidates the file is ambiguous, and that is what to
        # report, so neither one's shape is worth checking.
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
            # Blank lines, whitespace, or a lone BOM — what Excel writes when an
            # empty sheet is saved as "CSV UTF-8". The 0-byte guard cannot see it.
            df, parse_error = pd.DataFrame(), "empty"
        except ParserError:
            # The C parser gave up. An unclosed quote and a row with too many
            # fields raise the same error. Quote parity separates them: valid CSV
            # pairs every quote, and a doubled "" escape keeps the count even.
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
    return drop_unnamed_columns(df), meta


def _decodes_as_utf8(head):
    """Whether the sampled bytes are valid UTF-8.

    A character cut in half by the sample boundary is not an encoding fault, so a
    failure in the last 3 bytes of a full sample is allowed. Trimming a fixed
    number of trailing bytes instead just moves the boundary.
    """
    try:
        head.decode("utf-8")
    except UnicodeDecodeError as exc:
        return len(head) == _SAMPLE_BYTES and exc.start >= len(head) - 3
    return True


def _name_content_mismatch(uploaded_file, filename):
    """Error text when the extension and the actual bytes disagree, else "".

    Runs before the read so the message can say both what the file is called and
    what it actually is. One read answers all three questions: empty, workbook,
    decodable.
    """
    uploaded_file.seek(0)
    head = uploaded_file.read(_SAMPLE_BYTES)
    uploaded_file.seek(0)
    # Ordered before the magic-byte check so an empty .xlsx is called empty rather
    # than reported as plain text.
    if not head:
        return (f"Error: '{filename}' is empty (0 bytes). Check that it finished saving or "
                "downloading before uploading it.")

    is_spreadsheet_name = suffix_of(uploaded_file) in SPREADSHEET_SUFFIXES
    is_spreadsheet_content = _content_is_spreadsheet(head)
    if not is_spreadsheet_content and not _decodes_as_utf8(head):
        # Excel writes CP1252 alongside ';' separators, so this is common. Without
        # the check pandas raises and the page shows
        # "'utf-8' codec can't decode byte 0xe9 in position 12".
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


def _join_names(delimiters):
    """"comma", or "comma and semicolon", or "comma, semicolon and pipe"."""
    names = [_DELIMITER_NAMES[d] for d in delimiters]
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " and " + names[-1]


def _diagnose_table(df, read_meta, filename):
    """Explain, in the file's own terms, how it falls outside the supported shape.

    The supported shape is narrow on purpose: first sheet, header on row 1, one
    row per data point. Without these messages every breach shows up as
    get_features()' "No feature found in the uploaded file", which names neither
    the rule broken nor the fix.

    Structural only — whether the table has an identifier and a measurement is
    check_and_fix_df's and get_features' question.
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

    # The parser gave up before there was a frame to describe, so these come first.
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

    # Two viable separators means the file does not say which it uses. None means
    # the comma fallback ran, giving one column named after the whole header line.
    viable = read_meta.get("viable_delimiters")
    if viable is not None:
        if len(viable) > 1:
            error_msg += (
                f"Error: {where} can be read as {_join_names(viable)}-separated, and those give "
                "different columns. Re-export the table using one of them only, so the separator "
                "is unambiguous.\n"
            )
        elif read_meta.get("unusable_delimiter"):
            # It splits the file perfectly evenly; it is just not a character
            # this app can split on.
            error_msg += (
                f"Error: {where} appears to be {read_meta['unusable_delimiter']}-separated. "
                f"{_ALLOWED_SEPARATORS}\n"
            )
        elif read_meta.get("ragged"):
            # A blank between two separators is still a field and loads as NaN, so
            # a missing value never causes this. Do not offer it as a possibility.
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
        # The check above names the offending row whenever it can, so reaching
        # here means it could not and there is nothing more precise to say.
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

    # How many columns a table needs is get_features()' question. The reader only
    # objects when nothing would be left once blank columns go. check_and_fix_df
    # tests the same thing but never reaches it in the app, because the reader gets
    # there first and can name the sheet the frame came from.
    # Avoid df.drop() here: it copies the whole frame to answer a question that
    # needs no copy.
    if df.isna().all().all():
        error_msg += (f"Error: {where} has column names but no rows beneath them. Check that "
                      "the data rows were included when the file was saved.\n")

    return warning_msg, error_msg


# A bare number written the European way: comma for the decimal point.
_COMMA_DECIMAL = re.compile(r"-?\d+,\d+")
_HINT_SAMPLE_ROWS = 200


def _comma_decimal_hint(df):
    """Advice for a table whose numbers use comma decimal points, else "".

    Added to an error the file is already getting, never a rejection on its own:
    "3,4" in a `position` column is a label, and the raw text cannot tell a label
    from a decimal, so refusing the file over it would throw out tables whose
    measurements are fine.

    Runs on the parsed frame, so it covers spreadsheets too. A column counts only
    if every sampled value matches; as advice, a false positive is harmless.
    """
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        values = df[col].dropna().head(_HINT_SAMPLE_ROWS)
        if values.empty:
            continue
        if values.astype(str).str.strip().str.fullmatch(_COMMA_DECIMAL.pattern).all():
            example = str(values.iloc[0]).strip()
            return (f"\nNote: values like '{example}' in '{col}' "
                    "write the decimal point as a comma, so they are "
                    "read as text rather than numbers. Re-export the table with a full stop as "
                    "the decimal point.")
    return ""


def _as_html(msg):
    """Escape a message, then turn its line breaks into <br>.

    The only place a message becomes HTML. Messages are plain text with "\n"
    breaks so this can escape all of them in one go. They carry column names and
    cell values, and without escaping a column named "a<b" cuts its own warning
    short at "Warning: a" — the browser reads the rest as a tag.
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
    """The FOV column the analysis actually has: the configured name, or None.

    Call this *after* check_and_fix_df, never before. The fuzzy categorical match
    can rename a column *into* the configured name ("Image-Name" -> "image_name"),
    and the empty-column rule can drop one that was blank in every row -- so
    presence is only knowable once normalization has run.
    """
    if df is None or not fov_name_col:
        return None
    return fov_name_col if fov_name_col in df.columns else None


def load_table(uploaded_file, categorical_cols, use_data_extraction=True):
    """Load an uploaded table (CSV, delimited text or spreadsheet) and validate it.

    Returns `(df, feature_groups_dict, upload_complete, delimiter)`. The separator
    goes back so the exported script reuses this answer rather than detecting one
    of its own that might differ.

    Two layers, in order. The reader reports structural problems: wrong separator,
    no header, empty sheet, name and content disagreeing. Only then do
    check_and_fix_df and get_features speak about what the columns mean.
    """
    upload_complete = False
    df = feature_groups_dict = None
    delimiter = ","
    if uploaded_file is not None:
        filename = getattr(uploaded_file, "name", "") or "the uploaded file"
        # The read branch is chosen by suffix, so a renamed file reaches the wrong
        # parser and raises something that names neither cause nor fix.
        mismatch = _name_content_mismatch(uploaded_file, filename)
        if mismatch != "":
            _render_reject(mismatch)
            return None, None, False, delimiter
        suffix = suffix_of(uploaded_file)
        df, read_meta = _read_table_cached(uploaded_file, suffix)
        # A spreadsheet sets no delimiter; "," is what the export side uses there.
        delimiter = read_meta.get("delimiter", ",")
        # Kept for the hint below: check_and_fix_df returns None when it fails.
        table = df
        # Name any breach of the supported shape first. Otherwise check_and_fix_df
        # reports a missing identifier, which is the symptom rather than the cause.
        scope_warning, scope_error = _diagnose_table(df, read_meta, filename)
        if scope_error != "":
            _render_reject(scope_error, scope_warning)
            return None, None, False, delimiter
        unique_row_id_col = get_unique_row_id_col(use_data_extraction)
        fov_name_col = get_fov_name_col_analysis(use_data_extraction)
        df, warning_msg, error_msg = check_and_fix_df(df, categorical_cols, unique_row_id_col, fov_name_col)
        warning_msg = scope_warning + warning_msg

        if error_msg != "":
            # scope_warning is context for this error too: when the data sits on
            # sheet 2, "cell_id is missing" only makes sense alongside
            # "'Data' was skipped".
            _render_reject(error_msg + _comma_decimal_hint(table), warning_msg)
        else:
            _render_warning(warning_msg)
            df, feature_groups_dict, warning_msg, error_msg = get_features(df, categorical_cols, use_data_extraction=use_data_extraction)
            if error_msg != "":
                _render_reject(error_msg + _comma_decimal_hint(table))
            else:
                # Only the extraction branch genuinely expects this column -- extraction
                # always emits it. A user-table profile's fov_name_col may be a stale
                # extraction default the table never had, which resolve_effective_fov_col
                # already turns into a silent None; warning about it here would fire on
                # every load of a table that legitimately has no FOV column.
                if use_data_extraction and fov_name_col and resolve_effective_fov_col(df, fov_name_col) is None:
                    warning_msg += (f"Warning: the FOV column '{fov_name_col}' was not found. "
                                    "FOV Comparison is unavailable and the FOV name is left "
                                    "out of hover text.\n")
                _render_warning(warning_msg)
                st.write(f"Data uploaded successfully {happy_emoji}")
                upload_complete = True
    return df, feature_groups_dict, upload_complete, delimiter

def match_col_name(col, col_list):
    """
    match_col_name: a function that takes a column name and a list of canonical column names and returns the first canonical column name that matches the column name
    """
    for col_name in col_list:
        # fuzzy match the column name with the canonical column name
        # e.g. "cell_line", "cell line", "cell-line", "Cell line", "Cell_line", "cell_Lines" all match "cell_line"
        # "treatments", "Treatment", "Treatments" all match "treatment"
        col_processed = col.lower().replace(" ", "_").replace("-", "_")
        # Both sides get the same normalisation, so a configured name may carry the
        # spacing and hyphens of the header it names ("IL-18") and still match.
        col_name_processed = col_name.lower().replace(" ", "_").replace("-", "_")

        # Check for direct match, match after removing/adding 's'
        if (col_processed == col_name_processed or
            (col_processed.endswith('s') and col_processed[:-1] == col_name_processed) or
            (col_name_processed.endswith('s') and col_processed == col_name_processed[:-1])):
            return col_name
    return None

def get_feature_groups_data_extraction(cols):
    """
    feature_groups_dict: a dictionary. Keys are the names of the feature group and values are a list of columns that belong to the group.
    Only feature groups that have at least one column are included in the dictionary.
    """
    all_feature_extractors = get_all_feature_extractors()
    feature_groups_dict = {}
    feature_groups_dict["Uncategorized Features"] = []
    for col in cols:
        # column format: extractor_channelName:feature_name
        # e.g. "Lifetime fit_Channel 1: G(1st)"
        # Derived features form a single cross-channel group; their name has no
        # "{extractor}_{channel}" structure, so bucket them before the splits.
        if col.startswith("Derived: "):
            feature_groups_dict.setdefault("Derived Features", []).append(col)
            continue
        # first split by ":"
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

def get_feature_groups_user_defined(cols):
    all_feature_groups = get_all_feature_groups()
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

        # Only add to uncategorized if it wasn't found in any group
        if not found_group:
            feature_groups_dict["Uncategorized Features"].append(col)

    # Move "Uncategorized Features" to the end of the dictionary
    if "Uncategorized Features" in feature_groups_dict:
        uncategorized = feature_groups_dict.pop("Uncategorized Features")
        if uncategorized:
            feature_groups_dict["Uncategorized Features"] = uncategorized
    return feature_groups_dict

def coerce_majority_numeric_cols(df, skip_cols):
    """
    Attempt to convert non-categorical object columns to numeric.
    Only accept the conversion when <= 1% of non-null values are
    non-numeric (i.e. the column is overwhelmingly numeric with a few
    stray strings like "N/A").  Columns with more than 1% non-numeric
    values are left untouched (likely genuinely categorical/text).

    Must stay free of Streamlit/config dependencies — it is embedded verbatim
    into exported analysis scripts via inspect.getsource().
    """
    warning_msg = ""
    for col in df.columns:
        # A date or duration read from a spreadsheet arrives as datetime64 /
        # timedelta64, and pd.to_numeric turns those into nanoseconds-since-epoch —
        # a plottable "feature" with values like 1.7e18. The same column in a CSV is
        # text and stays out of the feature list, so converting here would also make
        # a workbook and its CSV export disagree about what the data contains.
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

def get_features(df, categorical_cols, use_data_extraction=True):
    """
    Extract all numeric features from the dataframe. Group them (by channel) based on the feature extractors:
    - morphology (mask morphology)
    - texture (texture features)
    - lifetime fit variables
    - lifetime fit free variables
    """
    unique_row_id_col = get_unique_row_id_col(use_data_extraction)
    error_msg = ""

    skip_cols = set([unique_row_id_col] + list(categorical_cols))
    df, warning_msg = coerce_majority_numeric_cols(df, skip_cols)

    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    if use_data_extraction:
        feature_groups_dict = get_feature_groups_data_extraction(numeric_cols)
    else:
        feature_groups_dict = get_feature_groups_user_defined(numeric_cols)
    all_numerical_features_cols = []
    for feature_group, cols in feature_groups_dict.items():
        all_numerical_features_cols.extend(cols)

    if len(all_numerical_features_cols) == 0:
        error_msg += "Error: No feature found in the uploaded file.\n"
        # "" not None: every other return in this module keeps the empty-string
        # convention, and a caller concatenating this warning would raise on None.
        return None, None, "", error_msg

    # keep only the columns that are later used in downstream analysis. The FOV
    # column, when the file has one, arrives through avilable_categorical_cols like
    # any other categorical -- it needs no slot of its own.
    avilable_categorical_cols = [col for col in categorical_cols if col in df.columns]
    columns_to_keep = [unique_row_id_col] + avilable_categorical_cols + all_numerical_features_cols

    # Name what the prune drops: a column that is neither the row id, nor a matched
    # categorical, nor numeric enough for the 1% rule. Up to 5 names, then a count —
    # same shape as the empty-column and NaN-column warnings.
    columns_to_keep_set = set(columns_to_keep)
    dropped = [col for col in df.columns if col not in columns_to_keep_set]
    if dropped:
        listed = ", ".join(dropped[:5])
        more = f" and {len(dropped) - 5} more" if len(dropped) > 5 else ""
        plural = "s" if len(dropped) > 1 else ""
        was_were = "were" if len(dropped) > 1 else "was"
        warning_msg += (f"Warning: {len(dropped)} column{plural} {was_were} not analysed "
                        f"(neither categorical nor numerical): {listed}{more}.\n")

    df = df[columns_to_keep]

    # Print columns that contain NaN values
    columns_with_na = df.columns[df.isna().any()].tolist()
    if columns_with_na:
        num_na_columns = len(columns_with_na)
        if num_na_columns <= 5:
            warning_msg += f"Warning: {', '.join(columns_with_na)} column{'s' if num_na_columns > 1 else ''} contain{'s' if num_na_columns == 1 else ''} NaN values.\n"
        else:
            warning_msg += f"Warning: {', '.join(columns_with_na[:5])} and {num_na_columns - 5} more columns contain NaN values.\n"

    return df, feature_groups_dict, warning_msg, error_msg

def check_and_fix_df(df, categorical_cols, unique_row_id_col, fov_name_col):
    """
    check for df's metadata:
    - single-cell unique_identifier
    - fill in na values for categorical columns

    Must stay free of Streamlit/config dependencies — it is embedded verbatim
    into exported analysis scripts via inspect.getsource().
    """
    warning_msg = error_msg = ""
    df = df.reset_index(drop=True)

    # drop off the all empty columns
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

    # No duplicate-column check: pandas de-duplicates headers while parsing, on both
    # read branches — "cell_id,T1,T1" arrives as ['cell_id', 'T1', 'T1.1'] from
    # read_csv and from read_excel alike — so df.columns.duplicated() is never true
    # for a file that came through a reader, here or in an exported script. Warning
    # about it for real would mean capturing the header row before pandas touches
    # it; the second column is not lost meanwhile, it is simply named 'T1.1'.

    # handle the required unique cell identifier column
    if unique_row_id_col not in df.columns:
        error_msg += f"Error: {unique_row_id_col} column is missing in the uploaded file. It is required. \n"
        return None, warning_msg, error_msg

    if df[unique_row_id_col].duplicated().any():
        original_row_count = len(df)
        first_duplicate = df[unique_row_id_col].duplicated()
        first_duplicate_value = df[unique_row_id_col][first_duplicate].iloc[0]
        first_duplicate_index = df.loc[first_duplicate].index[0]
        warning_msg += (f"Warning: duplicate values found in '{unique_row_id_col}'. The first is "
                        f"'{first_duplicate_value}' at row {first_duplicate_index}. Duplicate rows "
                        "were dropped, only the first was kept. ")
        # drop the duplicate rows, only keep the first one
        df = df.drop_duplicates(subset=[unique_row_id_col], keep="first")
        # after fixing the df, print out the number of rows removed
        rows_removed = original_row_count - len(df)
        if rows_removed > 0:
            warning_msg += f"{rows_removed} rows were removed.\n"

    # make sure unique_row_id_col is of type str
    df[unique_row_id_col] = df[unique_row_id_col].astype(str)
    # A present FOV column is a categorical like any other: the loop below stringifies
    # it and fills "N/A". An absent one is valid — load_table resolves it to None, the
    # plots drop the FOV hover label and the page hides FOV Comparison. Prepended here
    # rather than trusted from the caller because this function is getsource()-inlined
    # into standalone scripts, where the categorical list is a baked literal.
    if fov_name_col and fov_name_col not in categorical_cols:
        categorical_cols = list(categorical_cols) + [fov_name_col]

    for col in df.columns:
        matched_categorical_col = match_col_name(col, categorical_cols)
        if matched_categorical_col is not None:
            # Two headers can normalise to the same canonical name ("IL-18" and "IL_18"
            # both do). Renaming the second would leave two columns sharing a name and
            # every df[name] lookup would return a frame, so it keeps its own name and is
            # reported. An exactly-spelled column is never skipped, so it wins the name
            # regardless of column order.
            if matched_categorical_col != col and matched_categorical_col in df.columns:
                warning_msg += (f"Warning: column '{col}' also reads as the categorical column "
                                f"'{matched_categorical_col}', which is already present. "
                                f"'{col}' was left under its own name.\n")
                continue
            # rename the column to match the canonical categorical column name
            df.rename(columns={col: matched_categorical_col}, inplace=True)
            # fix na values, and make sure all the values are labels rather than numbers
            series = df[matched_categorical_col]
            # A numeric column that has blanks is read as float, so a plate or day number
            # would label itself "1.0" instead of the "1" that was typed. Whole numbers go
            # through a nullable integer cast first to keep the label intact; genuinely
            # fractional values keep their decimals.
            if pd.api.types.is_float_dtype(series):
                real = series.dropna()
                if not real.empty and (real % 1 == 0).all():
                    series = series.astype("Int64")
            # Fill from the original null mask, not by matching the stringified "nan" /
            # "<NA>" - a column with a genuine "nan" label would be caught by that.
            df[matched_categorical_col] = series.astype(str).where(series.notna(), "N/A")

    return df, warning_msg, error_msg