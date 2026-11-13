"""One upload's read and interpreted tables, kept for the session instead of rebuilt every rerun.

Reading and interpreting a 1M-row table took 0.75 s of every rerun. The state mapping is
passed in (the page gives st.session_state), so this module needs no Streamlit.
The frames it hands out are shared between reruns: callers must not modify them.
"""
from dataclasses import dataclass

from src import dataset_io

STATE_KEY = "_table_session"


@dataclass
class TableSession:
    key: tuple
    read: tuple                     # read_table's (raw, read_meta, delimiter, scope_warning, error)
    released: bool = False          # raw was swapped for its zero-row header once a decision was confirmed
    decision_key: tuple | None = None
    interpretation: tuple | None = None   # interpret_frame's six values


def upload_key(uploaded_file, generation):
    """Identify an upload: Streamlit's file_id, else the review's file generation with name and size."""
    file_id = getattr(uploaded_file, "file_id", None)
    if file_id is not None:
        return (file_id,)
    return (generation, getattr(uploaded_file, "name", None), getattr(uploaded_file, "size", None))


def current_read(state, uploaded_file, generation, need_table=True):
    """Return read_table's five values, reading the file once per upload.

    A refusal or structural error is kept like a success, so it stays on screen until
    the file is uploaded again. An exception propagates and keeps nothing. After
    release_raw the table is its zero-row header; `need_table` reads the file again.
    """
    key = upload_key(uploaded_file, generation)
    session = state.get(STATE_KEY)
    if session is None or session.key != key:
        session = state[STATE_KEY] = TableSession(key, dataset_io.read_table(uploaded_file))
    elif session.released and need_table:
        session.read, session.released = dataset_io.read_table(uploaded_file), False
    return session.read


def release_raw(state):
    """Keep only the raw table's header once a decision is interpreted.

    The analysis frame is all the plots read; holding the raw table too doubles a
    1M-row session (341 MB at 13 columns). Review reads the file again when it reopens.
    """
    session = state.get(STATE_KEY)
    if session and not session.released and session.interpretation is not None and session.read[0] is not None:
        raw, *rest = session.read
        session.read, session.released = (raw.iloc[:0], *rest), True


def current_interpretation(state, uploaded_file, generation, args, scope_warning):
    """Return (frame, feature_groups, upload_complete, row_id_col) for the confirmed decision.

    `args` is working_copy_arguments' result. The table is interpreted once per decision;
    its messages are drawn on every call, as interpret_table drew them.
    """
    current_read(state, uploaded_file, generation, need_table=False)
    session = state[STATE_KEY]
    decision_key = (tuple(args["categorical_cols"]), args["unique_row_id_col"], tuple(args["ignored_cols"]),
                    tuple((group, tuple(columns)) for group, columns in args["feature_groups"].items()),
                    scope_warning)
    if session.decision_key != decision_key:
        raw = current_read(state, uploaded_file, generation)[0]
        interpretation = dataset_io.interpret_frame(
            raw, args["categorical_cols"], args["unique_row_id_col"], None,
            ignored_cols=args["ignored_cols"], feature_groups=args["feature_groups"],
            scope_warning=scope_warning, use_data_extraction=False)
        session.decision_key, session.interpretation = decision_key, interpretation
    frame, feature_groups, upload_complete, row_id_col, warning_msg, error_msg = session.interpretation
    dataset_io.render_interpretation(upload_complete, warning_msg, error_msg)
    return frame, feature_groups, upload_complete, row_id_col


def clear_table_session(state):
    """Forget the upload and what was fitted on it."""
    state.pop(STATE_KEY, None)
    state.pop("_classification_memo", None)
