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
    decision_key: tuple | None = None
    interpretation: tuple | None = None   # interpret_frame's six values


def upload_key(uploaded_file, generation):
    """Identify an upload: Streamlit's file_id, else the review's file generation with name and size."""
    file_id = getattr(uploaded_file, "file_id", None)
    if file_id is not None:
        return (file_id,)
    return (generation, getattr(uploaded_file, "name", None), getattr(uploaded_file, "size", None))


def current_read(state, uploaded_file, generation):
    """Return read_table's five values, reading the file once per upload.

    A refusal or structural error is kept like a success, so it stays on screen until
    the file is uploaded again. An exception propagates and keeps nothing. The raw
    table is kept for the session: review needs it whenever it reopens, and reading it
    again then could be refused by a budget that the table itself has since lowered.
    """
    key = upload_key(uploaded_file, generation)
    session = state.get(STATE_KEY)
    if session is None or session.key != key:
        session = state[STATE_KEY] = TableSession(key, dataset_io.read_table(uploaded_file))
    return session.read


def current_interpretation(state, uploaded_file, generation, args, scope_warning):
    """Return (frame, feature_groups, upload_complete, row_id_col) for the confirmed decision.

    `args` is working_copy_arguments' result. The table is interpreted once per decision;
    its messages are drawn on every call, as interpret_table drew them.
    """
    raw = current_read(state, uploaded_file, generation)[0]
    session = state[STATE_KEY]
    decision_key = (tuple(args["categorical_cols"]), args["unique_row_id_col"], tuple(args["ignored_cols"]),
                    tuple((group, tuple(columns)) for group, columns in args["feature_groups"].items()),
                    scope_warning)
    if session.decision_key != decision_key:
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
