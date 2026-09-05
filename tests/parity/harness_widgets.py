"""Patch Streamlit widgets for headless app-vs-export parity checks.

Resolve inputs from label overrides, session state, or widget defaults so plots
take the requested branches. Override labels must match the app's spelling and case.
"""
import streamlit as st


class _Ctx:
    """Stand-in for the context managers st.columns()/expander()/etc. hand back."""

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _cols(spec=2, **kw):
    n = spec if isinstance(spec, int) else len(spec)
    return [_Ctx() for _ in range(n)]


def patch_streamlit(overrides=None):
    """Patch the input widgets.

    Resolve each input in order: overrides[label], st.session_state[key], then
    the widget's value, selected option, or default selection.
    """
    ov = overrides or {}

    def _label(args, kwargs):
        if "label" in kwargs:
            return kwargs["label"]
        return args[0] if args else ""

    def _session(kwargs):
        """(found, value) for this widget's session-state entry, if it has a key."""
        key = kwargs.get("key")
        if key is not None and key in st.session_state:
            return True, st.session_state[key]
        return False, None

    def checkbox(*a, **k):
        lbl = _label(a, k)
        if lbl in ov:
            return ov[lbl]
        found, val = _session(k)
        return val if found else k.get("value", False)

    def number_input(*a, **k):
        lbl = _label(a, k)
        if lbl in ov:
            return ov[lbl]
        found, val = _session(k)
        return val if found else k.get("value", 0)

    def slider(*a, **k):
        lbl = _label(a, k)
        if lbl in ov:
            return ov[lbl]
        found, val = _session(k)
        return val if found else k.get("value", k.get("min_value", 0))

    def selectbox(*a, **k):
        lbl = _label(a, k)
        if lbl in ov:
            return ov[lbl]
        found, val = _session(k)
        if found:
            return val
        opts = list(k.get("options", a[1] if len(a) > 1 else []))
        return opts[k.get("index", 0)] if opts else None

    def multiselect(*a, **k):
        lbl = _label(a, k)
        if lbl in ov:
            return ov[lbl]
        found, val = _session(k)
        return list(val) if found else list(k.get("default", []) or [])

    def radio(*a, **k):
        return selectbox(*a, **k)

    st.checkbox = checkbox
    st.number_input = number_input
    st.slider = slider
    st.selectbox = selectbox
    st.multiselect = multiselect
    st.radio = radio
    st.columns = _cols
    st.expander = lambda *a, **k: _Ctx()
    st.container = lambda *a, **k: _Ctx()
    st.popover = lambda *a, **k: _Ctx()
    st.tabs = lambda names, **k: [_Ctx() for _ in names]
