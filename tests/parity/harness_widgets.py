"""Patch Streamlit input widgets so app plot functions run headless and take the
same branch the exported script hard-codes.

Several plot functions in src/vis/ render their own widgets mid-plot (the GMM
hyperparameters, the k-means checkbox, the histogram bin width, the 2D marginal
type). Outside `streamlit run` those return None, so the app path would diverge
from the export for reasons that have nothing to do with parity.

Default behaviour returns each widget's OWN default (the `value` kwarg, or
`options[index]`), which is the state the app is in right after first render —
exactly what the export-state capture in pages/data_analysis.py assumes when the
matching session-state key is absent. `overrides` keyed by widget LABEL flips
individual widgets (e.g. turning the GMM checkbox on). Labels must match the app
source exactly, including capitalisation ('Marginal Plot Type', not '... plot ...').
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

    Resolution order per widget, matching real Streamlit closely enough to drive the
    app's own widget functions:
      1. `overrides[label]` — the harness forcing a specific control
      2. `st.session_state[key]` — a value the caller seeded, exactly as the app's
         `_collect_*` helpers read it back (this is what makes it possible to drive
         `filters_widget()` headlessly)
      3. the widget's own default (`value=` / `options[index]` / `default=`)
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
