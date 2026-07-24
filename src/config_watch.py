"""Cross-tab config staleness notice.

Streamlit has no server->other-session push: clicking "Update Configuration" in
one tab runs only *that* tab's script and cannot make other open tabs re-read the
config. Rather than silently reloading an already-open Data Extraction tab (which
could disrupt in-progress work or trigger redundant re-saves), this shows a small
banner telling the user their view is stale, so they can refresh when ready.

Mechanism: on every full page run we record the config-file mtime the page was
built from; a lightweight ``@st.fragment(run_every=...)`` re-checks that mtime and,
when another tab's save has bumped it, renders a "refresh to apply" warning. There
is no auto-rerun — the user stays in control of when the new config takes effect.
Any later full run (a browser refresh, Streamlit's "R", or any widget interaction)
re-reads config and resets the baseline, which clears the banner.
"""
import streamlit as st

from src.config import get_config_mtime

_SEEN_KEY = "_config_mtime_seen"
_STALE_MESSAGE = (
    "⚙️ The configuration was updated in another tab. **Reload this page** "
    "(i.e. hard browser refresh) to apply the new settings here."
)


def _render_stale_banner_if_needed() -> None:
    """Render the stale-config warning iff config.toml changed since this page ran.

    Reads the baseline recorded by :func:`notify_on_config_change`; renders nothing
    when the baseline is absent (never initialised) or still matches disk.
    """
    seen = st.session_state.get(_SEEN_KEY)
    if seen is not None and get_config_mtime() != seen:
        st.warning(_STALE_MESSAGE)


def notify_on_config_change(poll_interval: str = "2s") -> None:
    """Warn (do NOT auto-reload) when config.toml changes in another tab.

    Call once where the notice should appear. The surrounding page reads live
    config on this run, so we record the mtime it reflects; a fragment then polls
    every ``poll_interval`` and renders the banner if the on-disk mtime has moved
    ahead. Recording the baseline on *every* full run is deliberate: any refresh or
    interaction re-reads config and clears the banner, and the fragment ticks (which
    do not re-run this outer body) keep the banner up while the tab sits idle.
    """
    st.session_state[_SEEN_KEY] = get_config_mtime()

    @st.fragment(run_every=poll_interval)
    def _poll_config_mtime() -> None:
        _render_stale_banner_if_needed()

    _poll_config_mtime()
