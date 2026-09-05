"""Warn when another tab changes the configuration without interrupting this tab.

Each full page run records the config mtime. A polling fragment warns when the
file changes; the next full run reads current settings and resets the baseline.
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
    """Record this page's config mtime and poll for changes without auto-reloading.

    Call where the notice should appear. Fragment ticks keep the baseline while
    the tab is idle; each full page run updates it and clears any stale notice.
    """
    st.session_state[_SEEN_KEY] = get_config_mtime()

    @st.fragment(run_every=poll_interval)
    def _poll_config_mtime() -> None:
        _render_stale_banner_if_needed()

    _poll_config_mtime()
