"""Convert laser-rate inputs between displayed MHz and stored GHz."""

import streamlit as st

from src.widgets.analysis_widget_state import number_input_default


def laser_rate_input(label, value_ghz, *, key, max_value_mhz=1000.0):
    """Display MHz with two decimals and return GHz for existing calculations."""
    mhz_key = f"{key}_mhz"
    # A separate key prevents an open session's GHz value being read as MHz.
    # Carry its pending value forward only while initializing the new control.
    if mhz_key not in st.session_state:
        value_ghz = st.session_state.pop(key, value_ghz)
    rate_mhz = st.number_input(
        label,
        value=number_input_default(st.session_state, mhz_key, value_ghz * 1000),
        min_value=0.0, max_value=max_value_mhz,
        step=0.01, format="%.2f", key=mhz_key,
    )
    return rate_mhz / 1000
