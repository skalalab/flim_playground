import streamlit as st
import math



def fit_options(analysis_type):
    """
    Fit options widget for Streamlit app.
    """
    # 1) Define each field as a (name, factory) tuple
    fields = [
        ("duration",   lambda: st.number_input("Pulse Duration (ns)", value=12.5, step=0.1, format="%.1f")),
        ("num_components", lambda: st.number_input("Component No.", value=2, step=1, min_value=1, max_value=3)),
        ("fitting_algo",   lambda: st.selectbox("Algorithm", ["MLE", "WLS"], index=0)),
        ("time_bins",      lambda: st.number_input("Time Bins", value=256, step=256, min_value=256, max_value=512)),
    ]

    # add fix_shift for ROI Summing Fit
    if analysis_type == "ROI Summing Fit":
        fields.append(("fix_shift", lambda: st.checkbox("Fix the Shift", value=True)))

    # 2) figure out how many rows of up to 4 cols
    cols_per_row = 3
    num_rows = math.ceil(len(fields) / cols_per_row)

    # 3) render them
    results = {}
    for row in range(num_rows):
        cols = st.columns(cols_per_row)
        for col_idx in range(cols_per_row):
            idx = row * cols_per_row + col_idx
            if idx >= len(fields):
                break
            name, factory = fields[idx]
            with cols[col_idx]:
                results[name] = factory()

    # 4) unpack
    duration      = results["duration"]
    num_components = results["num_components"]
    fitting_algo   = results["fitting_algo"]
    time_bins      = results["time_bins"]
    fix_shift      = results.get("fix_shift", True)

    return duration, time_bins, num_components, fitting_algo, fix_shift
