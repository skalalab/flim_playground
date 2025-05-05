import streamlit as st
import math
import pandas as pd
import numpy as np
from plotly import graph_objects as go
from src.fit import choose_shift
from src.fit_helper import forward_pass, irf_shift, mle_likelihood, chi_square


def choose_shift_widget(metadata_df, duration, time_bins, num_components, fitting_algo, analysis_type, channel):
    period = duration / time_bins
    time_axis = np.linspace(0, (time_bins - 1) * period, time_bins, dtype=np.float64)
    error_msg, results = choose_shift(metadata_df, duration, time_bins, num_components, fitting_algo, analysis_type, channel)
    if error_msg != "":
        # Display error and stop
        st.error(error_msg)
        return error_msg # Return only the error message

    # display the shift in an interactive plot scatter plot, the y-axis is the shift. When click on the point, it will show the curve with the fitted line
    # prepare the data
    decay_curves = results["decay_curves"]
    shift_data = results["shift"]
    fitted_images = results["fitted_images"]
    amp1 = results["amp1"]
    t1 = results["t1"]
    offset = results["offset"]
    if num_components > 1:
        amp2 = results["amp2"]
        t2 = results["t2"]
    if num_components > 2:
        amp3 = results["amp3"]
        t3 = results["t3"]
    irf = results["irf"]
    # combines image_name and shift from results into a df
    plot_df =  pd.DataFrame({'image_name': metadata_df['image_name'], 'shift': shift_data})
    cols = st.columns(2)
    with cols[0]:
        fig = go.Figure()
        fig.add_trace(go.Box(
            y=plot_df['shift'],
            name="Shift",
            boxpoints='all',
            jitter=0.3,
            pointpos=0,
            marker=dict(
                color='lightgrey',
                size=6,
                opacity=1,
                line=dict(width=0.5, color='DarkSlateGrey')
            ),
            fillcolor='rgba(0,0,0,0)',
            line_color='rgba(0,0,0,0)',
            hovertext=plot_df['image_name'],
            customdata=plot_df['image_name'], # Assign image_name to customdata
        ))
        fig.update_layout(
            title=f"Shifts for {channel} channel",
            yaxis_title="Shift (bins)",
            xaxis_title="Image", # Add x-axis title
            showlegend=False, # Hide legend for single trace
            hovermode='closest', # Enable hover mode
        )
        event = st.plotly_chart(fig, on_select="rerun", key="shift_image_plot")
        
    with cols[1]:
        if event and event.selection and event.selection.points:
        # each point dict has point_index, customdata, x, y, etc.
            p = event.selection.points[0]         # first (or loop them)

            idx = p["point_index"]                # row index back into plot_df
            img_name = plot_df.iloc[idx]["image_name"]

            st.session_state["clicked_image_shift_plot"] = img_name
            # plot the decay curve with the fitted line
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=time_axis,
                y=decay_curves[idx],
                mode='markers',
                name='Decay Curve',
                line=dict(color='lightblue'),
                marker=dict(size=4),
                hovertemplate=f"Decay Curve: {decay_curves[idx]}<extra></extra>"
            ))

            amp1_data, t1_data, offset_data = amp1[idx], t1[idx], offset[idx]
            if num_components > 1:
                amp2_data, t2_data = amp2[idx], t2[idx]
            else:
                amp2_data, t2_data = None, None
            if num_components > 2:
                amp3_data, t3_data = amp3[idx], t3[idx]
            else:
                amp3_data, t3_data = None, None
            shift_data = shift_data[idx]
            st.write(f"Shift for {img_name}: {shift_data}")
            st.write(f"Offset for {img_name}: {offset_data}")
            #st.write(f"Amplitude 1 for {img_name}: {amp1_data}")
            st.write(f"t1 for {img_name}: {t1_data * 1000:.2f} ns")
            if num_components > 1:
                #st.write(f"Amplitude 2 for {img_name}: {amp2_data}")
                a1 = amp1_data / (amp1_data + amp2_data) 
                st.write(f"alpha 1 for {img_name}: {a1 * 100:.2f}%")
                st.write(f"t2 for {img_name}: {t2_data * 1000:.2f} ns")

            # shift the irf
            shifted_irf = irf_shift(irf, shift_data)
            fitted_curve = forward_pass(
                amp1=amp1_data,
                t1=t1_data,
                offset=offset_data,
                shifted_irf=shifted_irf,
                time_axis=time_axis,
                amp2=amp2_data,
                t2=t2_data,
                amp3=amp3_data,
                t3=t3_data
            )
            mle = mle_likelihood(fitted_curve, decay_curves[idx], start=0, end=-1)
            chiq = chi_square(fitted_curve, decay_curves[idx], start=0, end=-1)
            st.write(f"MLE for {img_name}: {mle:.2f}")
            st.write(f"Chi-square for {img_name}: {chiq:.2f}")
            fig2.add_trace(go.Scatter(
                x=time_axis,
                y=fitted_curve,
                mode='lines',
                name='Fitted Curve',
                line=dict(color='red'),
                hovertemplate="Fitted Curve: %{y}<extra></extra>"
            ))
        
            fig2.update_layout(
                title=f"Decay Curve and Fitted Line for {img_name}",
                xaxis_title="Time (ns)",
                yaxis_title="Intensity",
                showlegend=True,
            )
            st.plotly_chart(fig2, use_container_width=True)

   
    return error_msg

def fit_options(analysis_type):
    """
    Fit options widget for Streamlit app.
    """
    # 1) Define each field as a (name, factory) tuple
    fields = [
        ("duration",   lambda: st.number_input("Pulse Interval (ns)", value=12.5, step=0.1, format="%.1f")),
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
