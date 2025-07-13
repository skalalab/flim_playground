import streamlit as st
import math
import pandas as pd
import numpy as np
from plotly import graph_objects as go
from src.choose_shift import choose_shift
from src.fit_helper import forward_pass, irf_shift, mle_likelihood, chi_square


def display_shift_data_widget(results, analysis_type, metadata_df, time_axis, period, num_components, log_y, channel):
    # display the shift in an interactive plot scatter plot, the y-axis is the shift. When click on the point, it will show the curve with the fitted line
    # prepare the data
    decay_curves = results["decay_curves"]
    original_decay_curves = results["original_decay_curves"]
    shift_data = results["shift"]
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
    if analysis_type == "K-Flow":
        image_name = metadata_df['kflow_exp_name'].iloc[0]
        image_names = [image_name] * len(shift_data)
        plot_df =  pd.DataFrame({'image_name': image_names, 'shift': shift_data})
    else:
        plot_df =  pd.DataFrame({'image_name': metadata_df['image_name'], 'shift': shift_data})
    cols = st.columns(2)
    with cols[0]:
        fig = go.Figure()
        fig.add_trace(go.Box(
            y=plot_df['shift'],
            name="shift",
            boxpoints='all',
            jitter=0.3,
            pointpos=0,
            marker=dict(
                color='black',
                size=10,
                opacity=1,
                line=dict(width=0.5, color='DarkSlateGrey')
            ),
            fillcolor='rgba(0,0,0,0)',
            line_color='rgba(0,0,0,0)',
            hovertext=plot_df['image_name'],
            customdata=plot_df['image_name'], # Assign image_name to customdata
            hovertemplate="<b>Shift</b>: %{y}<br>Image: %{hovertext}<extra></extra>",
        ))
        fig.update_layout(
            title=dict(
                text=f"Shifts for {channel} channel",
                x=0.5,  # Center the title horizontally
                xanchor='center'  # Anchor the title to its center point
            ),
            yaxis_title="Shift (bins)",
            showlegend=False, # Hide legend for single trace
            hovermode='closest', # Enable hover mode
        )
        event = st.plotly_chart(fig, on_select="rerun", key=f"shift_image_plot_{channel}")
        
    with cols[1]:
        if event and event.selection and event.selection.points:
        # each point dict has point_index, customdata, x, y, etc.
            p = event.selection.points[0]         # first (or loop them)

            idx = p["point_index"]                # row index back into plot_df
            img_name = plot_df.iloc[idx]["image_name"]

            st.session_state["clicked_image_shift_plot"] = img_name
            # plot the decay curve with the fitted line
            fig2 = go.Figure()
            # Calculate bin numbers for hover display
            bin_numbers = time_axis / period
            fig2.add_trace(go.Scatter(
                x=time_axis,
                y=original_decay_curves[idx],
                mode='markers',
                name='Original Decay Curve',
                line=dict(color='lightblue'),
                marker=dict(size=4),
                customdata=bin_numbers,
                hovertemplate="bin #: %{customdata:.0f}<br>Intensity: %{y:.0f}<extra></extra>"
            ))

            # plot the original decay curve
            fig2.add_trace(go.Scatter(
                x=time_axis,
                y=decay_curves[idx],
                mode='markers',
                name='Decay Curve',
                line=dict(color='orange'),
                marker=dict(size=4),
                customdata=bin_numbers,
                hoverinfo='skip'
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

            fig2.add_trace(go.Scatter(
                x=time_axis,
                y=fitted_curve,
                mode='lines',
                name='Fitted Curve',
                line=dict(color='red'),
                hoverinfo='skip'
            ))
        
            # Add annotations with fitting parameters and statistics
            annotation_text = f"<b>Shift: {shift_data:.2f}</b><br>"
            annotation_text += f"<b>t1: {t1_data * 1000:.2f} ns</b><br>"
            if num_components > 1:
                a1 = amp1_data / (amp1_data + amp2_data)
                annotation_text += f"<b>α1: {a1 * 100:.2f}%</b><br>"
                annotation_text += f"<b>t2: {t2_data * 1000:.2f} ns</b><br>"
            if num_components > 2:
                a2 = amp2_data / (amp1_data + amp2_data + amp3_data)
                annotation_text += f"<b>α2: {a2 * 100:.2f}%</b><br>"
                annotation_text += f"<b>t3: {t3_data * 1000:.2f} ns</b><br>"
            annotation_text += f"<b>MLE: {mle:.2f}</b><br>"
            annotation_text += f"<b>χ²: {chiq:.2f}</b>"

            fig2.add_annotation(
                    text=annotation_text,
                    xref="paper", yref="paper",
                    x=0.5 if log_y else 0.98, y=0.02 if log_y else 0.98,
                    xanchor="center" if log_y else "right", yanchor="bottom" if log_y else "top",
                    showarrow=False,
                    bgcolor="rgba(255, 255, 255, 0.8)",
                    bordercolor="black",
                    borderwidth=1,
                    font=dict(size=12)
            )

            fig2.update_layout(
                title=f"Decay Curve and Fitted Line for {img_name}",
                xaxis_title="Time (ns)",
                yaxis_title="Intensity (log)",
                yaxis_type="log" if log_y else "linear",
                showlegend=True,
            )
            st.plotly_chart(fig2, use_container_width=True)
def choose_shift_widget(metadata_df, metadata_dict, channel_name, log_y=True):
    duration = metadata_dict["duration"]
    time_bins = metadata_dict["time_bins"]
    input_type = metadata_dict["input_type"]
    if "choose_shift" in metadata_dict and channel_name in metadata_dict["choose_shift"]:
        choose_shift_method = metadata_dict["choose_shift"][channel_name]
    else:
        return "Error: Choose shift method not found for channel: " + channel_name, None
    error_msg, results = choose_shift(metadata_df, metadata_dict, channel_name)
    if error_msg != "":
        return error_msg, None
    
    display_shift_data_widget(results, metadata_dict, channel_name, log_y)

    if choose_shift_method == "fit free" or "fix_shift" not in metadata_dict or metadata_dict["fix_shift"]:
        fix_shift = True
    else:
        fix_shift = False
    
    if fix_shift:
        median_shift = np.median(results["shift"])
        shift_data = st.number_input(f"{channel_name} Shift", value=median_shift, step=0.1, help=f"The shift for {channel_name} channel. The provided default value is the median of the shifts. You can change it to a specific value.")
    
    return error_msg, shift_data

def fit_options_widget(input_type, metadata_dict):
    """
    Fit options widget for Streamlit app.
    """
    # Create columns for layout
    cols_per_row = 3
    
    # First row - algorithm and fitting mode
    cols1 = st.columns(cols_per_row)
    with cols1[0]:
        fitting_algo = st.selectbox(
            "Algorithm", 
            ["MLE", "WLS"], 
            index=0, 
            key="fitting_algo",
            help="MLE: Maximum Likelihood Estimation. WLS: Weighted Least Squares."
        )
    with cols1[1]:
        fitting_mode = st.selectbox(
            "Fitting Mode", 
            ["Hybrid", "Global", "Local"], 
            index=0, 
            key="fitting_mode",
            help="Hybrid: use global fit to get a good initial guess, then use local fit to refine the fit. Global: use global fit to get the best fit. Local: use local fit to get the best fit."
        )
    
    # Add fix_shift for ROI Summing Fit
    fix_shift = None
    if input_type == "ROI Summing Fit":
        with cols1[2]:
            fix_shift = st.checkbox(
                "Fix the Shift", 
                value=True, 
                key="fix_shift",
                help="If True, the shift will be fixed for all images. If False, the shift will be estimated for each image."
            )
    
    # Handle channel-specific number of components
    channel_components = {}
    channels_fit = metadata_dict["channels_fit"]
    
    # Create additional rows for channel components
    if channels_fit:
        st.write("**Number of components:**")
        num_channels = len(channels_fit)
        num_rows = math.ceil(num_channels / cols_per_row)
        
        for row in range(num_rows):
            cols = st.columns(cols_per_row)
            for col_idx in range(cols_per_row):
                channel_idx = row * cols_per_row + col_idx
                if channel_idx >= num_channels:
                    break
                
                channel_name = channels_fit[channel_idx]
                with cols[col_idx]:
                    channel_components[channel_name] = st.number_input(
                        f"{channel_name}", 
                        value=metadata_dict[channel_name]["num_components"], 
                        step=1, 
                        min_value=1, 
                        max_value=3,
                        key=f"{channel_name}_num_components"
                    )
    
    # Update metadata_dict with results
    metadata_dict["fitting_algo"] = fitting_algo
    metadata_dict["fitting_mode"] = fitting_mode
    if input_type == "ROI Summing Fit":
        metadata_dict["fix_shift"] = fix_shift
    else: 
        # for other input types, fix the shift always
        metadata_dict["fix_shift"] = True
    
    # Update channel-specific components
    for channel_name in channels_fit:
        metadata_dict[channel_name]["num_components"] = channel_components[channel_name]
    
    # Add start and end widgets for each channel
    for channel_name in channels_fit:
        start, end = start_end_widget(metadata_dict["time_bins"], channel_name)
        metadata_dict[channel_name]["start"] = start
        metadata_dict[channel_name]["end"] = end
    
    return metadata_dict

def start_end_widget(time_bins, channel):
    col1, col2 = st.columns(2)
    with col1:
        start = st.number_input(
            f"{channel} Start (T1)", 
            value=0, 
            step=1, 
            min_value=0, 
            max_value=time_bins-1,
            key=f"{channel}_start"
        )
    with col2:
        end = st.number_input(
            f"{channel} End (T2)", 
            value=time_bins, 
            step=1, 
            min_value=1, 
            max_value=time_bins,
            key=f"{channel}_end"
        )

    return start, end
