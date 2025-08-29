import streamlit as st
import math
import pandas as pd
import numpy as np
from plotly import graph_objects as go
from src.choose_shift import choose_shift_fit_free, choose_shift_fit
from src.fit_helper import forward_pass, irf_shift, mle_likelihood, chi_square

def display_shift_data_widget(results, channel_name, choose_shift_method, time_axis=None, period=None, num_components=None, log_y=True):
    
    # combines image_name and shift from results into a df
    # kflow decay_id is the cell_name, otherwise it is the image_name
    plot_df =  pd.DataFrame({"decay_id": results["decay_id"], "shift": results["shift"]})
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
            hovertext=plot_df["decay_id"],
            customdata=plot_df["decay_id"], # Assign image_name to customdata
            hovertemplate="<b>Shift</b>: %{y}<br>%{hovertext}<extra></extra>",
        ))
        fig.update_layout(
            title=dict(
                text=f"Shifts for {channel_name} channel",
                x=0.5,  # Center the title horizontally
                xanchor='center'  # Anchor the title to its center point
            ),
            yaxis_title="Shift (bins)",
            showlegend=False, # Hide legend for single trace
            hovermode='closest', # Enable hover mode
        )
        if choose_shift_method == "fit free":
            st.plotly_chart(fig) # no event for fit free
            event = None
        else:
            # display the shift in an interactive plot scatter plot, the y-axis is the shift. When click on the point, it will show the curve with the fitted line
            # prepare the data
            try: 
                decay_curves = results["decay_curves"]
                #original_decay_curves = results["original_decay_curves"]
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
            except:
                return "Error: Results not found for channel: " + channel_name
            event = st.plotly_chart(fig, on_select="rerun", key=f"shift_image_plot_{channel_name}")

    with cols[1]:
        if event and event.selection and event.selection.points:
        # each point dict has point_index, customdata, x, y, etc.
            p = event.selection.points[0]         # first (or loop them)

            idx = p["point_index"]                # row index back into plot_df
            clicked_shift_identifier = plot_df.iloc[idx]["decay_id"]

            st.session_state["clicked_shift_plot"] = clicked_shift_identifier
            # plot the decay curve with the fitted line
            fig2 = go.Figure()
            # Calculate bin numbers for hover display
            try:
                bin_numbers = time_axis / period
            except:
                return "Error: Time axis not found for channel: " + channel_name
            
            # # plot the original decay curve
            # fig2.add_trace(go.Scatter(
            #     x=time_axis,
            #     y=original_decay_curves[idx],
            #     mode='markers',
            #     name='Original Decay Curve',
            #     line=dict(color='lightblue'),
            #     marker=dict(size=4),
            #     customdata=bin_numbers,
            #     hovertemplate="bin #: %{customdata:.0f}<br>Intensity: %{y:.0f}<extra></extra>"
            # ))

            fig2.add_trace(go.Scatter(
                x=time_axis,
                y=decay_curves[idx],
                mode='markers',
                name='Decay Curve',
                line=dict(color='orange'),
                marker=dict(size=4),
                customdata=bin_numbers,
                hovertemplate="bin #: %{customdata:.0f}<br>Intensity: %{y:.0f}<extra></extra>"
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
                title=f"Decay Curve and Fitted Line for {clicked_shift_identifier}",
                xaxis_title="Time (ns)",
                yaxis_title="Intensity (log)" if log_y else "Intensity",
                yaxis_type="log" if log_y else "linear",
                showlegend=True,
            )
            st.plotly_chart(fig2, use_container_width=True)

def choose_shift_widget(metadata_df, metadata_dict, fov_name_col, channel_name, log_y=True):
    error_msg = ""
    duration = metadata_dict["duration"]
    time_bins = metadata_dict["time_bins"]
    input_type = metadata_dict["decay_input_type"]
    if "channels_shift" in metadata_dict and channel_name in metadata_dict["channels_shift"]:
        choose_shift_method = metadata_dict["channels_shift"][channel_name]
    else:
        return "Error: Choose shift method not found for channel: " + channel_name, None
    if choose_shift_method == "fit free":
        error_msg, results = choose_shift_fit_free(metadata_df, time_bins, input_type, channel_name)
    else:
        try: 
            fitting_algo = metadata_dict["fitting_algo"]
            fitting_mode =  metadata_dict["fitting_mode"]
            num_components = metadata_dict[channel_name]["num_components"]
            start = metadata_dict[channel_name]["start"]
            end = metadata_dict[channel_name]["end"]
        except:
            return "Error: Fitting algorithm or mode or number of components not found for channel: " + channel_name, None
        error_msg, results = choose_shift_fit(metadata_df, duration, time_bins, num_components, fitting_algo, fitting_mode, input_type, channel_name, start=start, end=end)
    if error_msg != "":
        return error_msg, None
    
    period = duration / time_bins
    time_axis = np.linspace(0, (time_bins - 1) * period, time_bins, dtype=np.float64)
    display_shift_data_widget(results, channel_name, choose_shift_method, time_axis, period, metadata_dict[channel_name].get("num_components", 0), log_y)
    
    if metadata_dict["fix_shift"]:
        median_shift = np.median(results["shift"])
        print(median_shift)
        shift_data = st.number_input(f"{channel_name} Shift", value=median_shift, step=0.1, help=f"The shift for {channel_name} channel. The provided default value is the median of the shifts. You can change it to a specific value.")
    else:
        if "2D" in input_type:
            # get one shift value for each fov 
            fovs = metadata_df[fov_name_col].unique()
            decay_ids = results["decay_id"]
            shifts = results["shift"]
            fov_shifts = []
            for fov in fovs:
                shift_fov = np.median([shifts[decay_ids.index(decay_id)] for decay_id in decay_ids if decay_id.startswith(fov)])
                print(shift_fov)
                fov_shifts.append(shift_fov)
            shift_data = fov_shifts
        else:
            shift_data = results["shift"]

    return "", shift_data

def fit_options_widget(metadata_dict):
    """
    Fit options widget for Streamlit app.
    """
    # Create columns for layout
    cols_per_row = 2
    
    # First row - metric and fitting mode
    cols1 = st.columns(cols_per_row)
    with cols1[0]:
        fitting_algo = st.selectbox(
            "Metric", 
            ["MLE", "LS"], 
            index=0, 
            key="fitting_metric",
            help="MLE: Maximum Likelihood Estimation. LS: Least Squares."
        )
    with cols1[1]:
        fitting_mode = st.selectbox(
            "Fitting Mode", 
            ["Hybrid", "Global", "Local"], 
            index=0, 
            key="fitting_mode",
            help="Hybrid: use global fit to get a good initial guess, then use local fit to refine the fit. Global: use global fit to get the best fit. Local: use local fit to get the best fit."
        )
    
   
    # Handle channel-specific number of components
    channel_components = {}
    channels_fit = metadata_dict["Lifetime fit"]
    
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
   
    # Update channel-specific components
    for channel_name in channels_fit:
        metadata_dict[channel_name]["num_components"] = channel_components[channel_name]
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
