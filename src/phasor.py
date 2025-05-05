import plotly.graph_objects as go
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from roi_sum import sum_sdts
import seaborn as sns

def visualize_irf(irf, shifted_irf, image_timebin):

    fig, ax = plt.subplots(ncols=2, figsize=(6, 2))
    ax[0].plot(irf / max(irf), label = "irf")
    ax[0].plot(image_timebin / max(image_timebin), label = "sdt curve")
    ax[0].legend()
    ax[0].set_title("Before Shift") 
    ax[1].plot(shifted_irf/ max(shifted_irf), label = "shifted_irf")
    ax[1].plot(image_timebin / max(image_timebin), label = "sdt curve")
    ax[1].legend()
    ax[1].set_title("After Shift") 

    return fig

def zero_padding_shift(irf, shift):
    # zero padding shift
    shifted = np.zeros_like(irf)
    if shift > 0:
        shifted[shift:] = irf[:-shift]
    elif shift < 0:
        shifted[:shift] = irf[-shift:]
    else:
        shifted = irf
    return shifted

def irf_shift(images, selected_channel="NADH", scale_factor = 10):
    irf = np.array(images["original_irf"])
    num_bins = len(irf)
    timeBin_name = f"{selected_channel.lower()}_timebins"
    for image, properties in images.items():
        if image == "original_irf":
            continue
        image_timeBins = np.sum(properties[timeBin_name], axis=0)
        if len(image_timeBins) != num_bins:
            error_message = f"IRF and {image} have different timebins. IRF has {len(irf)} timebins, while {image} has {len(image_timeBins)} timebins."
            return None, error_message
        # shift the irf
        # scale the irf up 
        num_sclaed_bins = (num_bins - 1) * scale_factor + 1
        # interpolate the irf and image timebins
        interp_irf = np.interp(np.arange(num_sclaed_bins), np.arange(0, num_sclaed_bins, scale_factor), irf)
        interp_image_timeBins = np.interp(np.arange(num_sclaed_bins), np.arange(0, num_sclaed_bins, scale_factor), image_timeBins)
        # cross correlate the two
        corr_result = signal.correlate(interp_irf, interp_image_timeBins)
        # Find the peak of correlation with 
        peak_idx = np.argmax(corr_result)
        # Compute the shift
        shift = (num_sclaed_bins - 1) - peak_idx
        # shift the irf 
        shifted_irf = zero_padding_shift(interp_irf, shift)
        # downsample the shifted irf
        shifted_irf = shifted_irf[::scale_factor]
        # fig = visualize_irf(irf, shifted_irf, image_timeBins)

        properties["shifted_irf"] = shifted_irf
    
    return images, ""

def get_gs_coords(timebin, irf, f=0.08, duration=10):

    """
    Calculate the g and s coordinates for a given timebin
    f = 0.08 # laser repetition rate in [GHz]
    duration = 10 # duration of the timebin in [ns]
    """
    w = 2*np.pi*f
    time_axis = np.arange(0, duration, duration/len(timebin))
    G_IRF = np.dot(np.transpose(irf) , np.cos(w*time_axis)) / np.sum(irf)
    S_IRF = np.dot(np.transpose(irf) , np.sin(w*time_axis)) / np.sum(irf)
    cos_coeff = np.cos(w*time_axis)
    sin_coeff = np.sin(w*time_axis)
    
    # corrected coefficients
    corrected_cos_coeff = (G_IRF/(G_IRF**2 + S_IRF**2))*cos_coeff + (S_IRF/(G_IRF**2 + S_IRF**2))*sin_coeff
    corrected_sin_coeff = (-S_IRF/(G_IRF**2 + S_IRF**2))*cos_coeff + (G_IRF/(G_IRF**2 + S_IRF**2))*sin_coeff

    timebin_sum = np.sum(timebin)
    G = np.dot(timebin, corrected_cos_coeff) / timebin_sum
    S = np.dot(timebin, corrected_sin_coeff) / timebin_sum
    return G,S

def calculate_phasor(images, selected_channel="NADH"):
    """
    Perform phasor analysis on the sdt files
    """
    images, error_message = sum_sdts(images,selected_channel=selected_channel, write_tiff=False, write_sdt=False)
    if images is None:
        return None, error_message

    # shift the irf 
    # potenial error: the irf dimension is different from the sdt timebins
    images, error_message = irf_shift(images, selected_channel=selected_channel)
    if images is None:
        return None, error_message
    
    timeBin_name = f"{selected_channel.lower()}_timebins"
    sdt_Path = f"{selected_channel.lower()}_sdt"
    cell_labels = []
    timebins_imageName = []
    categories = []
    G_coords = []
    S_coords = []
    # calculate the phasor
    for image, properties in images.items():
        if image == "original_irf":
            continue
        irf = properties["shifted_irf"]
        timebins = properties[timeBin_name]
        timebins_imageName.append(np.array([image]*len(timebins)))
        cell_labels.append(properties["cells"])
         # use the parent folder name as the category
        image_parent = Path(properties[sdt_Path]).parent.name
        categories.append(np.array([image_parent]*len(timebins)))
        for timebin in timebins:
            G, S = get_gs_coords(timebin, irf)
            G_coords.append(G)
            S_coords.append(S)

        
    cell_labels = np.hstack(cell_labels)
    timebins_imageName = np.hstack(timebins_imageName)
    categories = np.hstack(categories)

    if len(G_coords) != 0:
        df = pd.DataFrame({"G": G_coords, "S": S_coords})
        # augment the dimensional reduction df with metadata
        df["image_name"] = timebins_imageName
        df["cell_labels"] = cell_labels
        df["base_name"] = df["image_name"] + "_" + df["cell_labels"].astype(str)
        df["color_category"] = categories
    
    else: 
        df = None
        return None, error_message
    return df, error_message


def phasor_plot(df, f=0.08):

    # Create the figure
    fig = go.Figure()

    # Set axis limits
    fig.update_layout(
        xaxis=dict(range=[-0.05, 1.05]),
        yaxis=dict(range=[-0.05, 0.55]), 
    )

    # Plot the curve
    u = np.arange(0, 100, 0.01)
    x_curve = 1 / (1 + u**2)
    y_curve = u / (1 + u**2)

    fig.add_trace(go.Scatter(
        x=x_curve,
        y=y_curve,
        mode='lines',
        line=dict(color='black'),
        name='Curve', 
        hoverinfo='skip',# Hide the hover info for this trace
        showlegend=False 
    ))

    # Calculate and plot specific points
    wt = 2 * np.pi * f * np.array([0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
    x_points = 1 / (1 + wt**2)
    y_points = wt / (1 + wt**2)

    fig.add_trace(go.Scatter(
        x=x_points,
        y=y_points,
        mode='markers',
        marker=dict(size=8, color='black'),
        name='Lifetime Markers', 
        hoverinfo='skip', # Hide the hover info for this trace,
        showlegend=False
    ))

    # Annotate the points
    lifetime_labels = ['0.5 ns', '1 ns', '2 ns', '3 ns', '4 ns', '5 ns']
    labels = len(lifetime_labels)
    label_coords = list(zip(x_points - 0.02, y_points + 0.03))[:labels]

    for i in range(labels):
        fig.add_annotation(
            x=label_coords[i][0],
            y=label_coords[i][1],
            text=lifetime_labels[i],
            showarrow=False,
            font=dict(size=10),
            xanchor='left'
        )
    
    # Add titles and axis labels
    title = 'Phasor Plot'
    fig.update_layout(
        title=title,
        xaxis_title='g',
        yaxis_title='s',
        font=dict(size=15),
        title_font=dict(size=20, family='Arial', color='black'),
        xaxis=dict(title_font=dict(size=15, family='Arial', color='black')),
        yaxis=dict(title_font=dict(size=15, family='Arial', color='black'))
    )

    # Add text inside the plot
    fig.add_annotation(
        x=0.8,
        y=0.5,
        text=f"{f * 1000} MHz",
        showarrow=False,
        font=dict(size=15, color='black'),
        xanchor='left'
    )

    # Maintain 1:2 aspect ratio with dynamic sizing
    fig.update_layout(
        autosize=True,  # Let Plotly automatically resize based on the screen
        xaxis=dict(scaleanchor="y"),  # Maintain aspect ratio (1:2)
        margin=dict(l=10, r=10, t=50, b=10),  # Adjust margins as needed
        hovermode='closest'
    )
    
    # plot the phasor coordinates
    
    unique_color_groups = df["color_category"].unique()
    alpha = 0.6 if len(unique_color_groups) > 1 else 1.0
    palette = sns.color_palette("tab10", n_colors=len(unique_color_groups))
    color_sequence = [f"rgba({int(color[0]*255)}, {int(color[1]*255)}, {int(color[2]*255)}, {alpha})" for color in palette]
    color_map = {t: color_sequence[i] for i, t in enumerate(unique_color_groups)}

    for g in unique_color_groups:
        g_df =  df[df["color_category"] == g]
        fig.add_trace(
            go.Scatter(
                x=g_df['G'],
                y=g_df['S'],
                mode='markers',
                name=f'{g}',
                text=g_df["base_name"],
                customdata=g_df["image_name"],
                hovertemplate="<b>%{text}</b>",
                marker=dict(color=color_map[g],size=3)
            ),
        )
    return fig
