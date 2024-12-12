import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import os
from dimension_reduction import dimension_reduction
from navigation import render_top_menu

st.set_page_config(layout="wide", initial_sidebar_state="collapsed")
# Render the top menu 
render_top_menu()

col1, col2 = st.columns([0.4, 1])
with col1:
    st.title("Clustering")
    method = st.selectbox(
        "Select a clustering & outlier detection method",
        ["PCA: fitted features", "PCA: raw data", "UMAP: fitted features", "UMAP: raw data", "Image Level Boxplots"],
    )  
    upload_complete = False 
    if method == "Image Level Boxplots" or "fitted features" in method:
        uploaded_csv = st.file_uploader("Upload the CSV file from Region Props", type=["csv"])
    
        if uploaded_csv is not None:
            upload_complete = True
        # Read the uploaded data
            df = pd.read_csv(uploaded_csv)
            numeric_cols = [ col for col in df.columns if pd.to_numeric(df[col], errors='coerce').notna().all()]
            nadh_cols = [c for c in numeric_cols if (c.startswith("nadh") or c.startswith("redox")) and "mean" in c and "stdev" not in c and "weighted" not in c]
            fad_cols = [c for c in numeric_cols if c.startswith("fad") and "mean" in c and "stdev" not in c and "weighted" not in c]
            morphology_cols = [c for c in numeric_cols if not c.startswith("nadh") and not c.startswith("fad") and "mask" not in c and "redox" not in c and "flirr" not in c]
            
            if method == "Image Level Boxplots":
                # Define a callback function to reset other menus
                def reset_other_menus(selected_menu):
                    selected_value = st.session_state[selected_menu]
                    if selected_value != "Select":  # Only reset if the selection is not "Select"
                        for menu in ["menu_nadh", "menu_fad", "menu_morphology"]:
                            if menu != selected_menu:
                                st.session_state[menu] = "Select"
                        st.session_state.selected_menu = selected_menu

                # Render the dropdowns with callbacks
                selected_nadh = st.selectbox(
                    "Nadh Variables", 
                    ["Select"] + nadh_cols, 
                    index=0, 
                    key="menu_nadh",
                    on_change=reset_other_menus, 
                    args=("menu_nadh",)
                )

                selected_fad = st.selectbox(
                    "Fad Variables", 
                    ["Select"] + fad_cols, 
                    index=0, 
                    key="menu_fad",
                    on_change=reset_other_menus, 
                    args=("menu_fad",)
                )

                selected_morphology = st.selectbox(
                    "Morphology Variables", 
                    ["Select"] + morphology_cols, 
                    index=0, 
                    key="menu_morphology",
                    on_change=reset_other_menus, 
                    args=("menu_morphology",)
                )

                selected_var =  selected_nadh if selected_nadh != "Select" else selected_fad if selected_fad != "Select" else selected_morphology
            else:
                nadh_vars = st.multiselect(
                    "Select NADH Variables",
                    options= ["All NADH Variables"] + nadh_cols ,
                    default=[],
                    help="Select one or more columns corresponding to NADH variables."
                )
                fad_vars = st.multiselect(
                    "Select FAD Variables",
                    options= ["All FAD Variables"] + fad_cols,
                    default=[],
                    help="Select one or more columns corresponding to FAD variables."
                )
                morphology_vars = st.multiselect(
                    "Select Morphology Variables",
                    options= ["All Morphology Variables"] + morphology_cols ,
                    default=[],
                    help="Select one or more columns corresponding to morphology variables."
                )
        
    elif "raw data" in method:
        st.markdown("Instead of asking user to upload raw data files separately, **user can copy and paste the \
                 *path* of the folder containing the sdt files and masks in the text box below.**")
        
        st.markdown("<h5 style='text-align: center; color: red;'>Note: due to security concerns, this tool only works offline, as it will expose your local file system</h5>", unsafe_allow_html=True)
        
        folder_path = st.text_input("Enter a folder path:")

        if folder_path and st.button("List Files"):
            if os.path.isdir(folder_path):
                files = os.listdir(folder_path)
                st.write(f"Files in `{folder_path}`:")
                st.write(files)
                upload_complete = True
            else:
                st.markdown("***Warning: The provided path is not a directory or doesn't exist.***")
        # uploaded_sdt = st.file_uploader("Upload the raw sdt file", type=["sdt"])
        # uploaded_mask = st.file_uploader("Upload the mask file", type=["tiff", "tif"])
        # if uploaded_sdt is not None and uploaded_mask is not None:
        #     upload_complete = True

    if upload_complete is False:
        st.write("Please upload a file/folder path to begin.")


with col2:
    if upload_complete: 
        if "fitted features" in method: 
            if "All NADH Variables" in nadh_vars:
                nadh_vars = nadh_cols
            if "All FAD Variables" in fad_vars:
                fad_vars = fad_cols
            if "All Morphology Variables" in morphology_vars:
                morphology_vars = morphology_cols
            if len(nadh_vars + fad_vars + morphology_vars) > 1:
                selected_vars = nadh_vars + fad_vars + morphology_vars
                X = df[selected_vars]
                if "PCA" in method: 
                    df_reduced, exp_var  = dimension_reduction(X, n_components=2, method="PCA")
                    axis_labels = ["PC1", "PC2"]
                elif "UMAP" in method:
                    df_reduced, exp_var = dimension_reduction(X, n_components=2, method="UMAP")
                    axis_labels = ["UMAP1", "UMAP2"]
                else: 
                    st.write("Method not supported")

                # Add treatment and base_name back to the df for plotting/hovering
                if "base_name" in df.columns:
                    df_reduced["base_name"] = df["base_name"]
                else:
                    # If no base_name column, create a dummy one
                   df_reduced["base_name"] = "Not Specified"
                if "treatment" in df.columns:
                    df_reduced["treatment"] = df["treatment"]
                else:
                    # If no treatment column, create a dummy one
                    df_reduced["treatment"] = "Not Specified"
 
                st.markdown("<h5 style='text-align: center; color: black;'>Hover over to find the base_name of the outliers</h5>", unsafe_allow_html=True)
            
                unique_treatments = df_reduced["treatment"].unique()
                palette = sns.color_palette("tab20", n_colors=len(unique_treatments))
                color_sequence = [f"rgba({int(color[0]*255)}, {int(color[1]*255)}, {int(color[2]*255)}, 0.6)" for color in palette]
                color_map = {t: color_sequence[i] for i, t in enumerate(unique_treatments)}

                # Create scatter plot
                fig = go.Figure()

                for t in unique_treatments:
                    t_df =  df_reduced[df_reduced["treatment"] == t]
                    fig.add_trace(
                        go.Scatter(
                            x=t_df[axis_labels[0]],
                            y=t_df[axis_labels[1]],
                            mode='markers',
                            name=f'{t}',
                            text=t_df["base_name"],
                            hovertemplate="<b>%{text}</b>",
                            marker=dict(color=color_map[t])
                        ),
                )

                # Update axis labels to include explained variance
                if exp_var is not None: 
                    fig.update_xaxes(title_text=f"{axis_labels[0]}({exp_var[0]:.2f}%)")
                    fig.update_yaxes(title_text=f"{axis_labels[1]}({exp_var[1]:.2f}%)")
                else:
                    fig.update_xaxes(title_text=f"{axis_labels[0]}")
                    fig.update_yaxes(title_text=f"{axis_labels[1]}")

                st.plotly_chart(fig, use_container_width=True)
            else:
                st.markdown("<h5 style='text-align: center; color: black;'>Please select at least two numeric variables for performing dimension reduction.</h5>", unsafe_allow_html=True)
        elif method == "Image Level Boxplots":
            if selected_var != "Select":
               #  st.markdown(f"<h5 style='text-align: center; color: black;'>You selected '{selected_var}'.</h5>", unsafe_allow_html=True)

               
                for index, row in df.iterrows():
                    basename = row['base_name']
                    try: 
                        # we assume that the image name is the base_name without the cell number (which is found after the last underscore)
                        df.at[index, 'image_name'] = basename.rsplit('_', 1)[0]
                    except: 
                        st.markdown("<h5 style='text-align: center; color: Red;'>Warning: We cannot infer image name from you base_name. We assume that the image name is the base_name without the cell number (which is found after the last underscore) </h5>", unsafe_allow_html=True)
                # Create a boxplot for the selected variable
                fig = px.box(df, x="image_name", y=selected_var, title=f"Boxplot for {selected_var}")
                st.plotly_chart(fig, use_container_width=True)
  
            else:
                st.markdown("<h5 style='text-align: center; color: black;'>Please select one variable to plot.</h5>", unsafe_allow_html=True)


        elif "raw data" in method:
            pass
    else:
        st.write("Waiting for file/folder path upload")