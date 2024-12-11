import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from navigation import render_top_menu

st.set_page_config(layout="wide", initial_sidebar_state="collapsed")
# Render the top menu 
render_top_menu()

col1, col2 = st.columns([0.4, 1])
with col1:
    st.title("Finding outliers")
    method = st.selectbox(
        "Select a outlier detection method",
        ["Image Level Boxplots", "PCA: fitted features", "PCA: raw data"]
    )  
    upload_complete = False 
    if method == "Image Level Boxplots" or method == "PCA: fitted features":
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
        
    elif method == "PCA: raw data":
        uploaded_sdt = st.file_uploader("Upload the raw sdt file", type=["sdt"])
        uploaded_mask = st.file_uploader("Upload the mask file", type=["tiff", "tif"])
        if uploaded_sdt is not None and uploaded_mask is not None:
            upload_complete = True

    if upload_complete is False:
        st.write("Please upload a file to begin.")


with col2:
    if upload_complete: 
        if method == "PCA: fitted features": 
            if "All NADH Variables" in nadh_vars:
                nadh_vars = nadh_cols
            if "All FAD Variables" in fad_vars:
                fad_vars = fad_cols
            if "All Morphology Variables" in morphology_vars:
                morphology_vars = morphology_cols
            if len(nadh_vars + fad_vars + morphology_vars) > 1:
                selected_vars = nadh_vars + fad_vars + morphology_vars
                X = df[selected_vars]
                # Standardize features before PCA
                X_std = StandardScaler().fit_transform(X)

                # Perform PCA to reduce to 2 components
                pca = PCA(n_components=2)
                principal_components = pca.fit_transform(X_std)
                df_pca = pd.DataFrame(principal_components, columns=["PC1", "PC2"])

                # Add treatment and base_name back to the df for plotting/hovering
                if "base_name" in df.columns:
                    df_pca["base_name"] = df["base_name"]
                else:
                    # If no base_name column, create a dummy one
                    df_pca["base_name"] = "Not Specified"
                if "treatment" in df.columns:
                    df_pca["treatment"] = df["treatment"]
                else:
                    # If no treatment column, create a dummy one
                    df_pca["treatment"] = "Not Specified"

                # Create a Plotly scatter plot
                # Hover data is set so that only base_name is shown on hover
                fig = px.scatter(
                    df_pca,
                    x="PC1",
                    y="PC2",
                    color="treatment",
                    hover_data={"base_name": True, "PC1": False, "PC2": False, "treatment": False},
                    title="PCA Plot"
                )

                # Update axis labels to include explained variance
                exp_var = pca.explained_variance_ratio_ * 100
                fig.update_xaxes(title_text=f"PC1 ({exp_var[0]:.2f}%)")
                fig.update_yaxes(title_text=f"PC2 ({exp_var[1]:.2f}%)")

                st.plotly_chart(fig, use_container_width=True)
            else:
                st.write("Please select at least two numeric variables for PCA.")
        elif method == "Image Level Boxplots":
            pass

        elif method == "PCA: raw data":
            pass
    else:
        st.write("Waiting for file upload")