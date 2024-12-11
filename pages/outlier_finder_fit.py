import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from navigation import render_top_menu

st.set_page_config(layout="wide", initial_sidebar_state="collapsed")

hide_sidebar_style = """
<style>
.css-1cpxqw2 {visibility: hidden;}
</style>
"""
st.markdown(hide_sidebar_style, unsafe_allow_html=True)

# Render the top menu on this page as well
render_top_menu()

st.title("Page 1")
st.write("This is Page 1. Use the top menu to go to other pages.")

# with col1:
#     st.title("Finding outliers using PCA")

#     # Upload file
#     uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    
#     if uploaded_file is not None:
#         # Read the uploaded data
#         df = pd.read_csv(uploaded_file)
        
#         # Extract column names (excluding 'base_name' if you want or keep it)
#         numeric_cols = [c for c in df.columns if c != "base_name"]

#         # Three dropdown menus for selecting variables
#         nadh_vars = st.multiselect(
#             "Select NADH Variables",
#             options=numeric_cols,
#             default=[],
#             help="Select one or more columns corresponding to NADH variables."
#         )

#         fad_vars = st.multiselect(
#             "Select FAD Variables",
#             options=numeric_cols,
#             default=[],
#             help="Select one or more columns corresponding to FAD variables."
#         )

#         morphology_vars = st.multiselect(
#             "Select Morphology Variables",
#             options=numeric_cols,
#             default=[],
#             help="Select one or more columns corresponding to morphology variables."
#         )
#     else:
#         # If no file is uploaded, just display a message
#         st.write("Please upload a file to begin.")

# with col2:
#     if uploaded_file is not None and len(nadh_vars + fad_vars + morphology_vars) > 0:
#         # Combine selected columns
#         selected_vars = nadh_vars + fad_vars + morphology_vars
        
#         # Ensure the selected vars are actually numeric
#         # (If not numeric, you might want to convert or filter them out)
#         df_selected = df.dropna(subset=selected_vars)  # remove rows with NaN in selected vars
#         X = df_selected[selected_vars].select_dtypes(include=[float, int])

#         if len(X.columns) > 1:
#             # Standardize features before PCA
#             X_std = StandardScaler().fit_transform(X)

#             # Perform PCA to reduce to 2 components
#             pca = PCA(n_components=2)
#             principal_components = pca.fit_transform(X_std)
#             df_pca = pd.DataFrame(principal_components, columns=["PC1", "PC2"])

#             # Add treatment and base_name back to the df for plotting/hovering
#             # (Only if they exist)
#             if "base_name" in df_selected.columns:
#                 df_pca["base_name"] = df_selected["base_name"]
#             if "treatment" in df_selected.columns:
#                 df_pca["treatment"] = df_selected["treatment"]
#             else:
#                 # If no treatment column, create a dummy one
#                 df_pca["treatment"] = "Not Specified"

#             # Create a Plotly scatter plot
#             # Hover data is set so that only base_name is shown on hover
#             fig = px.scatter(
#                 df_pca,
#                 x="PC1",
#                 y="PC2",
#                 color="treatment",
#                 hover_data={"base_name": True, "PC1": False, "PC2": False, "treatment": False},
#                 title="PCA Plot"
#             )

#             # Update axis labels to include explained variance
#             exp_var = pca.explained_variance_ratio_ * 100
#             fig.update_xaxes(title_text=f"PC1 ({exp_var[0]:.2f}%)")
#             fig.update_yaxes(title_text=f"PC2 ({exp_var[1]:.2f}%)")

#             st.plotly_chart(fig, use_container_width=True)
#         else:
#             st.write("Please select at least two numeric variables for PCA.")
#     else:
#         st.write("Waiting for file upload and variable selection...")