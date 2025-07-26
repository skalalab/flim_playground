import streamlit as st
import toml
from pathlib import Path
from src.config import load_config, save_config, get_unique_cell_id_col, get_fov_name_col, get_all_feature_extractors, get_categorical_cols
# Absolute path to the analysis config file (../../analysis_config.toml)
_ANALYSIS_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "analysis_config.toml"

def dataset_config_widget(use_data_extraction=True):
    # read from the data_extraction configuration and modify based on the use_data_extraction flag
    unique_row_id_col = get_unique_cell_id_col()
    fov_name_col = get_fov_name_col()
    categorical_cols = get_categorical_cols()
    categorical_cols.extend([fov_name_col])
    all_numeric_col_groups = get_all_feature_extractors()
    cfg = load_config(_ANALYSIS_CONFIG_PATH)
    if "unique_row_id_col" not in cfg:
        cfg["unique_row_id_col"] = unique_row_id_col
    if "fov_name_col" not in cfg:
        cfg["fov_name_col"] = fov_name_col
    if "categorical_cols" not in cfg:
        cfg["categorical_cols"] = categorical_cols
    if "all_numeric_col_groups" not in cfg:
        cfg["all_numeric_col_groups"] = all_numeric_col_groups
    if use_data_extraction:
        # do nothing 
        save_config(cfg, _ANALYSIS_CONFIG_PATH)
        return
    
    cols = st.columns(2)
    with cols[0]:
        cfg["unique_row_id_col"] = st.text_input("Unique Row ID", value= cfg["unique_row_id_col"], help="The column name that uniquely identifies each row in the dataset.")
    with cols[1]:
        cfg["fov_name_col"] = st.text_input("FOV Name (if available)", value= cfg["fov_name_col"])

    # Initialize session state for selected categorical columns if not exists
    if "selected_categorical_cols" not in st.session_state:
        st.session_state.selected_categorical_cols = cfg.get("categorical_cols", categorical_cols)
    
    # interactively let user add new categorical columns
    col1, col2 = st.columns(2)
    with col1:
        with st.form("add_categorical_col_form", clear_on_submit=True):
            new_categorical_col = st.text_input("Add categorical column to available categorical columns", placeholder="e.g., experiment")
            submitted = st.form_submit_button("Add")
            if submitted:
                if new_categorical_col not in categorical_cols:
                    categorical_cols.append(new_categorical_col)
                    cfg["categorical_cols"] = categorical_cols
                    # Add the new column to selected columns (user likely wants it selected)
                    if new_categorical_col not in st.session_state.selected_categorical_cols:
                        st.session_state.selected_categorical_cols.append(new_categorical_col)
    with col2:
        # render a multiselect for existing categorical columns
        # Use cfg["categorical_cols"] to ensure we get the most up-to-date list
        current_categorical_cols = cfg.get("categorical_cols", categorical_cols)
        
        # Filter session state to only include columns that still exist in current options
        valid_selected_cols = [col for col in st.session_state.selected_categorical_cols 
                              if col in current_categorical_cols]
        
        selected_categorical_cols = st.multiselect(
            "Select Categorical Columns", 
            current_categorical_cols, 
            default=valid_selected_cols,
            key="categorical_cols_multiselect"
        )
        # Update session state with current selection
        st.session_state.selected_categorical_cols = selected_categorical_cols

    if st.button("Save Configuration"):
        save_config(cfg, _ANALYSIS_CONFIG_PATH)
   
def get_unique_row_id_col():
    cfg = load_config(_ANALYSIS_CONFIG_PATH)
    return cfg.get("unique_row_id_col", "")
def get_fov_name_col_analysis():
    cfg = load_config(_ANALYSIS_CONFIG_PATH)
    return cfg.get("fov_name_col", "")
def get_categorical_cols_analysis():
    cfg = load_config(_ANALYSIS_CONFIG_PATH)
    categorical_cols = cfg.get("categorical_cols", [])

    ## platform specific categorical columns (used by 1d GMM and 2d GMM)
    if "GMM_group" not in categorical_cols:
        categorical_cols.append("GMM_group")
    if "2D_GMM_group" not in categorical_cols:
        categorical_cols.append("2D_GMM_group")
    return categorical_cols
def get_all_numeric_col_groups():
    cfg = load_config(_ANALYSIS_CONFIG_PATH)
    return cfg.get("all_numeric_col_groups", [])

unique_row_id_col = get_unique_row_id_col()
fov_name_col = get_fov_name_col_analysis()
categorical_cols = get_categorical_cols_analysis()
all_numeric_col_groups = get_all_numeric_col_groups()




