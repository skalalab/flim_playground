import streamlit as st
from streamlit_sortables import sort_items
from pathlib import Path
import sys
from src.config import load_config, save_config, get_unique_cell_id_col, get_fov_name_col, get_all_feature_extractors, get_categorical_cols

def _get_analysis_config_path() -> Path:
    """Get the analysis config file path, handling both development and bundled app scenarios."""
    # Check if running as a PyInstaller bundle
    if getattr(sys, '_MEIPASS', None):
        # Running as bundled app - save config to a persistent location
        # Use the directory where the executable is located
        exe_dir = Path(sys.executable).parent
        config_path = exe_dir / "analysis_config.toml"
        
        # If config doesn't exist in exe directory, try to copy from bundle
        if not config_path.exists():
            bundled_config = Path(sys._MEIPASS) / "analysis_config.toml"
            if bundled_config.exists():
                import shutil
                shutil.copy(bundled_config, config_path)
        
        return config_path
    else:
        # Running in development mode - use analysis_config.toml in project root
        return Path(__file__).resolve().parent.parent.parent / "analysis_config.toml"

# Absolute path to the analysis config file - handles both dev and bundled scenarios
_ANALYSIS_CONFIG_PATH = _get_analysis_config_path()

def dataset_config_widget(use_data_extraction=True):
    if "config_saved" not in st.session_state:
        st.session_state.config_saved = False
    if "config_reset" not in st.session_state:
        st.session_state.config_reset = False

    # read from the data_extraction configuration and modify based on the use_data_extraction flag
    unique_cell_id_col = get_unique_cell_id_col()
    fov_name_col = get_fov_name_col()
    categorical_cols = get_categorical_cols()
    categorical_cols.extend([fov_name_col])
    cfg = load_config(_ANALYSIS_CONFIG_PATH)
   
    if "unique_row_id_col" not in cfg:
        cfg["unique_row_id_col"] = unique_cell_id_col
    if "fov_name_col" not in cfg:
        cfg["fov_name_col"] = fov_name_col
    if "categorical_cols" not in cfg:
        cfg["categorical_cols"] = categorical_cols
        
    if use_data_extraction:
        # do nothing 
        save_config(cfg, _ANALYSIS_CONFIG_PATH)
        return
    
    st.header("Tell me about ur data")
    cols = st.columns(2)
    with cols[0]:
        cfg["unique_row_id_col"] = st.text_input("Unique Row ID", value= cfg["unique_row_id_col"], help="The column name that uniquely identifies each row in the dataset.")
    with cols[1]:
        cfg["fov_name_col"] = st.text_input("FOV column name (if applicable)", value= cfg["fov_name_col"], help="The column name that uniquely identifies each field of view in the dataset. Your dataset may not have this. It is ok.")
    
    selected_categorical_cols = st.multiselect(
        "Select Categorical Columns", 
        cfg.get("categorical_cols", categorical_cols), 
        default=cfg.get("categorical_cols", categorical_cols),
        help="Select the categorical columns you may have in this or future datasets.",
        key="categorical_cols_multiselect",
        accept_new_options=True
    )
  
    # now let user define feature groups
    feature_groups_widget()
    col1, col2 = st.columns(2)
   
    with col1:
        if st.button("Save Configuration"):
            cfg["categorical_cols"] = selected_categorical_cols
            # Also save feature groups if they exist in session state
            if "feature_groups" in st.session_state:
                cfg["feature_groups"] = st.session_state.feature_groups
            # Save selected numerical features
            if "all_numerical_features_multiselect" in st.session_state:
                cfg["all_numerical_features"] = st.session_state.all_numerical_features_multiselect
            save_config(cfg, _ANALYSIS_CONFIG_PATH)
            st.session_state.config_saved = True  # Set flag for success message
            st.rerun()
        
        # Display success message if flag is set
        if st.session_state.get("config_saved", False):
            st.success("Configuration saved successfully!")
            st.session_state.config_saved = False  # Clear the flag
    with col2:
        if st.button("Reset Configuration"):
            cfg = load_config(_ANALYSIS_CONFIG_PATH)
            cfg["unique_row_id_col"] = unique_cell_id_col
            cfg["fov_name_col"] = fov_name_col
            cfg["categorical_cols"] = categorical_cols
            cfg["feature_groups"] = {}
            cfg["all_numerical_features"] = []
            save_config(cfg, _ANALYSIS_CONFIG_PATH)
            st.session_state.config_reset = True  # Set flag for success message
            st.rerun()
        
        # Display success message if flag is set
        if st.session_state.get("config_reset", False):
            st.success("Configuration reset successfully!")
            st.session_state.config_reset = False  # Clear the flag

def feature_groups_widget():
    cfg = load_config(_ANALYSIS_CONFIG_PATH)
    all_feature_extractors = get_all_feature_extractors()
    
    # Initialize feature groups in config if not exists
    if "feature_groups" not in cfg:
        cfg["feature_groups"] = {}
        
    # step 1: use text area to let user copy and paste all the features
    raw = st.text_area(
        "Paste numerical features (comma, semicolon, or whitespace separated)",
        placeholder="feat1, feat2; feat3\nfeat4 feat5",
        key="paste_box",
    )
    
    # Parse features from raw text
    parsed_features = parse_features(raw)
    
    # Get features from config if available
    config_features = cfg.get("all_numerical_features", [])
    
    # Determine available features based on logic:
    # If config has all_numerical_features, use union of paste + config
    # If no config, use parsed features from paste area
    if config_features:
        available_features = list(set(parsed_features + config_features))
    else:
        available_features = parsed_features
    
    # Show multiselect for all numerical features
    if available_features:
        st.subheader("📊 Select Numerical Features")
        selected_features = st.multiselect(
            "Choose which features to use for analysis",
            options=available_features,
            default=config_features if config_features else available_features,
            help="Include the numerical features you want to organize into groups",
            key="all_numerical_features_multiselect"
        )
        
        # Use selected features as the main features list
        features = selected_features
        
        # Initialize session state for feature groups
        if "feature_groups" not in st.session_state:
            st.session_state.feature_groups = cfg.get("feature_groups", {})
        
        # Ensure feature_groups is a dictionary
        if not isinstance(st.session_state.feature_groups, dict):
            st.session_state.feature_groups = {}
        
        # Initialize sortable refresh counter
        if "sortable_refresh_key" not in st.session_state:
            st.session_state.sortable_refresh_key = 0
        
        # Track previous multiselect selection to detect changes
        if "previous_features" not in st.session_state:
            st.session_state.previous_features = features
        
        # Sync feature groups with multiselect selection
        # Remove features that are no longer selected from all groups
        features_set = set(features)
        previous_features_set = set(st.session_state.previous_features)
        
        # If selection changed, update groups and refresh sortable
        if features_set != previous_features_set:
            for group_name in st.session_state.feature_groups:
                # Filter out deselected features from each group
                st.session_state.feature_groups[group_name] = [
                    f for f in st.session_state.feature_groups[group_name] 
                    if f in features_set
                ]
            # Force sortable refresh when multiselect changes
            st.session_state.sortable_refresh_key += 1
            st.session_state.previous_features = features
    else:
        features = []
        # Initialize session state even when no features
        if "feature_groups" not in st.session_state:
            st.session_state.feature_groups = cfg.get("feature_groups", {})
        if "sortable_refresh_key" not in st.session_state:
            st.session_state.sortable_refresh_key = 0
        if "previous_features" not in st.session_state:
            st.session_state.previous_features = []
    
    if not features:
        st.info("Please paste numerical features above and we will help you organize them into groups.")
        return
    
    st.subheader("Feature Groups Management")
    
    # Create and Delete forms side by side
    col1, col2 = st.columns(2)
    
    with col1:
        # Form to create new feature groups
        with st.form("create_feature_group_form", clear_on_submit=True):
            st.write("**Create New Feature Group**")
            new_group_name = st.text_input(
                "Group Name", 
                placeholder="e.g. lifetime, morphology, texture",
                help="Enter a name for the new feature group"
            )
            submitted = st.form_submit_button("Create Group")
            
            if submitted and new_group_name:
                if new_group_name not in st.session_state.feature_groups:
                    st.session_state.feature_groups[new_group_name] = []
                    st.session_state.sortable_refresh_key += 1  # Force sortable refresh
                    st.success(f"Created feature group: '{new_group_name}'")
                    st.rerun()
                else:
                    st.error(f"Group '{new_group_name}' already exists!")
    
    with col2:
        # Delete feature groups
        if st.session_state.feature_groups:
            with st.form("delete_feature_group_form", clear_on_submit=True):
                st.write("**Delete Feature Group**")
                group_to_delete = st.selectbox(
                    "Select group to delete",
                    options=list(st.session_state.feature_groups.keys()),
                    help="Select a feature group to remove. Features will be moved back to available pool."
                )
                delete_submitted = st.form_submit_button("🗑️ Delete Group", type="secondary")
                
                if delete_submitted and group_to_delete:
                    # Remove the group from session state
                    del st.session_state.feature_groups[group_to_delete]
                    st.session_state.sortable_refresh_key += 1  # Force sortable refresh
                    st.success(f"Deleted feature group: '{group_to_delete}'")
                    st.rerun()
        else:
            st.info("Create some feature groups first to enable deletion.")
    
    # Drag and Drop Interface  
    if features or st.session_state.feature_groups:
        st.subheader("📋 Drag & Drop Feature Assignment")
        
        # Show helpful message when no features selected but groups exist
        if not features and st.session_state.feature_groups:
            st.warning("⚠️ No features selected in multiselect above. Your feature groups are now empty. Select features to populate groups again.")
        
        # Calculate uncategorized features
        categorized_features = set()
        for group_name, group_features in st.session_state.feature_groups.items():
            if isinstance(group_features, list):
                categorized_features.update(group_features)
        uncategorized_features = [f for f in features if f not in categorized_features]
        
        # Prepare containers for sortables (correct format for multi_containers)
        container_list = [
            {
                "header": "🔄 Available Features",
                "items": uncategorized_features
            }
        ]
        
        # Add feature group containers
        for group_name, group_features in st.session_state.feature_groups.items():
            container_list.append({
                "header": f"📁 {group_name}",
                "items": group_features
            })
        
        # Create the sortable interface
        try:
            sorted_items = sort_items(
                container_list,
                multi_containers=True,
                direction="vertical",
                key=f"feature_groups_sortable_{st.session_state.sortable_refresh_key}"
            )
            
            # Update session state with new assignments
            if sorted_items:
                # Check if there were actual changes by comparing the content
                has_changes = False
                new_assignments = {}
                
                for container in sorted_items:
                    header = container["header"]
                    items = container["items"]
                    
                    # Extract group name from header (remove emoji and spaces)
                    if header.startswith("📁 "):
                        group_name = header[2:].strip()  # Remove "📁 " prefix
                        if group_name in st.session_state.feature_groups:
                            new_assignments[group_name] = items
                            # Check if this group's items actually changed
                            if st.session_state.feature_groups[group_name] != items:
                                has_changes = True
                
                # Only update and rerun if there were actual changes
                if has_changes:
                    for group_name, items in new_assignments.items():
                        st.session_state.feature_groups[group_name] = items
                    st.rerun()
        except Exception as e:
            st.error(f"Error with drag and drop interface: {str(e)}")
            st.info("Please try refreshing the page or recreating your feature groups.")
   
    
def parse_features(text: str):
    if not text:
        return []
    # Normalize separators to newline, then split on any whitespace
    norm = text.replace(",", "\n").replace(";", "\n")
    toks = [t.strip() for t in norm.split()]
    # de-duplicate, keep first occurrence order
    out, seen = [], set()
    for t in toks:
        if t and t not in seen:
            seen.add(t); out.append(t)
    return out

def get_unique_row_id_col(use_data_extraction=True):
    if use_data_extraction:
        return get_unique_cell_id_col()
    cfg = load_config(_ANALYSIS_CONFIG_PATH)
    return cfg.get("unique_row_id_col", "")

def get_fov_name_col_analysis(use_data_extraction=True):
    if use_data_extraction:
        return get_fov_name_col()
    cfg = load_config(_ANALYSIS_CONFIG_PATH)
    return cfg.get("fov_name_col", "")

def get_categorical_cols_analysis(use_data_extraction=True):
    if use_data_extraction:
        data_extraction_categorical_cols = get_categorical_cols()
        fov_name_col = get_fov_name_col()
        return data_extraction_categorical_cols + [fov_name_col, "GMM_group", "2D_GMM_group", "k_means_cluster"]
    cfg = load_config(_ANALYSIS_CONFIG_PATH)
    categorical_cols = cfg.get("categorical_cols", [])

    ## platform specific categorical columns (used by 1d GMM and 2d GMM and K-Means clustering)
    if "GMM_group" not in categorical_cols:
        categorical_cols.append("GMM_group")
    if "2D_GMM_group" not in categorical_cols:
        categorical_cols.append("2D_GMM_group")
    if "k_means_cluster" not in categorical_cols:
        categorical_cols.append("k_means_cluster")
    return categorical_cols
def get_all_feature_groups():
    cfg = load_config(_ANALYSIS_CONFIG_PATH)
    return cfg.get("feature_groups", {})

def get_all_numerical_features():
    cfg = load_config(_ANALYSIS_CONFIG_PATH)
    return cfg.get("all_numerical_features", [])




