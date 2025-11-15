import streamlit as st
from streamlit_sortables import sort_items
from pathlib import Path
import sys
from src.config import load_config, save_config, get_unique_cell_id_col, get_fov_name_col, get_all_feature_extractors, get_categorical_cols

# Maximum number of profiles allowed
MAX_PROFILES = 10

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

def _migrate_old_config_to_profiles(cfg: dict) -> dict:
    """Migrate old config format (flat) to new profile-based format."""
    if "profiles" in cfg:
        return cfg  # Already migrated
    
    # Check if old format exists
    if any(key in cfg for key in ["unique_row_id_col", "fov_name_col", "categorical_cols", "feature_groups", "all_numerical_features"]):
        # Migrate to profile-based format
        profiles = {
            "default": {
                "unique_row_id_col": cfg.get("unique_row_id_col", ""),
                "fov_name_col": cfg.get("fov_name_col", ""),
                "categorical_cols": cfg.get("categorical_cols", []),
                "feature_groups": cfg.get("feature_groups", {}),
                "all_numerical_features": cfg.get("all_numerical_features", [])
            }
        }
        # Remove old keys
        for key in ["unique_row_id_col", "fov_name_col", "categorical_cols", "feature_groups", "all_numerical_features"]:
            cfg.pop(key, None)
        cfg["profiles"] = profiles
        cfg["current_profile"] = "default"
    
    return cfg

def _get_current_profile() -> str:
    """Get the current active profile from session state."""
    if "current_profile" not in st.session_state:
        cfg = load_config(_ANALYSIS_CONFIG_PATH)
        cfg = _migrate_old_config_to_profiles(cfg)
        st.session_state.current_profile = cfg.get("current_profile", "default")
        # Ensure profile exists
        if "profiles" not in cfg:
            cfg["profiles"] = {}
        if st.session_state.current_profile not in cfg["profiles"]:
            cfg["profiles"][st.session_state.current_profile] = {}
            save_config(cfg, _ANALYSIS_CONFIG_PATH)
    return st.session_state.current_profile

def _get_profile_config(profile_name: str = None) -> dict:
    """Get config for a specific profile, or current profile if None."""
    if profile_name is None:
        profile_name = _get_current_profile()
    
    cfg = load_config(_ANALYSIS_CONFIG_PATH)
    cfg = _migrate_old_config_to_profiles(cfg)
    
    if "profiles" not in cfg:
        cfg["profiles"] = {}
    if profile_name not in cfg["profiles"]:
        cfg["profiles"][profile_name] = {}
    
    return cfg["profiles"][profile_name]

def _save_profile_config(profile_name: str, profile_data: dict):
    """Save config for a specific profile."""
    cfg = load_config(_ANALYSIS_CONFIG_PATH)
    cfg = _migrate_old_config_to_profiles(cfg)
    
    if "profiles" not in cfg:
        cfg["profiles"] = {}
    
    cfg["profiles"][profile_name] = profile_data
    cfg["current_profile"] = profile_name
    save_config(cfg, _ANALYSIS_CONFIG_PATH)

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
    
    # Load and migrate config
    cfg = load_config(_ANALYSIS_CONFIG_PATH)
    cfg = _migrate_old_config_to_profiles(cfg)
    
    # Get current profile
    current_profile = _get_current_profile()
    profile_cfg = _get_profile_config(current_profile)
    
    # Initialize profile config with defaults if needed
    if "unique_row_id_col" not in profile_cfg:
        profile_cfg["unique_row_id_col"] = unique_cell_id_col
    if "fov_name_col" not in profile_cfg:
        profile_cfg["fov_name_col"] = fov_name_col
    if "categorical_cols" not in profile_cfg:
        profile_cfg["categorical_cols"] = categorical_cols
        
    if use_data_extraction:
        # do nothing 
        _save_profile_config(current_profile, profile_cfg)
        return
    
    st.header("Tell me about ur data")
    
    # Profile selector at top left in three columns
    cols_header = st.columns(3)
    
    # Get available profiles
    available_profiles = list(cfg.get("profiles", {}).keys())
    if not available_profiles:
        available_profiles = ["default"]
        cfg["profiles"] = {"default": {}}
    
    with cols_header[0]:
        # Profile selector
        selected_profile = st.selectbox(
            "Profile",
            options=available_profiles,
            index=available_profiles.index(current_profile) if current_profile in available_profiles else 0,
            help="Select which profile to configure",
            key="profile_selector"
        )
        
        # Update current profile if changed
        if selected_profile != current_profile:
            st.session_state.current_profile = selected_profile
            # Clear session state for feature groups to reload new profile's data (both old and new profile keys)
            old_feature_groups_key = f"feature_groups_{current_profile}"
            new_feature_groups_key = f"feature_groups_{selected_profile}"
            old_numerical_features_key = f"all_numerical_features_multiselect_{current_profile}"
            new_numerical_features_key = f"all_numerical_features_multiselect_{selected_profile}"
            old_categorical_cols_key = f"categorical_cols_multiselect_{current_profile}"
            new_categorical_cols_key = f"categorical_cols_multiselect_{selected_profile}"
            # Clear old profile's session state (optional, but helps with memory)
            if old_feature_groups_key in st.session_state:
                del st.session_state[old_feature_groups_key]
            if old_numerical_features_key in st.session_state:
                del st.session_state[old_numerical_features_key]
            if old_categorical_cols_key in st.session_state:
                del st.session_state[old_categorical_cols_key]
            # Clear new profile's session state to force reload from config
            if new_feature_groups_key in st.session_state:
                del st.session_state[new_feature_groups_key]
            if new_numerical_features_key in st.session_state:
                del st.session_state[new_numerical_features_key]
            if new_categorical_cols_key in st.session_state:
                del st.session_state[new_categorical_cols_key]
            st.rerun()
    
    with cols_header[1]:
        # Create new profile button
        if len(available_profiles) < MAX_PROFILES:
            with st.form("create_profile_form", clear_on_submit=True):
                new_profile_name = st.text_input(
                    "New Profile Name",
                    placeholder="e.g. Profile 2",
                    help=f"Create a new profile (max {MAX_PROFILES} profiles)",
                    key="new_profile_name_input"
                )
                create_submitted = st.form_submit_button("➕ Create Profile", type="secondary")
                
                if create_submitted and new_profile_name:
                    new_profile_name = new_profile_name.strip()
                    if new_profile_name and new_profile_name not in available_profiles:
                        # Create new profile with defaults
                        cfg["profiles"][new_profile_name] = {
                            "unique_row_id_col": unique_cell_id_col,
                            "fov_name_col": fov_name_col,
                            "categorical_cols": categorical_cols.copy(),
                            "feature_groups": {},
                            "all_numerical_features": []
                        }
                        cfg["current_profile"] = new_profile_name
                        save_config(cfg, _ANALYSIS_CONFIG_PATH)
                        st.session_state.current_profile = new_profile_name
                        # Clear session state to reload (profile-specific keys)
                        new_feature_groups_key = f"feature_groups_{new_profile_name}"
                        new_numerical_features_key = f"all_numerical_features_multiselect_{new_profile_name}"
                        if new_feature_groups_key in st.session_state:
                            del st.session_state[new_feature_groups_key]
                        if new_numerical_features_key in st.session_state:
                            del st.session_state[new_numerical_features_key]
                        st.rerun()
                    elif new_profile_name in available_profiles:
                        st.error(f"Profile '{new_profile_name}' already exists!")
        else:
            st.info(f"Maximum {MAX_PROFILES} profiles reached")
    
    with cols_header[2]:
        # Delete profile form
        if len(available_profiles) > 1:
            with st.form("delete_profile_form", clear_on_submit=True):
                st.write("**Delete Profile**")
                delete_submitted = st.form_submit_button("🗑️ Delete Profile", type="secondary")
                
                if delete_submitted:
                    # Delete the profile from config
                    del cfg["profiles"][selected_profile]
                    # If deleted profile was current, switch to first available profile
                    if cfg.get("current_profile") == selected_profile:
                        remaining_profiles = [p for p in available_profiles if p != selected_profile]
                        if remaining_profiles:
                            cfg["current_profile"] = remaining_profiles[0]
                            st.session_state.current_profile = remaining_profiles[0]
                    save_config(cfg, _ANALYSIS_CONFIG_PATH)
                    # Clear session state for deleted profile
                    deleted_feature_groups_key = f"feature_groups_{selected_profile}"
                    deleted_numerical_features_key = f"all_numerical_features_multiselect_{selected_profile}"
                    deleted_categorical_cols_key = f"categorical_cols_multiselect_{selected_profile}"
                    for key in [deleted_feature_groups_key, deleted_numerical_features_key, deleted_categorical_cols_key]:
                        if key in st.session_state:
                            del st.session_state[key]
                    st.success(f"Profile '{selected_profile}' deleted successfully!")
                    st.rerun()
        else:
            st.info("Cannot delete the only profile")
    
    # Reload profile config after potential profile switch
    current_profile = _get_current_profile()
    profile_cfg = _get_profile_config(current_profile)
    
    cols = st.columns(2)
    with cols[0]:
        profile_cfg["unique_row_id_col"] = st.text_input("Unique Row ID", value=profile_cfg.get("unique_row_id_col", unique_cell_id_col), help="The column name that uniquely identifies each row in the dataset.")
    with cols[1]:
        profile_cfg["fov_name_col"] = st.text_input("FOV column name (if applicable)", value=profile_cfg.get("fov_name_col", fov_name_col), help="The column name that uniquely identifies each field of view in the dataset. Your dataset may not have this. It is ok.")
    
    selected_categorical_cols = st.multiselect(
        "Select Categorical Columns", 
        profile_cfg.get("categorical_cols", categorical_cols), 
        default=profile_cfg.get("categorical_cols", categorical_cols),
        help="Select the categorical columns you may have in this or future datasets.",
        key=f"categorical_cols_multiselect_{current_profile}",
        accept_new_options=True
    )
  
    # now let user define feature groups
    feature_groups_widget()
    col1, col2 = st.columns(2)
   
    with col1:
        if st.button("Save Configuration"):
            profile_cfg["categorical_cols"] = selected_categorical_cols
            profile_cfg["unique_row_id_col"] = profile_cfg["unique_row_id_col"]
            profile_cfg["fov_name_col"] = profile_cfg["fov_name_col"]
            # Also save feature groups if they exist in session state (profile-specific)
            feature_groups_key = f"feature_groups_{current_profile}"
            if feature_groups_key in st.session_state:
                profile_cfg["feature_groups"] = st.session_state[feature_groups_key]
            # Save selected numerical features (profile-specific)
            numerical_features_key = f"all_numerical_features_multiselect_{current_profile}"
            if numerical_features_key in st.session_state:
                profile_cfg["all_numerical_features"] = st.session_state[numerical_features_key]
            _save_profile_config(current_profile, profile_cfg)
            st.session_state.config_saved = True  # Set flag for success message
            st.rerun()
        
        # Display success message if flag is set
        if st.session_state.get("config_saved", False):
            st.success("Configuration saved successfully!")
            st.session_state.config_saved = False  # Clear the flag
    with col2:
        if st.button("Reset Configuration"):
            profile_cfg["unique_row_id_col"] = unique_cell_id_col
            profile_cfg["fov_name_col"] = fov_name_col
            profile_cfg["categorical_cols"] = categorical_cols
            profile_cfg["feature_groups"] = {}
            profile_cfg["all_numerical_features"] = []
            _save_profile_config(current_profile, profile_cfg)
            # Clear session state to reload (profile-specific)
            feature_groups_key = f"feature_groups_{current_profile}"
            numerical_features_key = f"all_numerical_features_multiselect_{current_profile}"
            if feature_groups_key in st.session_state:
                del st.session_state[feature_groups_key]
            if numerical_features_key in st.session_state:
                del st.session_state[numerical_features_key]
            st.session_state.config_reset = True  # Set flag for success message
            st.rerun()
        
        # Display success message if flag is set
        if st.session_state.get("config_reset", False):
            st.success("Configuration reset successfully!")
            st.session_state.config_reset = False  # Clear the flag

def feature_groups_widget():
    # Get current profile and its config
    current_profile = _get_current_profile()
    profile_cfg = _get_profile_config(current_profile)
    all_feature_extractors = get_all_feature_extractors()
    
    # Initialize feature groups in profile config if not exists
    if "feature_groups" not in profile_cfg:
        profile_cfg["feature_groups"] = {}
        
    # step 1: use text area to let user copy and paste all the features
    raw = st.text_area(
        "Paste numerical features (comma, semicolon, or whitespace separated)",
        placeholder="feat1, feat2; feat3\nfeat4 feat5",
        key=f"paste_box_{current_profile}",
    )
    
    # Parse features from raw text
    parsed_features = parse_features(raw)
    
    # Get features from profile config if available
    config_features = profile_cfg.get("all_numerical_features", [])
    
    # Determine available features based on logic:
    # If config has all_numerical_features, use union of paste + config
    # If no config, use parsed features from paste area
    if config_features:
        available_features = list(set(parsed_features + config_features))
    else:
        available_features = parsed_features
    
    # Initialize profile-specific session state keys (needed in both if/else branches)
    feature_groups_key = f"feature_groups_{current_profile}"
    sortable_refresh_key_name = f"sortable_refresh_key_{current_profile}"
    previous_features_key = f"previous_features_{current_profile}"
    
    # Show multiselect for all numerical features
    if available_features:
        st.subheader("📊 Select Numerical Features")
        selected_features = st.multiselect(
            "Choose which features to use for analysis",
            options=available_features,
            default=config_features if config_features else available_features,
            help="Include the numerical features you want to organize into groups",
            key=f"all_numerical_features_multiselect_{current_profile}"
        )
        
        # Use selected features as the main features list
        features = selected_features
        
        # Initialize session state for feature groups (profile-specific)
        if feature_groups_key not in st.session_state:
            st.session_state[feature_groups_key] = profile_cfg.get("feature_groups", {})
        
        # Ensure feature_groups is a dictionary
        if not isinstance(st.session_state[feature_groups_key], dict):
            st.session_state[feature_groups_key] = {}
        
        # Initialize sortable refresh counter (profile-specific)
        if sortable_refresh_key_name not in st.session_state:
            st.session_state[sortable_refresh_key_name] = 0
        
        # Track previous multiselect selection to detect changes (profile-specific)
        if previous_features_key not in st.session_state:
            st.session_state[previous_features_key] = features
        
        # Sync feature groups with multiselect selection
        # Remove features that are no longer selected from all groups
        features_set = set(features)
        previous_features_set = set(st.session_state[previous_features_key])
        
        # If selection changed, update groups and refresh sortable
        if features_set != previous_features_set:
            for group_name in st.session_state[feature_groups_key]:
                # Filter out deselected features from each group
                st.session_state[feature_groups_key][group_name] = [
                    f for f in st.session_state[feature_groups_key][group_name] 
                    if f in features_set
                ]
            # Force sortable refresh when multiselect changes
            st.session_state[sortable_refresh_key_name] += 1
            st.session_state[previous_features_key] = features
    else:
        features = []
        # Initialize session state even when no features
        if feature_groups_key not in st.session_state:
            st.session_state[feature_groups_key] = profile_cfg.get("feature_groups", {})
        if sortable_refresh_key_name not in st.session_state:
            st.session_state[sortable_refresh_key_name] = 0
        if previous_features_key not in st.session_state:
            st.session_state[previous_features_key] = []
    
    if not features:
        st.info("Please paste numerical features above and we will help you organize them into groups.")
        return
    
    st.subheader("Feature Groups Management")
    
    # Create and Delete forms side by side
    col1, col2 = st.columns(2)
    
    with col1:
        # Form to create new feature groups
        with st.form(f"create_feature_group_form_{current_profile}", clear_on_submit=True):
            st.write("**Create New Feature Group**")
            new_group_name = st.text_input(
                "Group Name", 
                placeholder="e.g. lifetime, morphology, texture",
                help="Enter a name for the new feature group"
            )
            submitted = st.form_submit_button("Create Group")
            
            if submitted and new_group_name:
                if new_group_name not in st.session_state[feature_groups_key]:
                    st.session_state[feature_groups_key][new_group_name] = []
                    st.session_state[sortable_refresh_key_name] += 1  # Force sortable refresh
                    st.success(f"Created feature group: '{new_group_name}'")
                    st.rerun()
                else:
                    st.error(f"Group '{new_group_name}' already exists!")
    
    with col2:
        # Delete feature groups
        if st.session_state[feature_groups_key]:
            with st.form(f"delete_feature_group_form_{current_profile}", clear_on_submit=True):
                st.write("**Delete Feature Group**")
                group_to_delete = st.selectbox(
                    "Select group to delete",
                    options=list(st.session_state[feature_groups_key].keys()),
                    help="Select a feature group to remove. Features will be moved back to available pool."
                )
                delete_submitted = st.form_submit_button("🗑️ Delete Group", type="secondary")
                
                if delete_submitted and group_to_delete:
                    # Remove the group from session state
                    del st.session_state[feature_groups_key][group_to_delete]
                    st.session_state[sortable_refresh_key_name] += 1  # Force sortable refresh
                    st.success(f"Deleted feature group: '{group_to_delete}'")
                    st.rerun()
        else:
            st.info("Create some feature groups first to enable deletion.")
    
    # Drag and Drop Interface  
    if features or st.session_state[feature_groups_key]:
        st.subheader("📋 Drag & Drop Feature Assignment")
        
        # Show helpful message when no features selected but groups exist
        if not features and st.session_state[feature_groups_key]:
            st.warning("⚠️ No features selected in multiselect above. Your feature groups are now empty. Select features to populate groups again.")
        
        # Calculate uncategorized features
        categorized_features = set()
        for group_name, group_features in st.session_state[feature_groups_key].items():
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
        for group_name, group_features in st.session_state[feature_groups_key].items():
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
                key=f"feature_groups_sortable_{current_profile}_{st.session_state[sortable_refresh_key_name]}"
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
                        if group_name in st.session_state[feature_groups_key]:
                            new_assignments[group_name] = items
                            # Check if this group's items actually changed
                            if st.session_state[feature_groups_key][group_name] != items:
                                has_changes = True
                
                # Only update and rerun if there were actual changes
                if has_changes:
                    for group_name, items in new_assignments.items():
                        st.session_state[feature_groups_key][group_name] = items
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
    current_profile = _get_current_profile()
    profile_cfg = _get_profile_config(current_profile)
    return profile_cfg.get("unique_row_id_col", "")

def get_fov_name_col_analysis(use_data_extraction=True):
    if use_data_extraction:
        return get_fov_name_col()
    current_profile = _get_current_profile()
    profile_cfg = _get_profile_config(current_profile)
    return profile_cfg.get("fov_name_col", "")

def get_categorical_cols_analysis(use_data_extraction=True):
    if use_data_extraction:
        data_extraction_categorical_cols = get_categorical_cols()
        fov_name_col = get_fov_name_col()
        return data_extraction_categorical_cols + [fov_name_col, "GMM_group", "2D_GMM_group", "k_means_cluster"]
    current_profile = _get_current_profile()
    profile_cfg = _get_profile_config(current_profile)
    categorical_cols = profile_cfg.get("categorical_cols", [])

    ## platform specific categorical columns (used by 1d GMM and 2d GMM and K-Means clustering)
    if "GMM_group" not in categorical_cols:
        categorical_cols.append("GMM_group")
    if "2D_GMM_group" not in categorical_cols:
        categorical_cols.append("2D_GMM_group")
    if "k_means_cluster" not in categorical_cols:
        categorical_cols.append("k_means_cluster")
    return categorical_cols

def get_all_feature_groups():
    current_profile = _get_current_profile()
    profile_cfg = _get_profile_config(current_profile)
    return profile_cfg.get("feature_groups", {})

def get_all_numerical_features():
    current_profile = _get_current_profile()
    profile_cfg = _get_profile_config(current_profile)
    return profile_cfg.get("all_numerical_features", [])




