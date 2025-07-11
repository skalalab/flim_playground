"""
Numerical and Categorical Features for Data Extraction outputted CSV file. If new features are added, modify the lists below.
"""

from src.config import get_unique_cell_id_col, get_image_name_col

unique_cell_id_col = get_unique_cell_id_col()
image_name_col = get_image_name_col()
required_cols = [unique_cell_id_col, image_name_col]

# customizable categorical columns
categorical_cols = [ "experiment", "day", "hour", "cell_type", "media", "dish", "cell_line", "treatment", "condition", "patient_id", "replicate", "GMM_group", "2D_GMM_group"]


# Numerical Feature


# customizable numeric column dicionary that store the default names of the features we want to extract (for backward compatibility)
feature_groups_default = {
    "Nadh Fit": ["ntm", "na1", "na2", "nt1", "nt2", "nint", "normrr", "nadh_tau_mean_mean","nadh_a1_mean", "nadh_a2_mean", "nadh_t1_mean", "nadh_t2_mean"],
    "Fad Fit": ["ftm", "fa1", "fa2", "ft1", "ft2", "fint", "fad_tau_mean_mean", "fad_a1_mean", "fad_a2_mean", "fad_t1_mean", "fad_t2_mean"],
    "Mask Morphology": ["area", "perimeter", "solidity", "eccentricity", "axis_major_length", "axis_minor_length"],
    "Feature Distribution Fit": [],
    "Fit Free Nadh": ["phasor_x", "phasor_y"],
    "Fit Free Fad": ["phasor_x", "phasor_y"],
    "Feature Distribution Fit Free": [],
    "Mitochrondria Feature": ["mito_area"],
}

# customizable numeric column prefixes (in Data Extraction module, it will produce the features with their corresponding feature group prefixes)
# used in Visualization module and Classification module to math features in the uploaded csv file
feature_groups_prefix = {
    "Nadh Fit": "fit_nadh: ", # put redox in nadh
    "Fad Fit":"fit_fad: ",
    "Mask Morphology": "mask_morphology: ",
    "Feature Distribution Fit": "fd_fit: ",
    "Fit Free Nadh": "fit_free_nadh: ",
    "Fit Free Fad": "fit_free_fad: ",
    "Feature Distribution Fit Free": "fd_fit_free: ",
    "Mitochrondria Feature": "mito: ",

    "Uncategorized Features": "", # features that are not in the above groups, a fallback, should always be the last
}

# feature groups names that are used to name widgets that host those feature groups
# In Python 3.7+, dictionary keys maintain insertion order.
# Convert keys view to a list to explicitly capture this order.
feature_groups = list(feature_groups_prefix.keys())
# host all features that are capable of being extracted from raw data
# used only in data_extraction module to name the features that can be extracted 
feature_groups_features = {
    "Nadh Fit": ["a1", "a2", "t1", "t2", "tm", "intensity", "norm_redox"], # put redox in nadh 
    "Fad Fit":["a1", "a2", "t1", "t2", "tm", "intensity"],
    "Mask Morphology": ["area", "perimeter", "solidity", "eccentricity", "major_axis_length", "minor_axis_length", "circularity"],
    "Fit Free Nadh": ["G(1st)", "S(1st)", "G(2nd)", "S(2nd)", "Tau_phase", "Tau_m"],
    "Fit Free Fad": ["G(1st)", "S(1st)", "G(2nd)", "S(2nd)", "Tau_phase", "Tau_m"],
}

feature_distribution_vars = ["polarity"]
for feature_distribution_var in feature_distribution_vars:
    feature_groups_features["Feature Distribution Fit"] = [ "nadh_" +  feature  + "_" + feature_distribution_var for feature in feature_groups_features["Nadh Fit"]]
    feature_groups_features["Feature Distribution Fit"] += [ "fad_" +  feature  + "_" + feature_distribution_var for feature in feature_groups_features["Fad Fit"]]
    feature_groups_features["Feature Distribution Fit Free"] = [ "nadh_" + feature + "_" + feature_distribution_var for feature in feature_groups_features["Fit Free Nadh"] ]
    feature_groups_features["Feature Distribution Fit Free"] += [ "fad_" + feature + "_" + feature_distribution_var for feature in feature_groups_features["Fit Free Fad"]]

def get_full_feature_name(feature_groups):
    """
    takes the feature groups and returns the feature groups in human friendly names
    """
    feature_groups_names = {}
    for feature_group in feature_groups:
        if feature_group in feature_groups_prefix:
            prefix = feature_groups_prefix[feature_group]
            feature_groups_names[feature_group] = [prefix + feature for feature in feature_groups_features[feature_group]]
        else:
            feature_groups_names[feature_group] = feature_groups_features[feature_group]
    return feature_groups_names

def subset_feature_group_features(has_nadh=True, has_fad=True, fit_free=True, has_mask=True, feature_distribution=True):
    """
    Subset the feature groups based on the selected features
    """
    feature_groups_subset = {}
    if has_mask:
        feature_groups_subset["Mask Morphology"] = feature_groups_features["Mask Morphology"]
    if has_nadh:
        feature_groups_subset["Nadh Fit"] = feature_groups_features["Nadh Fit"]
        if feature_distribution:
            feature_groups_subset["Feature Distribution Fit"] = [
                feat for feat in feature_groups_features["Feature Distribution Fit"]
                if feat.startswith("nadh_")
            ]
        if fit_free:
            feature_groups_subset["Fit Free Nadh"] = feature_groups_features["Fit Free Nadh"]
            if feature_distribution:
                feature_groups_subset["Feature Distribution Fit Free"] = [
                    feat for feat in feature_groups_features["Feature Distribution Fit Free"]
                    if feat.startswith("nadh_")
                ]
    if has_fad:
        feature_groups_subset["Fad Fit"] = feature_groups_features["Fad Fit"]
        if feature_distribution:
            feature_groups_subset["Feature Distribution Fit"] = [
                feat for feat in feature_groups_features["Feature Distribution Fit"]
                if feat.startswith("fad_")
            ]
        if fit_free:
            feature_groups_subset["Fit Free Fad"] = feature_groups_features["Fit Free Fad"]
            if feature_distribution:
                feature_groups_subset["Feature Distribution Fit Free"] = [
                    feat for feat in feature_groups_features["Feature Distribution Fit Free"]
                    if feat.startswith("fad_")
                ]
    return feature_groups_subset