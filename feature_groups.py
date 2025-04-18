"""
The meta data for the Data Extraction outputted CSV file. Can be extended. If new features are added, modify the lists below.
"""
# required column: unique cell identifier
required_cols = ["cell_id", "image_name"]

# customizable categorical columns
categorical_cols = [ "experiment", "day", "cell_type", "media", "cell_line", "treatment", "dish", "patient_id", "GMM_group"]

# customizable numeric column dicionary that store the default names of the features we want to extract (for backward compatibility)
feature_groups_default = {
    "Nadh Fit": ["ntm", "na1", "na2", "nt1", "nt2", "nint", "normrr", "nadh_tau_mean_mean","nadh_a1_mean", "nadh_a2_mean", "nadh_t1_mean", "nadh_t2_mean"],
    "Fad Fit": ["ftm", "fa1", "fa2", "ft1", "ft2", "fint", "fad_tau_mean_mean", "fad_a1_mean", "fad_a2_mean", "fad_t1_mean", "fad_t2_mean"],
    "Mask Morphology": ["area", "perimeter", "solidity", "eccentricity", "major_axis_length", "minor_axis_length"],
    "Feature Distribution Fit": ["cell_mean", "cell_std", "cell_median"],
    "Fit Free Nadh": ["phasor_x", "phasor_y"],
    "Fit Free Fad": ["phasor_x", "phasor_y"],
    "Feature Distribution Fit Free": ["cell_mean", "cell_std", "cell_median"],
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
    "Nadh Fit": ["a1", "a2", "t1", "t2", "tm", "intensity", "redox"], # put redox in nadh 
    "Fad Fit":["a1", "a2", "t1", "t2", "tm", "intensity"],
    "Mask Morphology": ["area", "perimeter", "solidity", "eccentricity", "major_axis_length", "minor_axis_length"],
    "Fit Free Nadh": ["G(1st)", "S(1st)", "G(2nd)", "S(2nd)"],
    "Fit Free Fad": ["G(1st)", "S(1st)", "G(2nd)", "S(2nd)"],
}

feature_distribution_vars = ["polarity"]
for feature_distribution_var in feature_distribution_vars:
    feature_groups_features["Feature Distribution Fit"] = [ "nadh_" +  feature  + "_" + feature_distribution_var for feature in feature_groups_features["Nadh Fit"]]
    feature_groups_features["Feature Distribution Fit"] += [ "fad_" +  feature  + "_" + feature_distribution_var for feature in feature_groups_features["Fad Fit"]]
    feature_groups_features["Feature Distribution Fit Free"] = [ "nadh_" + feature + "_" + feature_distribution_var for feature in feature_groups_features["Fit Free Nadh"] ]
    feature_groups_features["Feature Distribution Fit Free"] += [ "fad_" + feature + "_" + feature_distribution_var for feature in feature_groups_features["Fit Free Fad"]]

def get_feature_name(feature_groups):
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