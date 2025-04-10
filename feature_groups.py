"""
The meta data for the Data Extraction outputted CSV file. Can be extended. If new features are added, modify the lists below.
"""
# required column: unique cell identifier
required_cols = ["cell_id", "image_name"]

# customizable categorical columns
categorical_cols = [ "experiment", "day", "cell_type", "media", "cell_line", "treatment", "patient_id"]

# customizable numeric column dicionary that store the default names of the features we want to extract (for backward compatibility)
feature_groups_default = {
    "Nadh Fit": ["ntm", "na1", "na2", "nt1", "nt2", "nint", "normrr", "nadh_tau_mean_mean","nadh_a1_mean", "nadh_a2_mean", "nadh_t1_mean", "nadh_t2_mean"],
    "Fad Fit": ["ftm", "fa1", "fa2", "ft1", "ft2", "fint", "fad_tau_mean_mean", "fad_a1_mean", "fad_a2_mean", "fad_t1_mean", "fad_t2_mean"],
    "Mask Morphology": ["area", "perimeter", "solidity", "eccentricity", "major_axis_length", "minor_axis_length"],
    "Feature Distribution Fit": ["cell_mean", "cell_std", "cell_median"],
    "Fit Free": ["phasor_x", "phasor_y"],
    "Feature Distribution Fit Free": ["cell_mean", "cell_std", "cell_median"],
}

# feature groups names
# Later we can do between group correlations
feature_groups = feature_groups_default.keys()

# customizable numeric column prefixes (in Feature Extraction module, it will produce the features with their corresponding feature group prefixes)
feature_groups_prefix = {
    "Nadh Fit": "fit_nadh: ", # put redox in nadh 
    "Fad Fit":"fit_fad: ",
    "Mask Morphology": "mask_morphology: ",
    "Feature Distribution Fit": "fd_fit: ",
    "Fit Free": "fit_free: ",
    "Feature Distribution Fit Free": "fd_fit_free: ",
}
# host all features that are capable of being extracted
feature_groups_features = {
    "Nadh Fit": ["a1", "a2", "t1", "t2", "tm", "intensity", "redox"], # put redox in nadh 
    "Fad Fit":["a1", "a2", "t1", "t2", "tm", "intensity"],
    "Mask Morphology": ["area", "perimeter", "solidity", "eccentricity", "major_axis_length", "minor_axis_length"],
    "Fit Free": ["G(1st)", "S(1st)", "G(2nd)", "S(2nd)"],
}

feature_distribution_vars = ["polarity"]
for feature_distribution_var in feature_distribution_vars:
    feature_groups_features["Feature Distribution Fit"] = [ "nadh_" +  feature  + "_" + feature_distribution_var for feature in feature_groups_features["Nadh Fit"]]
    feature_groups_features["Feature Distribution Fit"] += [ "fad_" +  feature  + "_" + feature_distribution_var for feature in feature_groups_features["Fad Fit"]]
    feature_groups_features["Feature Distribution Fit Free"] = [ feature + "_" + feature_distribution_var for feature in feature_groups_features["Fit Free"] ]

def get_feature_name(feature_groups):
    """
    takes the feature groups and returns the feature groups in human friendly names
    """
    feature_groups_names = {}
    for feature_group in feature_groups:
        if feature_group in feature_groups_prefixes:
            prefix = feature_groups_prefixes[feature_group]
            feature_groups_names[feature_group] = [prefix + feature for feature in feature_groups_features[feature_group]]
        else:
            feature_groups_names[feature_group] = feature_groups_features[feature_group]
    return feature_groups_names