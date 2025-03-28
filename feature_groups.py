"""
The meta data for the Data Extraction outputted CSV file. Can be extended. If new features are added, modify the lists below.
"""
# required column: unique cell identifier
required_cols = ["cell_id", "image_name"]

# customizable categorical columns
categorical_cols = [ "experiment", "day", "cell_type", "cell_line", "treatment", "patient_id"]

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
feature_groups_prefixes = {
    "Nadh Fit": ["fit_nadh"], # put redox in nadh 
    "Fad Fit":["fit_fad"],
    "Mask Morphology": ["mask_morphology"],
    "Feature Distribution Fit": ["fd_fit"],
    "Fit Free": ["fitFree"],
    "Feature Distribution Fit Free": ["fd_fitFree"],
}
