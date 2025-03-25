"""
The meta data for the Data Extraction outputted CSV file. Can be extended. If new features are added, modify the lists below.
"""
# customizable categorical columns
required_cols = ["cell_id", "image_name"]
categorical_cols = ["cell_line", "treatment", "day", "experiment", "cell_type", "patient_id"]

# the order of feature groups 
feature_groups_order = ["nadh_fit", "fad_fit", "mask_morphology", "feature_distribution_fit", "fit_free", "feature_distribution_fit_free"]
# customizable numeric column dicionary that store the default names of the features we want to extract
feature_groups_default = {
    "nadh_fit": ["nadh_a", "nadh_b", "nadh_c", "nadh_d", "nadh_e"],
    "fad_fit": ["fad_a", "fad_b", "fad_c", "fad_d", "fad_e"],
    "mask_morphology": ["area", "perimeter", "solidity", "eccentricity", "major_axis_length", "minor_axis_length"],
    "feature_distribution_fit": ["cell_mean", "cell_std", "cell_median"],
    "fit_free": ["phasor_x", "phasor_y"],
    "feature_distribution_fit_free": ["cell_mean", "cell_std", "cell_median"],
}

# customizable numeric column prefixes
feature_groups_prefixes = {
    "nadh_fit": ["nadh", "redox", "na", "nt","ntm", "nint", "normrr"], # put redox in nadh 
    "fad_fit":["fad", "fa", "ft", "ftm", "fint"],
    "mask_morphology": ["mask_morph", "mask_morphology_"],
    "feature_distribution_fit": ["fd_fit_"],
    "fit_free": ["fitFree_"],
    "feature_distribution_fit_free": ["fd_fitFree_"],
}
