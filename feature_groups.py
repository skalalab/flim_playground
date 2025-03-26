"""
The meta data for the Data Extraction outputted CSV file. Can be extended. If new features are added, modify the lists below.
"""
# customizable categorical columns
required_cols = ["cell_id", "image_name"]
categorical_cols = [ "experiment", "day", "cell_type", "cell_line", "treatment", "patient_id"]

# customizable numeric column dicionary that store the default names of the features we want to extract
feature_groups_default = {
    "Nadh Fit": ["nadh_a", "nadh_b", "nadh_c", "nadh_d", "nadh_e"],
    "Fad Fit": ["fad_a", "fad_b", "fad_c", "fad_d", "fad_e"],
    "Mask Morphology": ["area", "perimeter", "solidity", "eccentricity", "major_axis_length", "minor_axis_length"],
    "Feature Distribution Fit": ["cell_mean", "cell_std", "cell_median"],
    "Fit Free": ["phasor_x", "phasor_y"],
    "Feature Distribution Fit Free": ["cell_mean", "cell_std", "cell_median"],
}

# the order of feature groups 
feature_groups = feature_groups_default.keys()

# customizable numeric column prefixes
feature_groups_prefixes = {
    "Nadh Fit": ["nadh", "redox", "na", "nt","ntm", "nint", "normrr"], # put redox in nadh 
    "Fad Fit":["fad", "fa", "ft", "ftm", "fint"],
    "Mask Morphology": ["mask_morph", "mask_morphology_"],
    "Feature Distribution Fit": ["fd_fit_"],
    "Fit Free": ["fitFree_"],
    "Feature Distribution Fit Free": ["fd_fitFree_"],
}
