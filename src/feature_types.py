"""
Numerical and Categorical Features for Data Extraction outputted CSV file. If new features are added, modify the lists below.
"""

from src.config import get_unique_cell_id_col, get_fov_name_col

unique_cell_id_col = get_unique_cell_id_col()
fov_name_col = get_fov_name_col()
required_cols = [unique_cell_id_col, fov_name_col]

# customizable categorical columns
categorical_cols = [ "experiment", "day", "hour", "cell_type", "media", "dish", "cell_line", "treatment", "condition", "patient_id", "replicate", "GMM_group", "2D_GMM_group"]


# Numerical Feature
# # group by channel_name_feature_extractor

# all_feature_extractors = get_all_feature_extractors()
# # get all possible combinations of channel_name_feature_extractor
# all_numerical_feature_groups = [feature_extractor for feature_extractor in all_feature_extractors.keys()]