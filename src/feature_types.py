"""
Numerical and Categorical Features for Data Extraction outputted CSV file. If new features are added, modify the lists below.
"""

from src.config import get_unique_cell_id_col, get_image_name_col, get_all_channel_names, get_all_feature_extractors

unique_cell_id_col = get_unique_cell_id_col()
image_name_col = get_image_name_col()
required_cols = [unique_cell_id_col, image_name_col]

# customizable categorical columns
categorical_cols = [ "experiment", "day", "hour", "cell_type", "media", "dish", "cell_line", "treatment", "condition", "patient_id", "replicate", "GMM_group", "2D_GMM_group"]


# Numerical Feature
# group by channel_name_feature_extractor
all_channel_names = get_all_channel_names()
all_feature_extractors = get_all_feature_extractors()
# get all possible combinations of channel_name_feature_extractor
all_numerical_feature_groups = [channel_name + "_" + feature_extractor for channel_name in all_channel_names for feature_extractor in all_feature_extractors]