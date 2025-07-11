from pathlib import Path
from src.config import get_channel_names, get_feature_extractors

def parse_metadata_file(metadata_df):
    """
    Parse the metadata file and return a dictionary of metadata.
    metadata_df: pandas dataframe of metadata
    returns: 
    - feature_groups_features: dictionary of feature groups and their features that are a subset of the full feature groups features
    """
    metadata_dict = {}
    available_channels = get_channel_names()
    available_feature_extractors = get_feature_extractors(available_channels.values())
    selected_feature_extractors = {}
    for channel_name, feature_extractors in available_feature_extractors.items():
        if any(metadata_df.columns.str.contains(channel_name)):
            selected_feature_extractors[channel_name] = feature_extractors
   
    return metadata_dict