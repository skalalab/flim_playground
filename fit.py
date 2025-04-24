import numpy as np
import tifffile
from sdt_io import read_sdt150

def choose_shift(metadata_df, duration, time_bins, num_components, fitting_algo, analysis_type, channel):
    error_msg = ""
    for i, row in metadata_df.iterrows():
        image_name = row['image_name']
        if analysis_type == "ROI Summing Fit":
          
            irf_path = row.get('irf', None)
        try:
            irf = np.loadtxt(irf_path)
        except Exception as e:
            error_msg = f"Error reading the IRF file for image {image_name}: {e}"
            return error_msg, None
        if len(irf) != time_bins:
            error_msg = f"IRF length mismatch with specified time bins. IRF length: {len(irf)}, time bins: {time_bins}."
            return error_msg, None
        if analysis_type == "ROI Summing Fit":
            mask_path = row.get('mask', None)
            try: 
                mask = tifffile.imread(mask_path)
            except Exception as e:
                error_msg = f"Error reading the mask file for image {image_name}: {e}"
                return error_msg, None
            nadh_decay = row.get('nadh decay', None)
        

        