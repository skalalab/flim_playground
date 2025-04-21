from pathlib import Path
import streamlit as st   
import os
file_suffix_default = {
    'mask': '_mask.tiff',
    'nadh decay': 'n.sdt',
    'fad decay': 'f.sdt',
    'single cell features': '.csv',
    'nadh histogram': '_ch1.csv',
    'red histogram': '_ch2.csv',
    "nadh shift": "_n_shift.asc",
    "fad shift": "_f_shift.asc",
    'irf': '.txt',
}

spc_output_suffix = {
    "a1[%]": "_a1[%].asc",
    "a2[%]": "_a2[%].asc",
    "t1": "_t1.asc",
    "t2": "_t2.asc",
    "shift": "_shift.asc",
}

def list_files_with_suffix(folder_path, suffix):
    path = Path(folder_path)
    # rglob searches files recursively
    return [str(file) for file in path.rglob("*") if file.name.endswith(suffix)]

def list_files_with_filename(folder_path, filename):
    path = Path(folder_path)
    # rglob searches files recursively
    return [str(file) for file in path.rglob("*") if file.name == filename]


# def sdts_in_dir(folder_path, mask=True)
#     """
#     Handles single channel sdts. Mutli-channel sdts are not (yet) supported.
#     mask: bool, if True, it requires for mask files in the same folder
#     """
#     has_fad = has_nadh = False
#     fad_sdt_files = list_files_with_suffixes_and_keyword(folder_path, fad_suffix)
#     nadh_sdt_files = list_files_with_suffixes_and_keyword(folder_path, nadh_suffix)
    
#     error_msg = ""
#     if len(nadh_sdt_files) == 0 and len(fad_sdt_files) == 0:
#         error_msg += "no sdt file found! "
#         return {}, error_msg, has_nadh, has_fad
    
#     if mask:
#         mask_files = list_files_with_suffixes_and_keyword(folder_path, mask_suffix, ["mask", "cellpose"])
#         if len(mask_files) == 0:
#             error_msg += "no mask file found! "
#             return {}, error_msg, has_nadh, has_fad 
    
#     images = {}
   
#     if len(nadh_sdt_files) > 0:
#         for nadh_sdt in nadh_sdt_files:
#             # use the image_name as the key 
#             image_name = Path(nadh_sdt).name.removesuffix(nadh_suffix)           
#             if mask:
#             # find the associated mask
#                 mask = [path for path in mask_files if Path(path).name.startswith(image_name)]
#                 try:
#                     mask = mask[0]
#                     images[image_name] = {}
#                     images[image_name]["mask"] = mask
#                     images[image_name]["nadh_sdt"] = nadh_sdt
#                     has_nadh = True
#                 except:
#                     error_msg += f"no mask found for image {image_name}! "


#     if len(fad_sdt_files) > 0:
#         has_fad = True
#         for fad_sdt in fad_sdt_files:
#             image_name = Path(fad_sdt).name.removesuffix(fad_suffix) 
#             if image_name in images:
#                 images[image_name]["fad_sdt"] = fad_sdt
#             else:
#                 if mask: 
#                     mask = [path for path in mask_files if Path(path).name.startswith(image_name)]
#                     try:
#                         mask = mask[0]
#                         images[image_name] = {}
#                         images[image_name]["fad_sdt"] = fad_sdt
#                         images[image_name]["mask"] = mask
#                         has_fad = True
#                     except:
#                         error_msg += f"no mask found for image {image_name}! "
                     
#     return images, error_msg, has_nadh, has_fad

# def sdt_folder_check(folder_path, irf_check=False):
#     """
#     Check if the folder contains sdts and masks.
#     check_irf: bool, if True, it requires for irf files in the same folder
#     """
#     upload_complete = False
#     selected_channel = st.selectbox("Select NADH or FAD", ["NADH", "FAD"])
#     if folder_path and st.button("List Files & Run"):
#         if os.path.isdir(folder_path):
#             images, error_msg, has_nadh, has_fad = sdts_in_dir(folder_path)
#             if len(images) > 0:
#                 upload_complete = True
#             if error_msg != "":
#                 st.markdown(f"<h5 style='text-align: center; color: red'>{error_msg}</h5>", unsafe_allow_html=True)
#             if not has_nadh and selected_channel == "NADH":
#                 st.markdown(f"<h5 style='text-align: center; color: red'>No NADH sdts found in the folder.</h5>", unsafe_allow_html=True)
#                 upload_complete = False
#             if not has_fad and selected_channel== "FAD":
#                 st.markdown(f"<h5 style='text-align: center; color: red'>No FAD sdts found in the folder.</h5>", unsafe_allow_html=True)
#                 upload_complete = False

#             if upload_complete is False: 
#                 st.markdown(f"<h7 style='text-align: center;'>See error msgs. No sdt found or no mask associated with sdts found. \
#                             It looks for {nadh_suffix} suffix for nadh sdts, {fad_suffix} for fad sdts, and {mask_suffix} suffix \
#                             and 'mask' keyword for mask files. </h7>", unsafe_allow_html=True)
                
#             if irf_check:
#                 irf_file = list_files_with_suffixes_and_keyword(folder_path, irf_suffix, keywords=["irf"])

#                 try:
#                     irf_file = irf_file[0]
#                     with open(irf_file, "r") as f:
#                         irf = f.readlines()
#                     irf_array = []
#                     for line in irf:
#                         line = line.strip() # Remove any whitespace or newline characters
#                         if line:  # Check if the line is not empty
#                             try:
#                                 irf_array.append(int(line))  # Convert to float (or int if preferred)
#                             except ValueError:
#                                 upload_complete = False
#                                 st.markdown(f"<h5 style='text-align: center; color: red'>IRF file should contains numbers only.</h5>", unsafe_allow_html=True)
#                     images["original_irf"] = irf_array
#                 except:
#                     st.markdown(f"<h5 style='text-align: center; color: red'>No IRF file found in the folder.</h5>", unsafe_allow_html=True)
#                     upload_complete = False
#                     return {}, selected_channel, upload_complete
                
#         else:
#             st.markdown("***Warning: The provided path is not a directory or doesn't exist.***")
#             st.markdown("<h7 style='text-align: center; color: red;'>Note: this tool only works ***offline***, as the online app does not have access to your files.</h7>", unsafe_allow_html=True)
#             return {}, selected_channel, upload_complete
        
#         return images, selected_channel, upload_complete
#     return {}, selected_channel, upload_complete

# def get_sdts(folder_path, mask=True):
#     """
#     Check if the folder contains sdts.
#     """
#     error_msg = ""  
#     if os.path.isdir(folder_path):
#         sdt_files = list_files_with_suffixes_and_keyword(folder_path, sdt_suffix)
#         if len(sdt_files) == 0:
#             error_msg += "no sdt file found! "
#             return [], [], error_msg
#         if mask:
#             mask_files = list_files_with_suffixes_and_keyword(folder_path, mask_suffix, ["mask", "cellpose"])
#             if len(mask_files) == 0:
#                 error_msg += "No mask file found! "
#                 return [], [], error_msg
#             # now we have both sdt and mask files
#             # we need to align them
#             images = {}
#             for sdt in sdt_files:
#                 image_name = Path(sdt).name.removesuffix(sdt_suffix)
#                 mask = [path for path in mask_files if Path(path).name.startswith(image_name)]
#                 try:
#                     mask = mask[0]
#                     images[image_name] = {}
#                     images[image_name]["sdt"] = sdt
#                     images[image_name]["mask"] = mask
#                 except:
#                     error_msg += f"no mask found for image {image_name}! "
#             return [], images, error_msg
#         else:
#             return sdt_files, [], error_msg

#     else: 
#         error_msg += "The provided path is not a directory or doesn't exist. "
#         return [], [], error_msg   
