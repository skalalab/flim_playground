from pathlib import Path
    
fad_suffix = "f.sdt"
nadh_suffix = "n.sdt"
mask_suffix = ("tif", "tiff")
def list_files_with_suffixes_and_keyword(folder_path, suffixes, keyword=""):
    path = Path(folder_path)
    # rglob searches files recursively
    return [str(file) for file in path.rglob("*") if str(file).endswith(suffixes) and keyword in str(file)]


def sdts_in_dir(folder_path, mask=True):
    """
    Handles single channel sdts. Mutli-channel sdts are not (yet) supported.
    """
    fad_sdt_files = list_files_with_suffixes_and_keyword(folder_path, fad_suffix)
    nadh_sdt_files = list_files_with_suffixes_and_keyword(folder_path, nadh_suffix)
    
    error_msg = ""
    if len(nadh_sdt_files) == 0 and len(fad_sdt_files) == 0:
        error_msg += "no sdt file found! "
        return {}, error_msg
    
    if mask:
        mask_files = list_files_with_suffixes_and_keyword(folder_path, mask_suffix, "mask")
        if len(mask_files) == 0:
            error_msg += "no mask file found! "
            return {}, error_msg
    
    images = {}

    if len(nadh_sdt_files) > 0 :
        for nadh_sdt in nadh_sdt_files:
            # use the image_name as the key 
            image_name = Path(nadh_sdt).name.removesuffix(nadh_suffix) 
            images[image_name] = {}
            images[image_name]["nadh_sdt"] = nadh_sdt
            if mask:
            # find the associated mask
                mask = [path for path in mask_files if Path(path).name.startswith(image_name)]
                try:
                    images[image_name]["mask"] = mask[0]
                except:
                    error_msg += f"no mask found for image {image_name}! "
                    # if no mask found and mask is required, remove it
                    _ = images.pop(image_name)

    if len(fad_sdt_files) > 0:
        for fad_sdt in fad_sdt_files:
            image_name = Path(fad_sdt).name.removesuffix(fad_suffix) 
            if image_name in images:
                images[image_name]["fad_sdt"] = fad_sdt
            else:
                images[image_name] = {}
                images[image_name]["fad_sdt"] = fad_sdt
                if mask: 
                    mask = [path for path in mask_files if Path(path).name.startswith(image_name)]
                    try:
                        images[image_name]["mask"] = mask[0]
                    except:
                        error_msg += f"no mask found for image {image_name}! "
                        # if no mask found and mask is required, remove it
                        _ = images.pop(image_name)
    
    return images, error_msg