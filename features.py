import pandas as pd

def safe_split_with_logging(base_name):
    try:
        return base_name.rsplit('_', 1)[0]
    except Exception as e:   
        return "missing image name"



def get_cols(cols, weighted_cols = False):
    nadh_prefixes = ["nadh", "redox", "na", "nt","ntm", "nint", "normrr"] # put redox in nadh 
    fad_prefixes = ["fad", "fa", "ft", "ftm", "fint"]

    nadh_cols = [c for c in cols if any(c.startswith(prefix) for prefix in nadh_prefixes) and "stdev" not in c and (weighted_cols or "weighted" not in c)]
    fad_cols = [c for c in cols if any(c.startswith(prefix) for prefix in fad_prefixes) and "stdev" not in c and (weighted_cols or "weighted" not in c)]
    morphology_cols = [c for c in cols if not any(c.startswith(prefix) for prefix in nadh_prefixes + fad_prefixes) and "mask" not in c and "flirr" not in c and "Unnamed" not in c and "day" not in c and "date" not in c] 
    return nadh_cols, fad_cols, morphology_cols

def get_features(df):
    error_msg = ""
    numeric_cols = [col for col in df.columns if pd.to_numeric(df[col], errors='coerce').notna().all()]    
    nadh_cols, fad_cols, morphology_cols = get_cols(numeric_cols)
    if len(numeric_cols) == 0 or (len(nadh_cols) + len(fad_cols) + len(morphology_cols)) == 0:
        error_msg += "No feature found in the uploaded file."
    
    if "base_name" not in df.columns:
        error_msg += "<br> base_name column is missing in the uploaded file."
    
    return numeric_cols, nadh_cols, fad_cols, morphology_cols, error_msg

def fix_df(df):
    df["base_name"] = df["base_name"].fillna("missing base name")
    if "image_name" not in df.columns:
        df['image_name'] = df['base_name'].apply(safe_split_with_logging)
    else: 
        df["image_name"] = df["image_name"].fillna("missing image name")
    if "treatment" in df.columns:
        df["treatment"] = df["treatment"].fillna("Not Specified")
    else:
        # If no treatment column, create a dummy one
        df["treatment"] = "Not Specified"
    return df