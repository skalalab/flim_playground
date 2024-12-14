import pandas as pd

def safe_split_with_logging(base_name):
    try:
        return base_name.rsplit('_', 1)[0]
    except Exception as e:   
        return "missing image name"


def get_features(df):
    error_msg = ""
    numeric_cols = [col for col in df.columns if pd.to_numeric(df[col], errors='coerce').notna().all()]    
    nadh_cols = [c for c in numeric_cols if (c.startswith("n") or c.startswith("redox")) and "mean" in c and "stdev" not in c and "weighted" not in c]
    fad_cols = [c for c in numeric_cols if c.startswith("f") and "mean" in c and "stdev" not in c and "weighted" not in c]
    morphology_cols = [c for c in numeric_cols if not c.startswith("nadh") and not c.startswith("fad") and "mask" not in c and "redox" not in c and "flirr" not in c]
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