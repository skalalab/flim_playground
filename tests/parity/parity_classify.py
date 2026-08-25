"""Classification parity on the real inhibitors dataset.

The classification pipeline is shared wholesale with the export via
_extract_source(), so this is a regression check that the shared source keeps
producing identical splits, fits and predictions.

Run:  uv run python tests/parity/parity_classify.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from harness_common import (
    EXAMPLES,
    WORK_ROOT,
    Results,
    base_state,
    load_app_df,
    run_export,
)
from harness_widgets import patch_streamlit

CATS = ["cell_line", "treatment", "dish"]
CSV = EXAMPLES / "inhibitors.csv"
WORK = WORK_ROOT / "classify"
FEATS = ["Lifetime fit_nadh: t1", "Lifetime fit_nadh: t2", "Lifetime fit_nadh: a1",
         "Lifetime fit_nadh: tm", "Intensity morphology_nadh: area",
         "Intensity texture_nadh: intensity_sum"]

R = Results()


def run(method, threshold_method="None", sampling="None", class_weight="None"):
    print(f"\n=== Classification {method} (threshold={threshold_method}) — inhibitors.csv ===")
    patch_streamlit()
    from src.classify import run_classification

    df, _ = load_app_df(CSV, CATS, "cell_id", "image_name")
    df = df.dropna(subset=FEATS)
    df["classes"] = df[["cell_line"]].astype(str).agg("_".join, axis=1)
    classes = sorted(df["classes"].unique())

    err, app_res = run_classification(df[FEATS + ["classes"]], method, 0.7, sampling,
                                      class_weight, threshold_method,
                                      classifier_params={}, random_state=42)
    assert not err, err

    state = base_state("Classification", "inhibitors.csv", CATS,
                       analysis_columns=list(df.columns.drop("classes")),
                       method_params={"selected_features": FEATS,
                                      "classification_method": method,
                                      "splits": 0.7, "sampling_method": sampling,
                                      "class_weight": class_weight,
                                      "threshold_method": threshold_method,
                                      "classifier_params": {},
                                      "classify_by": ["cell_line"],
                                      "classify_classes": classes})
    ns, _ = run_export(state, CSV, WORK / f"{method}_{threshold_method}".replace(" ", "_"))

    app_m, exp_m = app_res["metrics"], ns["metrics"]
    R.check("accuracy", np.isclose(app_m["accuracy"], exp_m["accuracy"]),
            f"app={app_m['accuracy']:.6f} exp={exp_m['accuracy']:.6f}")
    R.check("balanced accuracy",
            np.isclose(app_m["balanced_accuracy"], exp_m["balanced_accuracy"]),
            f"app={app_m['balanced_accuracy']:.6f} exp={exp_m['balanced_accuracy']:.6f}")
    R.check(f"identical predictions ({len(ns['y_pred'])} test rows)",
            (np.asarray(app_res["y_pred"]) == np.asarray(ns["y_pred"])).all())
    R.check("per-class metrics",
            app_m["per_class"].keys() == exp_m["per_class"].keys()
            and all(np.isclose(app_m["per_class"][c]["f1_score"],
                               exp_m["per_class"][c]["f1_score"])
                    for c in app_m["per_class"]))


def main():
    run("Random Forest")
    run("Logistic Regression", threshold_method="Balanced Accuracy")
    return 0 if R.summary("Classification") else 1


if __name__ == "__main__":
    sys.exit(main())
