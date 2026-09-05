"""Exercise reading, role review, interpretation, and profile round trips for tables.

After saving a working copy, its profile must exactly match the same file on reload.
Run this standalone harness after generating its default datasets:
    uv run python tests/make_review_datasets.py
    uv run python tests/stress_review_datasets.py
"""
import argparse
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.column_roles import ROLE_LABELS  # noqa: E402
from src.dataset_io import (  # noqa: E402
    build_working_copy,
    interpret_table,
    read_table,
    review_blocking_reason,
)
from src.profile_matching import compare_columns  # noqa: E402
from src.widgets.analysis_config_widgets import (  # noqa: E402
    apply_column_groups,
    apply_column_roles,
    profile_known_columns,
    working_copy_arguments,
)

DATA = Path(__file__).resolve().parent / "review_data"
# What the reader accepts, so a sweep of a real folder skips its README and .DS_Store
# rather than reporting them as rejections the app would never have been asked for.
SWEEPABLE = {".csv", ".tsv", ".txt", ".xlsx", ".xlsm", ".ods"}


def upload(path):
    """Use UploadedFile so Streamlit's cache hashes upload contents instead of a file path."""
    from streamlit.runtime.uploaded_file_manager import UploadedFile, UploadedFileRec

    return UploadedFile(UploadedFileRec("id", path.name, "text/csv", path.read_bytes()), None)


def one(path):
    """Returns (verdict, details, failures) for a single file."""
    failures = []
    df, _meta, _delim, scope_warning, error = read_table(upload(path))
    if error:
        return "REJECTED", error.strip().splitlines()[0], failures

    file_cols = set(df.columns)
    roles, groups, _numeric = build_working_copy(df)
    blocked = review_blocking_reason(df, roles)
    counts = {}
    for role in roles.values():
        counts[ROLE_LABELS[role]] = counts.get(ROLE_LABELS[role], 0) + 1
    summary = " ".join(f"{label}={n}" for label, n in sorted(counts.items()))

    # The round trip: what a Save writes has to describe the file it was saved from.
    profile_cfg = {}
    apply_column_roles(profile_cfg, roles)
    apply_column_groups(profile_cfg, groups, group_names=list(dict.fromkeys(groups.values())))
    fit = compare_columns("saved", file_cols, profile_known_columns(profile_cfg))
    if not fit.is_exact:
        failures.append(f"save/re-upload is not an exact match: "
                        f"missing={list(fit.missing)} new={list(fit.new)}")

    if blocked:
        return "BLOCKED", f"{summary} | {blocked.replace(chr(10), ' ')[:96]}", failures

    args = working_copy_arguments(roles, groups)
    # User tables have no designated FOV role, matching the page's blank FOV argument.
    frame, feature_groups, ok, row_id = interpret_table(
        df.copy(), args["categorical_cols"], args["unique_row_id_col"],
        "", ignored_cols=args["ignored_cols"],
        feature_groups=args["feature_groups"], scope_warning=scope_warning,
        use_data_extraction=False)
    if not ok:
        return "NO PLOT", f"{summary} | interpret_table refused it", failures

    # Everything the pickers offer must be a real column of the analysis frame, and the
    # identifier must never be among them.
    offered = [col for cols in feature_groups.values() for col in cols]
    for col in offered:
        if col not in frame.columns:
            failures.append(f"picker offers '{col}', which is not in the frame")
    if row_id in offered:
        failures.append(f"the identifier '{row_id}' is offered as a measurement")
    for col in args["ignored_cols"]:
        if col in offered:
            failures.append(f"ignored column '{col}' is offered as a measurement")

    detail = (f"{summary} | id={row_id} | "
              f"groups={', '.join(f'{g}({len(c)})' for g, c in feature_groups.items())}")
    return "PLOTS", detail, failures


def main():
    # Use --dir for external datasets; the default synthetic set has documented expectations.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=DATA,
                        help="directory of tables to sweep (default: tests/review_data)")
    data = parser.parse_args().dir
    paths = sorted(p for p in data.iterdir()
                   if p.suffix.lower() in SWEEPABLE and not p.name.startswith("~$"))
    if not paths:
        sys.exit(f"No tables in {data}. Run: uv run python tests/make_review_datasets.py")
    problems = []
    print(f"{'file':<34} {'verdict':<9} details")
    print("-" * 118)
    for path in paths:
        try:
            verdict, detail, failures = one(path)
        except Exception:                                            # noqa: BLE001
            verdict, detail, failures = "CRASH", "", ["\n" + traceback.format_exc()]
        print(f"{path.name:<34} {verdict:<9} {detail}")
        for failure in failures:
            print(f"{'':<34} {'':<9} !! {failure}")
            problems.append((path.name, failure))
    print()
    if problems:
        print(f"{len(problems)} problem(s) across {len({n for n, _ in problems})} file(s)")
        sys.exit(1)
    print(f"{len(paths)} files, no problems")


if __name__ == "__main__":
    main()
