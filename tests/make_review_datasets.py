"""Synthetic tables for exercising the review-table gate.

Each file targets one path through the gate: a role auto-detect rule, an auto-grouping
rule, a profile-matching outcome, or a reader rejection. Regenerate with

    uv run python tests/make_review_datasets.py
    uv run python tests/make_review_datasets.py --report
    uv run python tests/make_review_datasets.py --out ~/Downloads/flim_review_data

The default output directory, tests/review_data/, is gitignored. Seeded random values
keep the generated cases repeatable; --report prints reader and detector results.

The intended sequence for a browser pass is the `pdl1_*` family, in order:
rep1 auto-detects and is saved as a new profile, rep2 is an exact match that must skip
the gate entirely, rep3 and rep4 must land in the chooser with the counts named below.
"""
import argparse
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

SEED = 20260828
RAGGED_LINE = 5988   # line, not row: line 1 is the header
DEFAULT_OUT = Path(__file__).resolve().parent / "review_data"

# filename -> (what it is for, what the gate should do with it)
MANIFEST = {}


def register(name, purpose, expected):
    MANIFEST[name] = (purpose, expected)
    return name


def rng():
    return np.random.default_rng(SEED)


# ---------------------------------------------------------------- the pdl1 family

def _fov_names(n, r, n_fov=25):
    fovs = [f"A{1 + i // 5:02d}_f{1 + i % 5:02d}" for i in range(n_fov)]
    return r.choice(fovs, size=n)


def _lifetime_block(n, r, prefix, t1, t2):
    """Four correlated columns named `{prefix}_{feature}` -- the prefix rule's input."""
    a1 = r.beta(5, 3, size=n)
    return {
        f"{prefix}_t1_mean": r.normal(t1, t1 * 0.08, size=n).round(1),
        f"{prefix}_t2_mean": r.normal(t2, t2 * 0.08, size=n).round(1),
        f"{prefix}_a1_mean": a1.round(4),
        f"{prefix}_tm_mean": (a1 * t1 + (1 - a1) * t2).round(1),
    }


def _morphology_block(n, r):
    area = r.lognormal(7.0, 0.35, size=n).round(1)
    return {
        "Area": area,
        "Perimeter": (2.6 * np.sqrt(np.pi * area)).round(1),
        "Eccentricity": r.beta(2, 3, size=n).round(4),
        "Solidity": r.beta(9, 2, size=n).round(4),
    }


def pdl1_frame(n, r, *, notes=True):
    """The 18-column base shape every pdl1_* file is a variation of."""
    cols = {
        "cell_id": np.arange(1, n + 1),
        "image_name": _fov_names(n, r),
        "treatment": r.choice(["DMSO", "PD-L1"], size=n),
        # "Day 2" before "Day 10" is the natural-sort case
        "day": r.choice(["Day 2", "Day 5", "Day 10"], size=n),
    }
    cols.update(_lifetime_block(n, r, "nadh", 480.0, 2900.0))
    cols.update(_lifetime_block(n, r, "fad", 320.0, 2400.0))
    cols.update(_morphology_block(n, r))
    cols["redox_ratio"] = r.beta(4, 4, size=n).round(4)
    if notes:
        # Repeated free-text notes remain Categorical until the user changes their role.
        cols["notes"] = [f"acquired {1 + i % 28:02d}/03, operator {'AB'[i % 2]}" for i in range(n)]
    return pd.DataFrame(cols)


def build_pdl1_family(out):
    r = rng()

    rep1 = pdl1_frame(1204, r)
    write(out, register(
        "pdl1_rep1.csv",
        "The base shape. Nothing has seen it before.",
        "Auto-detect: cell_id=Row ID, everything textual Categorical, 13 Numerical. "
        "Groups nadh (4) and fad (4) from the prefix rule; Area/Perimeter/Eccentricity/"
        "Solidity/redox_ratio ungrouped. Save as -> 'pdl1'.",
    ), rep1)

    rep2 = pdl1_frame(980, rng())
    write(out, register(
        "pdl1_rep2.csv",
        "Same 18 columns as rep1, different values and row count.",
        "Exact match once pdl1 is saved: 18 shared, 0 missing, 0 new. Must skip the "
        "gate entirely and land on the plot with the summary bar showing pdl1.",
    ), rep2)

    rep3 = rep1.copy()
    rep3["nadh_t3_mean"] = r.normal(6100, 400, size=len(rep3)).round(1)
    rep3["plate"] = r.choice([1, 2, 3], size=len(rep3))
    write(out, register(
        "pdl1_rep3.csv",
        "rep1 plus a third lifetime component and an integer plate number.",
        "Chooser: pdl1 shows 18 shared, 0 missing, 2 new. nadh_t3_mean joins the "
        "existing nadh group through its saved siblings. plate is guessed Numerical "
        "and can be changed to Categorical in the review table.",
    ), rep3)

    rep4 = rep1.drop(columns=["fad_t2_mean", "fad_a1_mean", "notes"]).copy()
    rep4["nadh_a2_mean"] = (1 - rep4["nadh_a1_mean"]).round(4)
    rep4["Circularity"] = r.beta(6, 2, size=len(rep4)).round(4)
    rep4["plate"] = r.choice([1, 2, 3], size=len(rep4))
    rep4["operator"] = r.choice(["AB", "CD"], size=len(rep4))
    rep4["run_date"] = r.choice(["2026-03-01", "2026-03-08"], size=len(rep4))
    write(out, register(
        "pdl1_rep4_partial.csv",
        "rep1 with three columns gone and five added.",
        "Chooser: pdl1 shows 15 shared, 3 missing, 5 new -- ranked below rep3's fit. "
        "The three missing columns get no row in the table. Saving to pdl1 would make "
        "rep1/rep2/rep3 stop matching; this is what 'Save as' is for.",
    ), rep4)

    r2 = rng()
    n = 150
    iris = pd.DataFrame({
        "sepal_length": r2.normal(5.8, 0.8, size=n).round(1),
        "sepal_width": r2.normal(3.0, 0.4, size=n).round(1),
        "petal_length": r2.normal(3.8, 1.7, size=n).round(1),
        "petal_width": r2.normal(1.2, 0.75, size=n).round(1),
        "species": np.repeat(["setosa", "versicolor", "virginica"], n // 3),
    })
    write(out, register(
        "unrelated_iris.csv",
        "A table with nothing in common with pdl1.",
        "No shared columns, so pdl1 is absent from the chooser but remains available "
        "in Manage profiles. No Row ID candidate: measurements have fractional values "
        "and species repeats, so rows are numbered.",
    ), iris)


# ------------------------------------------------------------- identifier edge cases

def build_identifier_cases(out):
    r = rng()
    n = 300

    # Fractional measurements and repeating text provide no identifier candidate.
    write(out, register(
        "no_row_id.csv",
        "No column can serve as an identifier.",
        "No Row ID guessed. resolve_row_id_col invents 'Row number' (strings 1..N), "
        "which must NOT appear in the numerical feature pickers, and hover says 'ID'.",
    ), pd.DataFrame({
        "treatment": r.choice(["DMSO", "PD-L1"], size=n),
        "intensity": r.normal(1000, 120, size=n).round(2),
        "Area": r.lognormal(7, 0.3, size=n).round(1),
    }))

    write(out, register(
        "row_number_taken.csv",
        "A real column already called 'Row number', which cannot itself be the ID.",
        "The invented identifier must be suffixed to 'Row number.1' rather than "
        "overwriting the file's own column.",
    ), pd.DataFrame({
        "Row number": r.choice(["n/a", "tbd", "-"], size=n),
        "treatment": r.choice(["DMSO", "PD-L1"], size=n),
        "intensity": r.normal(1000, 120, size=n).round(2),
    }))

    dup = pd.DataFrame({
        "cell_id": np.concatenate([np.arange(1, n), [1]]),
        "treatment": r.choice(["DMSO", "PD-L1"], size=n),
        "intensity": r.normal(1000, 120, size=n).round(2),
    })
    write(out, register(
        "duplicate_ids.csv",
        "cell_id repeats once, so it is all-but-distinct.",
        "Auto-detect refuses it as Row ID (uniqueness is required) and calls it "
        "Numerical. Assigning Row ID in the review table must block Save with a "
        "duplicate-ID error; the loader must also reject it without dropping rows.",
    ), dup)

    write(out, register(
        "numeric_id_traps.csv",
        "Distinct fractional and integer measurements without a real identifier.",
        "id_float holds halves and remains Numerical. npix is unique and whole, so "
        "it becomes Row ID until corrected in review. See tests/test_auto_detect_roles.py.",
    ), pd.DataFrame({
        "id_float": (np.arange(n) + 0.5),
        "treatment": r.choice(["DMSO", "PD-L1"], size=n),
        "Area": r.lognormal(7, 0.3, size=n).round(1),
        "npix": r.permutation(np.arange(5000, 5000 + n)),
    }))


# ------------------------------------------------------------------ grouping rules

def build_grouping_cases(out):
    r = rng()
    n = 400
    base = {
        "cell_id": np.arange(1, n + 1),
        "image_name": _fov_names(n, r),
        "treatment": r.choice(["DMSO", "PD-L1"], size=n),
    }

    extraction = dict(base)
    for ch in ("ch1", "ch2"):
        for feat in ("T1", "T2", "A1"):
            extraction[f"Lifetime fit_{ch}: {feat}"] = r.normal(1500, 300, size=n).round(1)
        extraction[f"Intensity morphology_{ch}: Area"] = r.lognormal(7, 0.3, size=n).round(1)
    write(out, register(
        "extraction_style.csv",
        "A colleague's Data Extraction output, analysed through the user-table branch.",
        "The shared-prefix rule yields 'Lifetime fit' and 'Intensity morphology', "
        "combining channels at the first underscore. Channel-specific groups can be "
        "assigned in review and saved in a profile.",
    ), pd.DataFrame(extraction))

    flat = dict(base)
    for name in ("Area", "Perimeter", "Solidity", "Length", "Width"):
        flat[name] = r.lognormal(6, 0.3, size=n).round(2)
    write(out, register(
        "flat_names.csv",
        "Measurement names with no separator in them at all.",
        "No group is guessed -- _prefix returns None for every column, so all five "
        "land in Uncategorized Features. The empty-groups starting state.",
    ), pd.DataFrame(flat))

    mixed = dict(base)
    for name in ("n.t1.mean", "n.t2.mean", "f-t1-mean", "f-t2-mean"):
        mixed[name] = r.normal(1500, 300, size=n).round(1)
    for name in ("Cyto: area", "Cyto: perim"):
        mixed[name] = r.lognormal(7, 0.3, size=n).round(1)
    mixed["solo_feature"] = r.normal(1, 0.2, size=n).round(3)
    write(out, register(
        "mixed_separators.csv",
        "Three separator styles in one file, plus one prefix carried by a single column.",
        "Groups n (2) and Cyto (2), from '.' and ': '. The f-t1/f-t2 pair is NOT a "
        "group: a hyphen is deliberately not a separator, so that 'anti-PD1_dose' "
        "groups on 'anti-PD1' rather than on 'anti'. solo_feature is not one either "
        "-- a prefix needs two members. Both fall to Uncategorized Features, which is "
        "what the earliest-separator rule needing no precedence order looks like.",
    ), pd.DataFrame(mixed))


# ----------------------------------------------------------------- content traps

def build_content_cases(out):
    r = rng()
    n = 500

    mostly = r.normal(500, 50, size=n).round(2).astype(object)
    mostly[r.choice(n, size=2, replace=False)] = "n/a"          # 0.4% -> coerces
    half = r.normal(500, 50, size=n).round(2).astype(object)
    half[r.choice(n, size=150, replace=False)] = "below LOD"    # 30% -> stays text

    write(out, register(
        "content_traps.csv",
        "One column of every kind auto-detect has to judge from its values.",
        "blank_col: all empty -> the one column Ignore is guessed for, Preview reads "
        "'empty - will be dropped'. mostly_numeric: 0.4% strings -> coerced, so "
        "Numerical. half_numeric: 30% strings -> Categorical. plate: 3 integer levels "
        "-> guessed Numerical, must be demoted by hand. constant_col: one level, "
        "filterable but useless. notes: 500 distinct -> Categorical and visibly wrong.",
    ), pd.DataFrame({
        "cell_id": np.arange(1, n + 1),
        "treatment": r.choice(["DMSO", "PD-L1"], size=n),
        "plate": r.choice([1, 2, 3], size=n),
        "constant_col": ["batch A"] * n,
        "mostly_numeric": mostly,
        "half_numeric": half,
        "blank_col": [None] * n,
        "notes": [f"free text {i}" for i in range(n)],
        "Area": r.lognormal(7, 0.3, size=n).round(1),
    }))

    write(out, register(
        "all_text.csv",
        "Not one numeric column.",
        "The review table's own validation must disable the save button with 'no "
        "column is Numerical'.",
    ), pd.DataFrame({
        "cell_id": [f"c{i:03d}" for i in range(n)],
        "treatment": r.choice(["DMSO", "PD-L1"], size=n),
        "day": r.choice(["Day 2", "Day 10"], size=n),
    }))

    write(out, register(
        "markup_names.csv",
        "Column names containing characters Plotly and Streamlit read as markup.",
        "'a<b' must survive intact in the review table, the hover template (escaped by "
        "vis/helpers.hover_field) and every reader message (escaped by _as_html). A "
        "name eaten after '<' is the regression this file exists to catch.",
    ), pd.DataFrame({
        "cell_id": np.arange(1, n + 1),
        "a<b": r.normal(10, 1, size=n).round(3),
        "Tau (ns) <ch1>": r.normal(2.4, 0.3, size=n).round(3),
        "treatment & control": r.choice(["DMSO", "PD-L1"], size=n),
    }))

    write(out, register(
        "single_column.csv",
        "One genuine column, no separator anywhere in the file.",
        "The reader accepts a single column. Review marks intensity Numerical with "
        "no Row ID, so analysis adds generated row numbers.",
    ), pd.DataFrame({"intensity": r.normal(1000, 120, size=n).round(2)}))


# ------------------------------------------------------------------ reader paths

def build_reader_cases(out):
    r = rng()
    n = 200
    frame = pd.DataFrame({
        "cell_id": np.arange(1, n + 1),
        "treatment": r.choice(["DMSO", "PD-L1"], size=n),
        "Lifetime fit_ch1: T1": r.normal(480, 40, size=n).round(1),
        "Lifetime fit_ch1: T2": r.normal(2900, 200, size=n).round(1),
    })

    write(out, register(
        "tab_separated.tsv",
        "Plain tab-separated text.",
        "Reads as four columns. The delimiter travels to the exported script rather "
        "than being re-detected there.",
    ), frame, sep="\t")

    write(out, register(
        "pipe_spaces.txt",
        "Pipe-separated, with spaces and a colon inside the header names.",
        "The sniffer trap: pandas' sep=None picks the SPACE here and shreds "
        "'Lifetime fit_ch1: T1' into nonsense columns. The strict rule must pick '|'.",
    ), frame, sep="|")

    decimal = frame.copy()
    for col in ("Lifetime fit_ch1: T1", "Lifetime fit_ch1: T2"):
        decimal[col] = decimal[col].map(lambda v: str(v).replace(".", ","))
    write(out, register(
        "semicolon_decimal.csv",
        "European export: ';' separator and ',' as the decimal mark.",
        "Separator resolves to ';'. The two lifetime columns arrive as text, so they "
        "are guessed Categorical -- and _comma_decimal_hint should say why.",
    ), decimal, sep=";")

    ambiguous = out / register(
        "ambiguous_separators.csv",
        "Every row splits consistently on both ';' and ','.",
        "Two candidates qualify, so the file is rejected as ambiguous rather than "
        "ranked. No tie-break exists, deliberately.",
    )
    lines = ["left;middle,right"] + [f"{i};{i * 2},{i * 3}" for i in range(1, 60)]
    ambiguous.write_text("\n".join(lines) + "\n", encoding="utf-8")

    ragged = out / register(
        "ragged_late.csv",
        "6000 rows, one of them short -- far past the 64 KB sample.",
        f"Must be rejected naming line {RAGGED_LINE}. The sample only ever picks the "
        "separator; "
        "_first_ragged_line streams the whole file to verify it. A pass here would "
        "mean the rule had quietly become 'rectangular for the first 64 KB'.",
    )
    rows = ["cell_id,treatment,intensity"]
    for i in range(1, 6001):
        rows.append(f"{i},DMSO,{1000 + i}" if i + 1 != RAGGED_LINE else f"{i},DMSO")
    ragged.write_text("\n".join(rows) + "\n", encoding="utf-8")

    # Spreadsheets
    numeric_header = frame.rename(columns={"Lifetime fit_ch1: T1": 2024})
    path = out / register(
        "numeric_header.xlsx",
        "A header cell holding the number 2024, and a second sheet.",
        "Two things at once: the sheet warning names 'Notes' as skipped, and the "
        "numeric header must be stringified to '2024' or every df[name] lookup "
        "downstream misses it.",
    )
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        numeric_header.to_excel(writer, sheet_name="Data", index=False)
        pd.DataFrame({"comment": ["ignore this sheet"]}).to_excel(
            writer, sheet_name="Notes", index=False)

    path = out / register(
        "data_on_sheet_two.xlsx",
        "First sheet blank, real table on sheet 2.",
        "Rejected by _diagnose_table, which must name the rule (first sheet only) "
        "rather than let it surface as 'No feature found in the uploaded file'.",
    )
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame().to_excel(writer, sheet_name="Cover", index=False)
        frame.to_excel(writer, sheet_name="Data", index=False)

    path = out / register(
        "opendocument.ods",
        "OpenDocument happy path.",
        "Reads like the xlsx branch; index_col=False must NOT be passed here (it is a "
        "read_csv idiom and would eat the header).",
    )
    frame.to_excel(path, engine="odf", index=False)


def build_pathological_cases(out):
    """Shapes that are legal but awkward -- the ones a gate is most likely to fall over."""
    r = rng()
    n = 120

    dupes = out / register(
        "duplicate_headers.csv",
        "Two columns both called 'T1'.",
        "pandas de-duplicates while parsing, so the frame has 'T1' and 'T1.1' and the "
        "review table shows two rows. The saved profile must record the de-duplicated "
        "names, or re-uploading the same file lands in the chooser forever.",
    )
    lines = ["cell_id,T1,T1"] + [f"{i},{400 + i},{2400 + i}" for i in range(1, n + 1)]
    dupes.write_text("\n".join(lines) + "\n", encoding="utf-8")

    wide = {"cell_id": np.arange(1, 41)}
    for block in range(20):
        for feature in ("t1", "t2", "a1", "tm", "chi2", "int", "sd", "n", "x", "y"):
            wide[f"ch{block:02d}_{feature}"] = r.normal(500, 50, size=40).round(2)
    write(out, register(
        "wide_200_cols.csv",
        "201 columns, 40 rows.",
        "The review table is 201 rows and every one carries a Preview. Watch that the "
        "editor stays usable and that editing a row responds promptly.",
    ), pd.DataFrame(wide))

    write(out, register(
        "one_row.csv",
        "A header and exactly one data row.",
        "cell_id is the first qualifying identifier; Area remains Numerical because "
        "it has a fractional value. Plots of one point must not crash the gate; the "
        "category preview reads '(1 level)'.",
    ), pd.DataFrame({"cell_id": [1], "treatment": ["DMSO"], "Area": [412.5]}))

    write(out, register(
        "all_empty_columns.csv",
        "Headers with nothing underneath any of them.",
        "Never reaches the gate: with no values anywhere, the reader sees a header and "
        "no rows and rejects it by name. The all-empty *column* case is content_traps.csv, "
        "where the rest of the table gives the reader rows to count.",
    ), pd.DataFrame({"cell_id": [None] * 3, "treatment": [None] * 3, "Area": [None] * 3}))

    write(out, register(
        "unicode_names.csv",
        "Greek, superscripts and a very long header.",
        "τ₁ and friends must survive the round trip through analysis_config.toml and back, "
        "and a 120-character header must not break the table's layout.",
    ), pd.DataFrame({
        "细胞编号": np.arange(1, n + 1),
        "τ₁ (ps)": r.normal(480, 40, size=n).round(1),
        "α₁": r.beta(5, 3, size=n).round(4),
        "a very long column name that goes on and on describing exactly how the "
        "measurement was taken and by whom": r.normal(1, 0.1, size=n).round(3),
        "処理": r.choice(["対照", "投薬"], size=n),
    }))


# ---------------------------------------------------------------------- plumbing

def write(out, name, frame, sep=","):
    frame.to_csv(out / name, index=False, sep=sep)


def write_readme(out):
    lines = [
        "# Synthetic tables for the review-table gate",
        "",
        "Regenerated by `tests/make_review_datasets.py` -- edit the script, not these files.",
        "",
    ]
    for name, (purpose, expected) in MANIFEST.items():
        path = out / name
        size = f"{path.stat().st_size / 1024:.0f} KB" if path.exists() else "?"
        lines += [f"## `{name}` ({size})", "", purpose, "", f"**Expected:** {expected}", ""]
    (out / "README.md").write_text("\n".join(lines), encoding="utf-8")


class _Upload(io.BytesIO):
    """The minimum of UploadedFile the reader helpers touch: a name, read and seek."""

    def __init__(self, path):
        super().__init__(path.read_bytes())
        self.name = path.name


def report(out):
    """Report reader diagnostics, roles after the analysis coercion rule, and inferred
    groups.
    """
    from src.column_roles import detect_column_groups
    from src.dataset_io import (
        _first_ragged_line,
        _resolve_delimiter,
        detect_roles,
        resolve_row_id_col,
    )
    from src.profile_matching import compare_columns

    print(f"\n{'file':<26} {'sep':>5} {'cols':>5}  roles                            groups")
    print("-" * 108)
    for name in MANIFEST:
        path = out / name
        if path.suffix in (".xlsx", ".ods"):
            frame = pd.read_excel(path, sheet_name=0)
            kinds = sorted({type(c).__name__ for c in frame.columns})
            print(f"{name:<26} {'xl':>5} {len(frame.columns):>5}  header types: {kinds}")
            continue

        upload = _Upload(path)
        delimiter, viable, present, unusable = _resolve_delimiter(upload)
        upload.seek(0)
        if len(viable) > 1:
            print(f"{name:<26} {'--':>5} {'--':>5}  REJECTED -- ambiguous: "
                  + " ".join(repr(d) for d in viable))
            continue
        if not viable and present:
            print(f"{name:<26} {'--':>5} {'--':>5}  REJECTED -- ragged sample "
                  f"(candidates present: {' '.join(repr(d) for d in present)})")
            continue
        # No candidate present at all is a genuine single-column file, which the
        # reader passes on: how many columns a table needs is get_features' question.
        ragged = _first_ragged_line(upload, delimiter)
        upload.seek(0)
        if ragged:
            print(f"{name:<26} {delimiter!r:>5} {'--':>5}  REJECTED -- ragged at line {ragged[0]}")
            continue

        df = pd.read_csv(path, sep=delimiter, index_col=False, low_memory=False)
        roles = detect_roles(df)
        numeric = [col for col, role in roles.items() if role == "numerical"]
        groups = detect_column_groups(numeric)
        by_role = {}
        for col, role in roles.items():
            by_role.setdefault(role, []).append(col)
        grouped = {}
        for col, group in groups.items():
            grouped.setdefault(group, []).append(col)
        summary = " ".join(f"{role}={len(cols)}" for role, cols in sorted(by_role.items()))
        row_id = by_role.get("row_id", [""])[0]
        if not row_id:
            _framed, row_id = resolve_row_id_col(df.copy(), "")
            row_id += "  (invented)"
        print(f"{name:<26} {delimiter!r:>5} {len(df.columns):>5}  {summary:<32} "
              + (", ".join(f"{g}({len(c)})" for g, c in grouped.items()) or "none"))
        print(f"{'':<26} {'':>5} {'':>5}  id={row_id}")

    # The matching counts the pdl1 family's README claims, against a profile saved
    # from rep1 -- the first step of the browser sequence.
    profile = set(pd.read_csv(out / "pdl1_rep1.csv", nrows=0).columns)
    print(f"\nagainst a profile saved from pdl1_rep1 ({len(profile)} columns):")
    for name in ("pdl1_rep2.csv", "pdl1_rep3.csv", "pdl1_rep4_partial.csv", "unrelated_iris.csv"):
        cols = set(pd.read_csv(out / name, nrows=0).columns)
        fit = compare_columns("pdl1", cols, profile)
        print(f"  {name:<24} {len(fit.shared):>3} shared \u00b7 {len(fit.missing):>2} missing \u00b7 "
              f"{len(fit.new):>2} new   {'EXACT -> skips the gate' if fit.is_exact else 'chooser'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()

    out = args.out.expanduser()
    out.mkdir(parents=True, exist_ok=True)
    build_pdl1_family(out)
    build_identifier_cases(out)
    build_grouping_cases(out)
    build_content_cases(out)
    build_reader_cases(out)
    build_pathological_cases(out)
    write_readme(out)
    print(f"{len(MANIFEST)} tables written to {out}")
    if args.report:
        report(out)


if __name__ == "__main__":
    main()
