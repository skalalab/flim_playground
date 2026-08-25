"""Contract for the opacity ramp's treatment of missing data.

Opacity is the only ordinal visual channel, so a slot in ``np.linspace`` is a RANK.
"N/A" is the loader's marker for a missing categorical, not a level, and it used to take
a slot: which one depended on how the string happened to sort (last among numeric levels,
so absent data drew at max_opacity -- the most prominent thing on the plot), and taking
one compressed the real levels into what was left.

Nothing else covers this. No golden baseline fixture has "N/A" in an opacity column, so
sina_baseline.json and export_baseline.json stay byte-identical either way and cannot
catch a regression here.
"""
import inspect
import sys

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from src.vis.helpers import create_opacity_mapping, natural_tuple_sort

FAILS = []
def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + ("" if cond else f"   {detail}"))
    if not cond:
        FAILS.append(name)

NA = "N/A"
NA_ALPHA = 0.15
MIN, MAX = 0.3, 1.0

# The four shapes a real opacity column takes. Before the fix each gave N/A a different
# alpha -- 1.00, 0.65, 0.30, 0.77 respectively -- purely from string comparison.
SHAPES = {
    "numeric levels":   ["1", "2", "3"],
    "capitalised words": ["Control", "Treated"],
    "lowercase words":  ["old", "young"],
    "mixed":            ["Day 1", "Day 2", "control"],
}

for name, real in SHAPES.items():
    with_na = create_opacity_mapping(real + [NA])
    without = create_opacity_mapping(real)

    check(f"[{name}] N/A gets the reserved alpha, not a rank",
          with_na[NA] == NA_ALPHA, with_na)
    check(f"[{name}] N/A is fainter than every real level",
          all(with_na[NA] < with_na[level] for level in real), with_na)
    # The load-bearing one: N/A must not consume a slot. Before the fix, three real
    # levels plus N/A spanned 0.30/0.53/0.77 instead of 0.30/0.65/1.00.
    check(f"[{name}] adding N/A leaves every real level's alpha untouched",
          {k: v for k, v in with_na.items() if k != NA} == without, (with_na, without))
    check(f"[{name}] real levels still span the full ramp",
          min(without.values()) == MIN and max(without.values()) == MAX, without)
    check(f"[{name}] real levels rise with natural sort order",
          [k for k, _v in sorted(with_na.items(), key=lambda kv: kv[1]) if k != NA]
          == natural_tuple_sort(real), with_na)
    check(f"[{name}] N/A is the last key, so it ranks last in the legend",
          list(with_na)[-1] == NA, list(with_na))

# One alpha for N/A across every column shape: the arbitrariness is what was wrong.
alphas = {create_opacity_mapping(real + [NA])[NA] for real in SHAPES.values()}
check("N/A's alpha does not depend on the real levels", alphas == {NA_ALPHA}, alphas)

# Edge cases.
check("a lone real level keeps max_opacity, N/A stays reserved",
      create_opacity_mapping(["Control", NA]) == {"Control": MAX, NA: NA_ALPHA},
      create_opacity_mapping(["Control", NA]))
check("an all-N/A column maps only N/A", create_opacity_mapping([NA]) == {NA: NA_ALPHA},
      create_opacity_mapping([NA]))
check("no N/A present is untouched behaviour",
      create_opacity_mapping(["a", "b", "c"]) == dict(zip("abc", np.linspace(MIN, MAX, 3))),
      create_opacity_mapping(["a", "b", "c"]))
check("a substring of N/A is a real level, not the marker",
      create_opacity_mapping(["N/AA", "z"]) == {"N/AA": MIN, "z": MAX},
      create_opacity_mapping(["N/AA", "z"]))

# Export parity: export_script inlines this function's source with _extract_source, which
# copies the signature and defaults but NO surrounding module state. Had na_opacity been a
# module-level constant, every generated script would NameError on it. Compile the
# extracted source against only the names the export actually provides.
namespace = {"np": np, "natural_tuple_sort": natural_tuple_sort}
exec(inspect.getsource(create_opacity_mapping), namespace)
inlined = namespace["create_opacity_mapping"](["1", "2", NA])
check("the source the export inlines needs no module-level state",
      inlined == create_opacity_mapping(["1", "2", NA]), inlined)

print("\n" + ("ALL PASS" if not FAILS else f"{len(FAILS)} FAILING: {FAILS}"))
sys.exit(1 if FAILS else 0)
