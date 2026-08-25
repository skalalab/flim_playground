"""Contract for donor-batch interleaving.

Emitting one trace per donor paints each donor entirely over the previous one, so the
last donor in every group is systematically the most visible. Batching each donor and
cycling through them removes that bias while keeping one legend entry per donor.
"""
import sys
import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from src.vis.helpers import interleave_point_batches

FAILS = []
def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + ("" if cond else f"   {detail}"))
    if not cond:
        FAILS.append(name)

levels = {"P1": np.arange(0, 60), "P2": np.arange(60, 120), "P3": np.arange(120, 165)}
out = interleave_point_batches(levels)

check("every point is emitted exactly once",
      sorted(np.concatenate([idx for _l, idx in out])) == list(range(165)))
check("no batch is empty", all(len(idx) > 0 for _l, idx in out))
check("a batch only holds points of its own level",
      all(set(idx) <= set(levels[l]) for l, idx in out))

order = [l for l, _idx in out]
first_seen = {l: order.index(l) for l in levels}
second = {}
for pos, l in enumerate(order):
    if l in second:
        continue
    if pos > first_seen[l]:
        second[l] = pos
check("every level starts before any level takes a second turn",
      max(first_seen.values()) < min(second.values()), (first_seen, second))
check("no level runs two batches back to back while others remain",
      all(order[i] != order[i + 1] for i in range(len(order) - 1)
          if len(set(order[i:])) > 1), order[:12])

check("deterministic", [(l, list(i)) for l, i in interleave_point_batches(levels)]
                      == [(l, list(i)) for l, i in out])
check("batches are sampled across the level, not contiguous blocks",
      any(max(np.diff(sorted(idx))) > 1 for l, idx in out if l == "P1" and len(idx) > 2))

small = interleave_point_batches({"A": np.arange(3), "B": np.arange(3, 6)})
check("a level too small to batch still emits once each",
      len(small) == 2 and sorted(np.concatenate([i for _l, i in small])) == list(range(6)), small)
check("batches hold at least ~5 points where possible",
      all(len(idx) >= 5 for _l, idx in interleave_point_batches({"A": np.arange(40)})),
      [len(i) for _l, i in interleave_point_batches({"A": np.arange(40)})])
check("one level is a no-op ordering", len({l for l, _ in interleave_point_batches({"A": np.arange(100)})}) == 1)
check("empty input gives no batches", interleave_point_batches({}) == [])
check("a level with no points is skipped",
      [l for l, _ in interleave_point_batches({"A": np.arange(10), "B": np.array([], dtype=int)})] .count("B") == 0)

print("\n" + ("ALL PASS" if not FAILS else f"{len(FAILS)} FAILING: {FAILS}"))
sys.exit(1 if FAILS else 0)
