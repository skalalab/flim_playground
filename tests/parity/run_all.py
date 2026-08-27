"""Run every app-vs-export parity harness and report one verdict.

Run:  uv run python tests/parity/run_all.py

Exits non-zero if any check fails. Known, documented gaps are reported but do not
fail the run — see README.md.
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = ["parity_phasor.py", "parity_methods.py", "parity_classify.py",
           "parity_controls.py"]


def _src_mtimes():
    return {path: path.stat().st_mtime_ns
            for path in (HERE.parents[1] / "src").rglob("*.py")}


def main():
    # inspect.getsource() re-reads the file through linecache but indexes it with
    # the already-imported code object's line numbers, so a source edit landing
    # mid-run splices generated scripts out of the wrong functions. That produced
    # a NameError that vanished on the next run and looked like flakiness.
    before = _src_mtimes()
    failures = []
    for name in SCRIPTS:
        print(f"\n{'=' * 70}\n  {name}\n{'=' * 70}")
        proc = subprocess.run([sys.executable, str(HERE / name)], check=False)
        if proc.returncode != 0:
            failures.append(name)

    print(f"\n{'=' * 70}")
    if _src_mtimes() != before:
        print("PARITY INCONCLUSIVE: src/ changed while the harnesses were running, so "
              "generated scripts may have been built from stale line numbers. Re-run on "
              "a quiet tree.")
        return 1
    if failures:
        print(f"PARITY FAILED in: {', '.join(failures)}")
        return 1
    print("ALL PARITY HARNESSES PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
