"""Parse a workbook in a child process that is stopped at a memory or time cap.

A spreadsheet's cost cannot be told from its size: a 1.6 KB file with one far-away
value makes the reader allocate past 8 GB, and a 17 KB file whose cells share one long
text costs 0.7 GB. The server therefore never opens a workbook itself. No internal
imports: the child starts with pandas and nothing else of the app.
"""
import io
import os
import pickle
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd
import psutil

# launcher.py answers this flag, so the bundled app can start itself as the child.
WORKBOOK_CHILD_FLAG = "--read-workbook"
POLL_SECONDS = 0.02
_ROOT = str(Path(__file__).resolve().parents[1])


def parse_workbook(source):
    """Return (first sheet with stringified headers, every sheet name)."""
    with pd.ExcelFile(source, engine="calamine") as workbook:
        sheet_names = [str(name) for name in workbook.sheet_names]
        df = workbook.parse(sheet_name=0)
    df.columns = [str(col) for col in df.columns]
    return df, sheet_names


def _peak_bytes():
    """This process's peak resident memory."""
    try:
        import resource
    except ImportError:  # Windows
        return psutil.Process().memory_info().peak_wset
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024


def main(input_path, output_path):
    """Child entry point: pickle ("ok", frame, sheet_names, peak) or ("error", text) to output_path."""
    try:
        # From bytes, as the server always read uploads: the format comes from the content.
        frame, sheet_names = parse_workbook(io.BytesIO(Path(input_path).read_bytes()))
        result = ("ok", frame, sheet_names, _peak_bytes())
    except Exception as error:  # noqa: BLE001 - whatever the reader raises goes back as its message
        result = ("error", str(error) or type(error).__name__)
    with open(output_path, "wb") as out:
        pickle.dump(result, out, protocol=pickle.HIGHEST_PROTOCOL)
    return 0


def child_command():
    if getattr(sys, "frozen", False):
        return [sys.executable, WORKBOOK_CHILD_FLAG]
    return [sys.executable, "-m", "src.spreadsheet_child"]


def read_workbook_capped(data, *, memory_limit, time_limit):
    """Parse workbook bytes in a watched child.

    Return ("ok", (frame, sheet_names)), ("error", reader message), or ("memory" | "time" |
    "failed", None). Unpickling the child's result adds no exposure: the child runs the
    same code and parser the server ran in-process before.
    """
    with tempfile.TemporaryDirectory(prefix="fp_workbook_") as folder:
        source, result = os.path.join(folder, "input"), os.path.join(folder, "output")
        with open(source, "wb") as out:
            out.write(data)
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, [_ROOT, os.environ.get("PYTHONPATH")])))
        try:
            child = subprocess.Popen(
                [*child_command(), source, result], cwd=_ROOT, env=env,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except OSError:
            return "failed", None
        started, top = time.monotonic(), 0
        while child.poll() is None:
            try:
                top = max(top, psutil.Process(child.pid).memory_info().rss)
            except psutil.Error:
                pass
            stopped = ("memory" if top > memory_limit
                       else "time" if time.monotonic() - started > time_limit else None)
            if stopped:
                child.kill()
                child.wait()
                return stopped, None
            time.sleep(POLL_SECONDS)
        try:
            with open(result, "rb") as saved:
                outcome = pickle.load(saved)
        except Exception:  # noqa: BLE001 - a missing, cut-off or garbage result is one failure
            outcome = None
    if isinstance(outcome, tuple) and len(outcome) == 2 and outcome[0] == "error":
        return "error", str(outcome[1])
    if isinstance(outcome, tuple) and len(outcome) == 4 and outcome[0] == "ok":
        # The child reports its own peak, so a spike between two polls still counts.
        if max(top, outcome[3]) > memory_limit:
            return "memory", None
        return "ok", (outcome[1], outcome[2])
    # No usable result: a child that died large ran out of memory; otherwise it just failed.
    return ("memory" if top >= memory_limit / 2 else "failed"), None


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:3]))
