"""Put the repo root on sys.path so `import src...` resolves in every test module.

pytest only prepends the test file's own directory, so a module that imports
`src.*` without inserting the root itself is importable only when it happens to
be collected alongside one that does -- it fails when run on its own. Doing it
here once covers every test module regardless of how it is invoked; the existing
per-file `sys.path.insert` calls remain harmless.
"""
import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
