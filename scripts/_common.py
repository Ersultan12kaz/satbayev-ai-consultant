"""Shared paths and helpers for the pipeline scripts."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
EVAL = DATA / "eval"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
LOGS = ROOT / "logs"

for p in (RAW, PROCESSED, MODELS, REPORTS, FIGURES, LOGS):
    p.mkdir(parents=True, exist_ok=True)

# make Windows console print UTF-8 (Kazakh / Russian text)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


class Tee:
    """Duplicate stdout into a log file (logs/<step>.log) so results can be reviewed later."""

    def __init__(self, name: str):
        try:
            self.file = open(LOGS / f"{name}.log", "w", encoding="utf-8")
        except OSError:                   # the log is still held by an older run: write a second log
            self.file = open(LOGS / f"{name}_{time.strftime('%H%M%S')}.log", "w", encoding="utf-8")
        self.stdout = sys.stdout
        sys.stdout = self
        # errors (tracebacks) also go to the log file
        err_self = self

        class _Err:
            def write(self, s):
                sys.__stderr__.write(s)
                err_self.file.write(s)
                err_self.file.flush()

            def flush(self):
                sys.__stderr__.flush()

            def isatty(self):
                return False

            def __getattr__(self, name):
                return getattr(sys.__stderr__, name)

        sys.stderr = _Err()

    def write(self, s):
        self.stdout.write(s)
        self.file.write(s)
        self.file.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def isatty(self):
        return False

    def __getattr__(self, name):          # encoding, fileno, ... -> real stdout
        return getattr(self.stdout, name)


def banner(title: str):
    line = "=" * 78
    print(f"\n{line}\n{title}\n{line}")


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class Timer:
    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *a):
        self.sec = time.time() - self.t0
