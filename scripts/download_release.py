"""Download the trained models and documents from the GitHub release and unpack them into the project.

    python scripts/download_release.py            # everything (models + documents)
    python scripts/download_release.py --models   # only models.zip (enough to run the assistant)
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = "Ersultan12kaz/satbayev-ai-consultant"

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def fetch(url: str, dest: Path) -> None:
    with urllib.request.urlopen(url) as r, open(dest, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {dest.name}: {done / 1e6:.0f} / {total / 1e6:.0f} MB", end="", flush=True)
    print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v1.0")
    ap.add_argument("--models", action="store_true", help="download only models.zip")
    a = ap.parse_args()
    base = f"https://github.com/{REPO}/releases/download/{a.tag}/"
    tmp = ROOT / "_release"
    tmp.mkdir(exist_ok=True)
    listing = tmp / "release_files.json"
    fetch(base + "release_files.json", listing)
    names = list(json.loads(listing.read_text(encoding="utf-8")))
    if a.models:
        names = [n for n in names if n.startswith("models")]
    for n in names:
        z = tmp / n
        fetch(base + n, z)
        with zipfile.ZipFile(z) as zf:
            zf.extractall(ROOT)
        print(f"  unpacked {n}")
    print("Done. Start the assistant: 3_start_assistant.bat")


if __name__ == "__main__":
    main()
