"""Upload the trained models and the university documents to the GitHub release (they are too big for git).

GitHub does not accept files > 100 MB in a repository, so everything the assistant needs to run without
re-training is packed into zip files (each < 1.9 GB) and attached to the release of tag v1.0:

    models.zip              models/ — all trained models + search index (the assistant needs only this)
    data_processed.zip      data/processed/ — extracted text and chunks (for re-training)
    documents_*.zip         data/raw/ — downloaded official documents and website pages

On another computer:  git clone …  →  0b_download_models.bat  (scripts/download_release.py)  →  3_start_assistant.bat
Not uploaded: _training_files/ (≈9 GB Wikipedia dumps and checkpoints — only for training the LLM/translator,
re-created by 12-16_*.bat), data/app.db (user accounts and chat logs), .env (secrets).

    python scripts/23_upload_release.py [--tag v1.0]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "_release"
PART_LIMIT = 1_800_000_000          # GitHub release assets must be < 2 GiB
STORED = {".pdf", ".zip", ".png", ".jpg", ".jpeg", ".docx", ".xlsx", ".pptx", ".gz", ".bz2", ".npz"}

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def files_under(*paths: Path) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        if p.is_file():
            out.append(p)
        elif p.is_dir():
            out.extend(sorted(f for f in p.rglob("*") if f.is_file() and "__pycache__" not in f.parts))
    return out


def pack(name: str, files: list[Path]) -> list[Path]:
    """Write name.zip (or name_1.zip, name_2.zip … when bigger than PART_LIMIT). Paths inside are relative to ROOT."""
    if not files:
        return []
    groups: list[list[Path]] = [[]]
    size = 0
    for f in files:
        s = f.stat().st_size
        if groups[-1] and size + s > PART_LIMIT:
            groups.append([])
            size = 0
        groups[-1].append(f)
        size += s
    made = []
    for i, g in enumerate(groups, 1):
        zname = f"{name}.zip" if len(groups) == 1 else f"{name}_{i}.zip"
        z = OUT / zname
        with zipfile.ZipFile(z, "w", allowZip64=True) as zf:
            for f in g:
                comp = zipfile.ZIP_STORED if f.suffix.lower() in STORED else zipfile.ZIP_DEFLATED
                zf.write(f, f.relative_to(ROOT).as_posix(), compress_type=comp, compresslevel=1)
        print(f"  {zname}: {len(g)} files, {z.stat().st_size / 1e6:.0f} MB", flush=True)
        made.append(z)
    return made


def gh(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    r = subprocess.run(["gh", *args], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if check and r.returncode != 0:
        raise SystemExit(f"gh {' '.join(args[:3])} … failed:\n{r.stdout}\n{r.stderr}")
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v1.0")
    ap.add_argument("--skip-pack", action="store_true", help="upload the zips already in _release/")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    if not a.skip_pack:
        for old in OUT.glob("*.zip"):
            old.unlink()                 # our own zips from an earlier run of this script
        print("Packing (this takes a few minutes) ...", flush=True)
        pack("models", files_under(ROOT / "models"))
        pack("data_processed", files_under(ROOT / "data" / "processed"))
        raw = ROOT / "data" / "raw"
        if raw.is_dir():
            big = [d for d in sorted(raw.iterdir()) if d.is_dir() and d.name in ("ru", "kk", "en", "web")]
            for d in big:
                pack(f"documents_{d.name}", files_under(d))
            rest = [p for p in sorted(raw.iterdir()) if p not in big]
            pack("documents_other", files_under(*rest))

    zips = sorted(OUT.glob("*.zip"))
    manifest = {z.name: z.stat().st_size for z in zips}
    (OUT / "release_files.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")

    notes = ("Trained models (all trained from scratch by the team) and the official Satbayev University "
             "documents used by the assistant.\n\n"
             "After `git clone`, run `0b_download_models.bat` (or `python scripts/download_release.py`) and then "
             "`3_start_assistant.bat` — no training needed.\n\n"
             "Files: " + ", ".join(f"`{k}` ({v / 1e6:.0f} MB)" for k, v in manifest.items()))
    if gh("release", "view", a.tag, check=False).returncode != 0:
        gh("release", "create", a.tag, "--title", f"{a.tag} — trained models and documents", "--notes", notes)
        print(f"Release {a.tag} created", flush=True)
    else:
        gh("release", "edit", a.tag, "--notes", notes)
    for z in zips + [OUT / "release_files.json"]:
        print(f"  uploading {z.name} ...", flush=True)
        gh("release", "upload", a.tag, str(z), "--clobber")
    url = gh("release", "view", a.tag, "--json", "url", "--jq", ".url").stdout.strip()
    (ROOT / "logs").mkdir(exist_ok=True)
    (ROOT / "logs" / "github_release_url.txt").write_text(url + "\n", encoding="utf-8")
    print(f"\nRELEASE DONE: {url}")


if __name__ == "__main__":
    main()
