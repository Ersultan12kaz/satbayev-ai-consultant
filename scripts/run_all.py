"""Run the whole pipeline: download -> extract -> chunk -> train 4 models -> index -> evaluate.

    python scripts/run_all.py              # everything
    python scripts/run_all.py --skip-download
    python scripts/run_all.py --from 4     # start from step 4
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = [
    (1, "01_download_documents.py", "Download official documents"),
    (1, "01b_download_website.py", "Download pages of the university websites"),
    (2, "02_extract_text.py", "Extract + clean text"),
    (3, "01c_extra_sources.py", "Other public sources (encyclopedia, gov.kz) + read them"),
    (3, "03_chunking.py", "Chunking"),
    (4, "04_train_language_detector.py", "Train language detector"),
    (5, "05_train_scope_classifier.py", "Train scope classifier"),
    (6, "06_train_embeddings.py", "Train embedding model"),
    (7, "07_build_index.py", "Build vector + BM25 index"),
    (7, "07b_extract_facts.py", "Extract facts (rector, pro-rectors) from signature blocks"),
    (8, "08_train_reranker.py", "Train reranker"),
    (9, "09_evaluate.py", "Evaluate on the test questions (4 experiments)"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--from", dest="start", type=int, default=1)
    ap.add_argument("--to", dest="end", type=int, default=9)
    args = ap.parse_args()
    t_all = time.time()
    status_file = HERE.parent / "logs" / "run_all_status.json"
    status_file.parent.mkdir(exist_ok=True)
    status = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "steps": {}, "finished": False}

    def dump():
        status_file.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")

    for n, script, title in STEPS:
        if n < args.start or n > args.end or (n == 1 and args.skip_download):
            continue
        print(f"\n########## [{n}/9] {title} ##########", flush=True)
        t0 = time.time()
        status["steps"][script[:-3]] = {"step": n, "state": "running"}
        dump()
        r = subprocess.run([sys.executable, str(HERE / script)], cwd=str(HERE.parent), stderr=subprocess.PIPE,
                           text=True, encoding="utf-8", errors="replace")
        if r.stderr:                       # warnings / traceback: show them and keep them in logs/
            print(r.stderr, flush=True)
            (status_file.parent / f"{script[:-3]}_errors.log").write_text(r.stderr, encoding="utf-8")
        status["steps"][script[:-3]].update(state="ok" if r.returncode == 0 else "failed",
                                        seconds=round(time.time() - t0, 1))
        dump()
        if r.returncode != 0 and script.startswith("01c"):
            print("  other public sources could not be added (no internet?) - continuing without them", flush=True)
            continue
        if r.returncode != 0:
            print(f"\nSTEP {n} FAILED ({script}). See logs/ for details.")
            sys.exit(r.returncode)
        print(f"########## step {n} done in {time.time() - t0:.1f}s ##########", flush=True)
    status["finished"] = True
    dump()
    print(f"\nALL DONE in {(time.time() - t_all) / 60:.1f} min. Start the assistant: "
          f"python -m uvicorn backend.app.main:app  ->  http://127.0.0.1:8000")


if __name__ == "__main__":
    main()
