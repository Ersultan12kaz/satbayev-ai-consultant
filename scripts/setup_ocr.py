"""Download Tesseract OCR language models (Russian, Kazakh, English) into <project>/tessdata."""
import sys
from pathlib import Path

import requests

from _common import ROOT

URL = "https://github.com/tesseract-ocr/tessdata/raw/main/{}.traineddata"
TESSDATA = ROOT / "tessdata"


def main():
    TESSDATA.mkdir(exist_ok=True)
    for lang in ("rus", "kaz", "eng", "osd"):
        target = TESSDATA / f"{lang}.traineddata"
        if target.exists() and target.stat().st_size > 100_000:
            print(f"  {lang}: already downloaded")
            continue
        print(f"  downloading {lang}.traineddata ...", flush=True)
        r = requests.get(URL.format(lang), timeout=600)
        r.raise_for_status()
        target.write_bytes(r.content)
        print(f"  {lang}: {len(r.content) / 1e6:.1f} MB")
    sys.path.insert(0, str(ROOT))
    from ml.ocr import available_langs, tesseract_cmd
    print("  tesseract:", tesseract_cmd())
    print("  languages:", sorted(available_langs()))


if __name__ == "__main__":
    main()
