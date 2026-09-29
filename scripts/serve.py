"""Start the web assistant (FastAPI + uvicorn) and open the browser when it is ready.
All server output is also written to logs/server.log."""
import threading
import time
import urllib.request
import webbrowser

from _common import ROOT, Tee  # noqa: F401  (sets sys.path, UTF-8 console)

URL = "http://127.0.0.1:8000"


def open_when_ready():
    for _ in range(240):
        try:
            with urllib.request.urlopen(URL + "/api/health", timeout=2) as r:
                if r.status == 200:
                    print(f"\n  >>> Assistant is ready: {URL}   (admin: {URL}/admin.html, API docs: {URL}/docs)\n",
                          flush=True)
                    webbrowser.open(URL)
                    return
        except Exception:
            time.sleep(1)


def main():
    Tee("server")
    print("Starting Satbayev AI Consultant... (first start copies the index into the database, ~1 min)")
    threading.Thread(target=open_when_ready, daemon=True).start()
    import uvicorn
    uvicorn.run("backend.app.main:app", host="127.0.0.1", port=8000, log_level="info")


if __name__ == "__main__":
    main()
