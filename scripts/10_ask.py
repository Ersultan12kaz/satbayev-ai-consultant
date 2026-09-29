"""Ask the trained assistant from the terminal (no web server needed).

    python scripts/10_ask.py "Как оформить академический отпуск?"
    python scripts/10_ask.py            # interactive mode
"""
import sys

from _common import MODELS

from ml.rag import RAGPipeline


def show(res):
    print("\n" + res["text"])
    for s in res["sources"]:
        print(f"   [{s['n']}] {s['title']} — p. {s['page']}  {s['url']}")
    print(f"   language={res['lang']}  status={res['status']}  confidence={res['confidence']} "
          f"({res['confidence_score']})  time={res['timings']['total_ms']:.0f} ms")


def main():
    pipe = RAGPipeline.load(MODELS)
    if len(sys.argv) > 1:
        show(pipe.answer(" ".join(sys.argv[1:])))
        return
    print("Satbayev AI Consultant — type a question (RU / KZ / EN), empty line to exit")
    while True:
        try:
            q = input("\n> ").strip()
        except EOFError:
            break
        if not q:
            break
        show(pipe.answer(q))


if __name__ == "__main__":
    main()
