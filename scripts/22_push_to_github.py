"""Publish the project to GitHub stage by stage, following the diploma plan.

Plan, section 27 (stages 1-12) and section 29 (branches main / develop / feature/*):

    main ── Этап 1 (requirements + architecture)
      └─ develop
           ├─ feature/rag         Этап 2 data, 3 document pipeline, 4 vectors, 5 first RAG,
           │                      6 multilingual (RU/KZ/EN + translator), own LLM
           ├─ feature/backend     Этап 7 FastAPI, Этап 9 auth + admin API
           ├─ feature/frontend    Этап 8 chat UI, Этап 9 admin panel
           ├─ feature/evaluation  Этап 10 test questions, Этап 11 metrics + reports
           └─ Этап 12 Docker + deployment
    main <── develop  (release v1.0)

Every feature branch is merged into develop with a merge commit, so GitHub shows the branch history.
Commits are built with git plumbing and a private index: the project folder itself is never changed.
Nothing large or secret is committed: models/, data/raw, data/processed, data/lm, data/translation,
logs/, .env, _training_files/, _to_delete/ are in .gitignore; files > 50 MB are refused.

Run through 22_push_to_github.bat (it logs in to GitHub first).
    python scripts/22_push_to_github.py [--repo NAME] [--private] [--no-push]
"""
from __future__ import annotations

import argparse
import fnmatch
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_MB = 50
TRAILER = ("Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n"
           "Claude-Session: https://claude.ai/code/session_01YPg3DCEb7dvwZwhRf5U6NK")

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# (branch, commit title, body, path patterns). Patterns are git pathspecs relative to the repo root.
STAGES = [
    ("main", "Этап 1: требования и архитектура системы",
     "README, архитектура (docs/architecture), исследование RAG / LLM / embeddings / vector DB /\n"
     "multilingual NLP / hallucination (docs/research), зависимости и пример .env.",
     ["README.md", "PROGRESS.md", ".gitignore", ".env.example", "requirements.txt",
      "requirements-postgres.txt", "docs"]),

    ("feature/rag", "Этап 2: сбор официальных документов Satbayev University",
     "Загрузка документов official.satbayev.university/ru/docs, страниц сайтов университета\n"
     "и дополнительных открытых источников.",
     ["scripts/_common.py", "scripts/01_download_documents.py", "scripts/01b_download_website.py",
      "scripts/01c_extra_sources.py", "10_update_documents.bat", "11_add_website_pages.bat"]),
    ("feature/rag", "Этап 3: document pipeline (PDF / DOC → текст → чанки)",
     "Извлечение текста из PDF, DOC/DOCX, HTML, OCR для сканов, очистка и разбиение на чанки.",
     ["ml/__init__.py", "ml/text.py", "ml/documents.py", "ml/convert.py", "ml/msdoc.py", "ml/ocr.py",
      "scripts/02_extract_text.py", "scripts/03_chunking.py", "scripts/setup_ocr.py",
      "6_ocr_scanned_documents.bat"]),
    ("feature/rag", "Этап 4: эмбеддинги и векторный индекс",
     "Собственная модель эмбеддингов (subword encoder, обучена с нуля), BM25, гибридный поиск,\n"
     "построение индекса (numpy локально / pgvector в Docker).",
     ["ml/embeddings.py", "ml/bm25.py", "ml/retriever.py",
      "scripts/06_train_embeddings.py", "scripts/07_build_index.py"]),
    ("feature/rag", "Этап 5: первый RAG-прототип",
     "question → scope → retrieval → reranker → answer: классификатор темы, тип вопроса, реранкер,\n"
     "композитор ответа, база фактов, память диалога, лексикон понятий, Википедия как запасной источник.",
     ["ml/rag.py", "ml/scope.py", "ml/qtype.py", "ml/reranker.py", "ml/composer.py", "ml/generator.py",
      "ml/facts.py", "ml/followup.py", "ml/lexicon.py", "ml/wiki.py", "ml/training_data.py",
      "scripts/05_train_scope_classifier.py", "scripts/07b_extract_facts.py",
      "scripts/08_train_reranker.py", "scripts/10_ask.py", "scripts/run_all.py",
      "2_train_all.bat", "9_retrain_models.bat"]),
    ("feature/rag", "Этап 6: мультиязычность RU / KZ / EN",
     "Определитель языка (в т.ч. казахский без казахских букв и смешанный каз+рус),\n"
     "собственный переводчик Transformer (52 M, обучен с нуля).",
     ["ml/langdetect.py", "ml/nmt", "scripts/04_train_language_detector.py",
      "scripts/11_prepare_translation_data.py", "scripts/12_train_translator.py",
      "15_translator_1_data.bat", "16_translator_2_train.bat"]),
    ("feature/rag", "Собственная языковая модель (GPT 42 M) для ответов своими словами",
     "BPE-токенизатор 32k, предобучение с нуля, дообучение на примерах «вопрос → документы → ответ»,\n"
     "проверки ответа (язык, факты, связь с вопросом).",
     ["ml/lm", "scripts/13_lm_corpus.py", "scripts/14_lm_tokenize.py", "scripts/15_lm_pretrain.py",
      "scripts/16_lm_sft_data.py", "scripts/17_lm_finetune.py",
      "12_llm_1_prepare.bat", "13_llm_2_pretrain.bat", "14_llm_3_finetune.bat"]),

    ("feature/backend", "Этап 7: backend на FastAPI",
     "POST /api/chat, GET /api/conversations, POST /api/documents, DELETE /api/documents/{id};\n"
     "SQLAlchemy (SQLite локально / PostgreSQL + pgvector в Docker).",
     ["backend/__init__.py", "backend/app/__init__.py", "backend/app/main.py", "backend/app/schemas.py",
      "backend/app/core/__init__.py", "backend/app/core/config.py", "backend/app/core/database.py",
      "backend/app/models/__init__.py", "backend/app/models/document.py",
      "backend/app/models/chunk.py", "backend/app/models/conversation.py",
      "backend/app/services", "backend/app/api/__init__.py", "backend/app/api/chat.py",
      "backend/app/api/documents.py", "scripts/serve.py",
      "0_install_tools.bat", "1_setup.bat", "3_start_assistant.bat", "8_fix_windows_block.bat"]),
    ("feature/backend", "Этап 9: авторизация и API админ-панели",
     "Регистрация / вход (PBKDF2 + JWT), роли, ограничение частоты запросов, статистика и логи.",
     ["backend/app/core/security.py", "backend/app/models/user.py",
      "backend/app/api/auth.py", "backend/app/api/admin.py"]),

    ("feature/frontend", "Этап 8: веб-интерфейс чата",
     "Портал студента и чат: выбор языка RU / KZ / EN, история, источники (скрыты по умолчанию).",
     ["frontend/index.html", "frontend/chat.html", "frontend/app.js", "frontend/portal.js",
      "frontend/portal.css", "frontend/style.css", "frontend/i18n.js"]),
    ("feature/frontend", "Этап 9: админ-панель",
     "Документы, статистика, пользователи, логи.",
     ["frontend/admin.html", "frontend/admin.js"]),

    ("feature/evaluation", "Этап 10: тестовые вопросы и автотесты",
     "300+ вопросов для оценки, вопросы вне темы и без ответа, 1156 вопросов большого теста,\n"
     "pytest-тесты, конфигурации запуска PyCharm.",
     ["data/eval", "tests", "4_run_tests.bat", ".run"]),
    ("feature/evaluation", "Этап 11: эксперименты, метрики и отчёты",
     "accuracy / recall / precision / hallucination rate / latency / citation accuracy, графики обучения,\n"
     "итоговые отчёты тестирования.",
     ["ml/metrics.py", "scripts/09_evaluate.py", "scripts/21_big_test.py",
      "7_evaluate_only.bat", "21_big_test.bat", "reports", "evaluation"]),

    ("develop", "Этап 12: Docker и развёртывание",
     "docker-compose (PostgreSQL + pgvector, backend, nginx), скрипты публикации и доучивания.",
     ["Dockerfile", "docker-compose.yml", "docker", "5_push_to_github.bat", "22_push_to_github.bat",
      "scripts/22_push_to_github.py", "20_improve_assistant.bat"]),
]
FEATURES = ["feature/rag", "feature/backend", "feature/frontend", "feature/evaluation"]


def git(*args: str, check: bool = True, quiet: bool = False) -> str:
    r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if check and r.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} failed:\n{r.stdout}\n{r.stderr}")
    if not quiet and r.stdout.strip():
        print(r.stdout.strip())
    return r.stdout.strip()


def gh(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["gh", *args], cwd=ROOT, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")


def has_commits() -> bool:
    return subprocess.run(["git", "rev-parse", "--verify", "HEAD"], cwd=ROOT,
                          capture_output=True).returncode == 0


def refuse_big_files(env: dict | None = None) -> None:
    out = git_env(env, "diff", "--cached", "--name-only", "-z", *(["HEAD"] if has_commits() and env is None else []))
    for p in [x for x in out.split("\0") if x]:
        f = ROOT / p
        if f.is_file() and f.stat().st_size > MAX_MB * 1024 * 1024:
            git_env(env, "rm", "--cached", "-q", "--", p)
            print(f"  ! skipped (> {MAX_MB} MB): {p}")


def git_env(env: dict | None, *args: str) -> str:
    import os
    e = dict(os.environ, **(env or {}))
    r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", env=e)
    if r.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} failed:\n{r.stdout}\n{r.stderr}")
    return r.stdout.strip()


class History:
    """Builds commits with a private index (git plumbing): the working folder is never touched,
    so no file of the project can disappear while branches are created."""

    def __init__(self) -> None:
        self.tmp_index = ROOT / ".git" / "stage_index"
        self.env = {"GIT_INDEX_FILE": str(self.tmp_index)}
        self.tips: dict[str, str] = {}

    def _tree_with(self, base: str | None, paths: list[str]) -> str:
        if base:
            git_env(self.env, "read-tree", base)
        else:
            git_env(self.env, "read-tree", "--empty")
        existing = [p for p in paths if (ROOT / p).exists()]
        if existing:
            git_env(self.env, "add", "--", *existing)
        elif paths == ["."]:
            git_env(self.env, "add", "-A")
        big = []
        for p in [x for x in git_env(self.env, "ls-files", "-z").split("\0") if x]:
            f = ROOT / p
            if f.is_file() and f.stat().st_size > MAX_MB * 1024 * 1024:
                big.append(p)
        if big:
            git_env(self.env, "rm", "--cached", "-q", "--", *big)
            for p in big:
                print(f"  ! skipped (> {MAX_MB} MB): {p}")
        return git_env(self.env, "write-tree")

    def _commit(self, tree: str, parents: list[str], title: str, body: str) -> str:
        args = ["commit-tree", tree]
        for p in parents:
            args += ["-p", p]
        args += ["-m", title]
        if body:
            args += ["-m", body]
        args += ["-m", TRAILER]
        return git_env(None, *args)

    def commit(self, branch: str, title: str, body: str, paths: list[str]) -> None:
        tip = self.tips.get(branch)
        tree = self._tree_with(tip, paths)
        if tip and tree == git_env(None, "rev-parse", f"{tip}^{{tree}}"):
            print(f"  (nothing to commit) {title}")
            return
        self.tips[branch] = self._commit(tree, [tip] if tip else [], title, body)
        n = len(git_env(None, "diff-tree", "--no-commit-id", "--name-only", "-r", "--root",
                        self.tips[branch]).splitlines())
        print(f"  [{branch}] {title}  ({n} files)")

    def merge(self, src: str, dst: str, title: str, body: str, paths: list[str] | None) -> None:
        if src not in self.tips or dst not in self.tips:
            return
        if paths is None:           # dst has nothing that src lacks: take src's tree
            tree = git_env(None, "rev-parse", f"{self.tips[src]}^{{tree}}")
        else:                       # union: dst tree + the files the feature branch added
            tree = self._tree_with(self.tips[dst], paths)
        self.tips[dst] = self._commit(tree, [self.tips[dst], self.tips[src]], title, body)
        print(f"  <- merged {src} into {dst}")

    def save(self) -> None:
        for b, sha in self.tips.items():
            git("update-ref", f"refs/heads/{b}", sha)
        git("symbolic-ref", "HEAD", "refs/heads/main")
        git("read-tree", "main")                  # real index = main, working folder unchanged
        subprocess.run(["git", "update-index", "-q", "--refresh"], cwd=ROOT, capture_output=True)
        try:
            self.tmp_index.unlink()
        except OSError:
            pass


def build_history() -> None:
    h = History()
    feature_paths: dict[str, list[str]] = {}
    merged: set[str] = set()
    for branch, title, body, paths in STAGES:
        if branch in FEATURES:
            if branch not in h.tips:              # every feature starts from develop (parallel work)
                h.tips[branch] = h.tips["develop"]
            feature_paths.setdefault(branch, []).extend(paths)
        elif branch == "develop":
            for f in FEATURES:                    # finish features before the deployment stage
                if f in h.tips and f not in merged:
                    h.merge(f, "develop", f"Merge branch '{f}' into develop", "", feature_paths[f])
                    merged.add(f)
        h.commit(branch, title, body, paths)
        if branch == "main" and "develop" not in h.tips:
            h.tips["develop"] = h.tips["main"]
    h.commit("develop", "Прочие файлы проекта", "Оставшиеся вспомогательные файлы.", ["."])
    h.merge("develop", "main", "Release v1.0: мультиязычный RAG-ассистент Satbayev University",
            "Слияние develop -> main: все этапы плана 1-12.", None)
    h.save()
    git("tag", "-f", "-a", "-m", "Diploma version 1.0", "v1.0", "main")
    print("  tag v1.0")


def update_existing() -> None:
    git("add", "-A", quiet=True)
    refuse_big_files()
    if git("diff", "--cached", "--name-only", quiet=True):
        git("commit", "-q", "-m", "Обновление проекта", "-m", "Изменения после доработки ассистента.",
            "-m", TRAILER)
        git("branch", "-f", "develop", "HEAD")
        print("  new commit on main (develop moved to the same commit)")
    else:
        print("Nothing new to commit.")


def configure_identity() -> str:
    login = gh("api", "user", "--jq", ".login").stdout.strip()
    uid = gh("api", "user", "--jq", ".id").stdout.strip()
    if not login:
        raise SystemExit("Not logged in to GitHub: run  gh auth login  (22_push_to_github.bat does it).")
    git("config", "user.name", login)
    git("config", "user.email", f"{uid}+{login}@users.noreply.github.com" if uid else
        f"{login}@users.noreply.github.com")
    print(f"GitHub user: {login}")
    return login


def ensure_remote(repo: str, private: bool) -> str:
    login = gh("api", "user", "--jq", ".login").stdout.strip()
    full = f"{login}/{repo}"
    if subprocess.run(["git", "remote", "get-url", "origin"], cwd=ROOT, capture_output=True).returncode:
        if gh("repo", "view", full, "--json", "url").returncode != 0:
            r = gh("repo", "create", full, "--private" if private else "--public",
                   "--description",
                   "Intelligent multilingual RAG virtual assistant for Satbayev University students "
                   "(KZ / RU / EN), all models trained from scratch — diploma project")
            if r.returncode != 0:
                raise SystemExit(f"gh repo create failed:\n{r.stdout}\n{r.stderr}")
            print(f"Created repository {full}")
        git("remote", "add", "origin", f"https://github.com/{full}.git")
    for topic in ("rag", "diploma", "fastapi", "nlp", "kazakh", "satbayev-university"):
        gh("repo", "edit", full, "--add-topic", topic)
    return f"https://github.com/{full}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="satbayev-ai-consultant")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--no-push", action="store_true", help="build the local history only")
    a = ap.parse_args()

    if not (ROOT / ".git").exists():
        git("init", "-q", "-b", "main")
    git("config", "core.autocrlf", "true")
    git("config", "core.quotepath", "false")
    if not a.no_push:
        configure_identity()
    elif not git("config", "user.name", check=False, quiet=True):
        git("config", "user.name", "local-test")
        git("config", "user.email", "local-test@example.com")

    if has_commits():
        print("Repository already has history → adding a new commit (main and develop).")
        update_existing()
    else:
        print("Building the history stage by stage:")
        build_history()

    print()
    print(git("log", "--oneline", "--graph", "--all", "-n", "40", quiet=True))
    if a.no_push:
        return
    url = ensure_remote(a.repo, a.private)
    branches = ["main", "develop"] + [b for b in FEATURES if subprocess.run(
        ["git", "rev-parse", "--verify", b], cwd=ROOT, capture_output=True).returncode == 0]
    git("push", "-u", "origin", *branches)
    git("push", "-f", "origin", "--tags")
    (ROOT / "logs").mkdir(exist_ok=True)
    (ROOT / "logs" / "github_url.txt").write_text(url + "\n", encoding="utf-8")
    print(f"\nDONE: {url}")


if __name__ == "__main__":
    main()
