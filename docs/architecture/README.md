# Архитектура системы (Этап 1)

## Общая схема

```
                     SATBAYEV AI CONSULTANT
                              │
          ┌───────────────────┴───────────────────┐
       Студент                                  Админ
          │                                       │
   Портал + чат (frontend/)             Админ-панель (frontend/admin.*)
          │   RU / KZ / EN                        │ документы, статистика,
          ▼                                       ▼ пользователи, логи
                    FastAPI backend (backend/app)
                              │
                              ▼
                     RAG-конвейер (ml/rag.py)
   ┌──────────────────────────────────────────────────────────────────┐
   │ 1. small talk / память диалога         ml/followup.py              │
   │ 2. определение языка (kk/ru/en,        ml/langdetect.py            │
   │    казахский без каз. букв, смешанный)                             │
   │ 3. классификатор темы (университет?)   ml/scope.py                 │
   │ 4. расширение запроса (понятия,        ml/lexicon.py               │
   │    двуязычный словарь)                                             │
   │ 5. база фактов (ректор, институты…)    ml/facts.py                 │
   │ 6. гибридный поиск: эмбеддинги + BM25  ml/embeddings.py, bm25.py,  │
   │                                        retriever.py                │
   │ 7. реранкер                            ml/reranker.py              │
   │ 8. защита от галлюцинаций, тип вопроса ml/qtype.py                 │
   │ 9. ответ: композитор предложений       ml/composer.py              │
   │    + собственная GPT (42 M)            ml/lm/                      │
   │10. перевод ответа на язык вопроса      ml/nmt/ (Transformer 52 M)  │
   │11. не найдено → Википедия (только о    ml/wiki.py                  │
   │    университете) → «не найдено»                                    │
   └──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
               Ответ + источники (скрыты по умолчанию)
```

Все модели (эмбеддинги, классификаторы, реранкер, языковая модель, переводчик) обучены нами с нуля.
Готовые или платные модели и внешние API не используются.

## Хранилище

| Режим | База данных | Векторы |
|---|---|---|
| Локально (PyCharm, `3_start_assistant.bat`) | SQLite `data/app.db` | numpy в памяти |
| Docker (`docker-compose.yml`) | PostgreSQL | pgvector (`docker/init.sql`) |

## API (Этап 7)

| Метод | Путь | Назначение |
|---|---|---|
| POST | `/api/chat` | вопрос → ответ (язык, источники, id разговора) |
| GET | `/api/conversations`, `/api/conversations/{id}` | история разговоров |
| POST | `/api/messages/{id}/feedback` | оценка ответа |
| GET / POST / DELETE | `/api/documents`, `/api/documents/{id}` | управление документами |
| POST | `/api/auth/register`, `/api/auth/login`; GET `/api/auth/me` | авторизация (JWT) |
| GET / POST | `/api/admin/stats`, `/api/admin/logs`, `/api/admin/reload` | админ-панель |

Swagger: <http://127.0.0.1:8000/docs>.

## Структура репозитория и план (раздел 29)

| В плане | В репозитории | Содержимое |
|---|---|---|
| `backend/` | `backend/` | FastAPI, модели БД, авторизация |
| `frontend/` | `frontend/` | портал, чат, админ-панель (HTML/JS) |
| `rag/` | `ml/` + `scripts/` | RAG-конвейер и обучение моделей (01–21) |
| `data/` | `data/eval/` | тестовые вопросы (документы и модели не хранятся в Git) |
| `evaluation/questions` | `data/eval/` | 300 вопросов + вне темы + без ответа + большой тест 1156 |
| `evaluation/metrics` | `ml/metrics.py`, `reports/*.json` | метрики |
| `evaluation/results` | `reports/*.md`, `reports/figures/` | отчёты и графики |
| `docs/architecture` | `docs/architecture/` | этот файл |
| `docs/research` | `docs/research/` | обзор технологий |
| `docker-compose.yml` | `docker-compose.yml`, `Dockerfile`, `docker/` | развёртывание |

## Ветки Git (раздел 29) и роли (раздел 28)

| Ветка | Этапы плана | Ответственный |
|---|---|---|
| `main` | стабильная версия (релиз v1.0) | вся команда |
| `develop` | интеграция, Этап 12 Docker | вся команда |
| `feature/rag` | Этапы 2–6 + собственная LLM | Студент 2 — ML / RAG |
| `feature/backend` | Этапы 4, 7, 9 (API) | Студент 1 — Backend + Database |
| `feature/frontend` | Этапы 8, 9 (админ-панель) | Студент 3 — Frontend + Analytics |
| `feature/evaluation` | Этапы 10–11 | совместно |
