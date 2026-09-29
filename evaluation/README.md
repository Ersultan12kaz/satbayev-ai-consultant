# Evaluation (Этапы 10–11)

Скрипты пишут результаты в `data/eval/` и `reports/`, поэтому файлы лежат там. Здесь — указатель, как в разделе 29 плана.

| Папка плана | Где лежит | Что внутри |
|---|---|---|
| `questions/` | [`data/eval/`](../data/eval) | `questions.jsonl` — 300 вопросов (RU/KZ/EN, 10 категорий); `out_of_scope.jsonl` — вне темы; `unanswerable.jsonl` — без ответа; `big_test.json` — 1156 вопросов (казахский без казахских букв, опечатки, смешанный каз+рус и т.д.) |
| `metrics/` | [`ml/metrics.py`](../ml/metrics.py), [`reports/*.json`](../reports) | Recall@K, Precision@K, MRR, accuracy, faithfulness, hallucination rate, citation accuracy, latency; метрики обучения моделей |
| `results/` | [`reports/*.md`](../reports), [`reports/figures/`](../reports/figures) | отчёт экспериментов 1–4, итоговое тестирование, графики обучения |

Запуск: `7_evaluate_only.bat` (300 вопросов, эксперименты 1–4) и `21_big_test.bat` (1156 вопросов).
