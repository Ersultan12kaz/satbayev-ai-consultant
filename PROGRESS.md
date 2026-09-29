# Прогресс (где остановились)

Дата: 25.09.2026

## Готово
- [x] Весь код проекта (ml/, scripts/, backend/, frontend/, docker, tests) — в этой папке
- [x] Python 3.11, Git, GitHub CLI установлены (0_install_tools.bat)
- [x] Библиотеки установлены в .venv (1_setup.bat -> logs/setup_status.txt = SETUP OK)
- [x] Автотесты: 25/25 passed (на тестовом корпусе)
- [~] Шаг 1: загрузка документов с official.satbayev.university — начата, прервана
      (продолжение докачивает только недостающие файлы)

## Осталось
1. `2_train_all.bat` — докачать документы -> извлечь текст -> chunking -> обучить 4 модели -> индекс -> оценка (300 вопросов)
2. Проверить reports/evaluation_report.md и logs/*.log, исправить ошибки если будут
3. `3_start_assistant.bat` — проверить чат RU/KZ/EN и админ-панель
4. `5_push_to_github.bat` — опубликовать на GitHub (вход в GitHub в браузере)
