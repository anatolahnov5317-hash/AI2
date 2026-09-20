# AI2 v6 — точка продолжения после блока 1

Блок 1/3 завершён локально. Блоки 2 и 3 не выполнялись.
Репозиторий: `anatolahnov5317-hash/AI2`.
Ветка: `implementation/v06-block1-candidates`, основана на `05e22df941d7f51ef2c2f33813e5e0a4f3c7ff02`.
Версия кода и тестов: SHA-256 `e149bc036df3d6b15c951535ac29a712f9ef2245d70a61a8ba9431e773a92317`.

Результат: 683/683 теста, 26 новых проверок; Ruff/format/Pyright/compileall пройдены.
Два первоначально отсутствующих результата получены отдельным повтором на том же коде.
Подробности и ограничения: [01_candidates_core.md](01_candidates_core.md).
Проверенные ID: [results/block1-verified.json](results/block1-verified.json).
Разработческие трассы: [results/block1-development.json](results/block1-development.json).

Модель: `docs/results/v05_model_42.json`, прежние веса, seed 42, хеш и fingerprint
записаны в разработческих трассах. Никакого скрытого обучения при загрузке нет.
Среда: Python 3.12.14, NumPy 2.3.5; `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`.

Новый маршрут: `interpret` → `propose` → сравнение кандидатов общей памятью →
выбор/уточнение → прежние динамика и транзакция мира. Контракты кандидатов находятся
в `learning/hypotheses.py`, поиск — в `candidate_search.py`, выбор — в
`candidate_selection.py`. История сессии сохраняет наблюдение и диагностические
кандидаты, но ещё не является долговременным архивом альтернатив.

Следующий шаг — блок 2: согласование происхождения между гипотезами, архив
подавленных вариантов, выбор эпизодов и контрсвидетельств, возврат с повторной
проверкой. Не объявлять текущее ранжирование обученным вниманием. Не использовать
раскрытые наборы v0.5 как новый независимый тест.

Перед доработкой блока 2 получить эту ветку и проверить состояние:

```bash
git fetch origin implementation/v06-block1-candidates
git switch --detach origin/implementation/v06-block1-candidates
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 scripts/run_test_batches.py --output /tmp/ai2-v6-block2-start test_hypothesis_candidates
```

Для изменений создать отдельную ветку от этой версии. Не переключаться на main
как на актуальную реализацию: там остаётся прежний MVP. Основная ветка и прежние PR
не переписаны. Проверки выполнять сериями, промежуточные результаты сохранять.
