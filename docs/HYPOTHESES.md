# Реестр гипотез AI2

Версия: 1.0, 13 сентября 2026 года.

Этот файл определяет, какие утверждения проверяет проект и при каком результате
механизм исключается из основного пути. Формулировка гипотезы, primary metric,
данные, baseline, seed и порог решения фиксируются до просмотра тестового
результата.

## H1 Разреженная факторная память

**Утверждение.** Разреженная factor memory улучшает извлечение релевантного
опыта при том же бюджете контекста и хранения.

- Primary metrics: `evidence_recall_at_k`, end-to-end answer accuracy,
  `bytes_per_item`.
- Baselines: BM25, dense RAG, hybrid BM25+dense.
- Данные: Synthetic Factors, private vertical suite, затем LongMemEval.
- Условие отказа: нет статистически значимого выигрыша над лучшим baseline или
  выигрыш исчезает после уравнивания бюджета.
- После отказа: оставить sparse memory исследовательским backend, не включать в
  production path.

## H2 Несколько контекстов

**Утверждение.** Параллельные контекстные преобразования уменьшают ошибки при
неоднозначной интерпретации.

- Primary metric: exact task success на AI2 Ambiguity Suite.
- Baselines: single prompt, multi-query retrieval, prompt ensemble с равным
  числом LLM-вызовов.
- Secondary metrics: abstention precision, calibration, context overlap.
- Условие отказа: выигрыш исчезает против prompt ensemble или не переносится на
  private suite.
- После отказа: оставить Context Bank только как исследовательский режим.

## H3 Локальное онлайн-обучение

**Утверждение.** Локальное обновление factor/context memory быстрее адаптируется
к новой области без полного переобучения.

- Primary metrics: learning curve и cost-to-target.
- Baselines: append-only RAG, обновление dense index, лёгкий adapter.
- Проверка забывания: retention на старой области после каждого пакета
  обновлений.
- Условие отказа: AI2 требует больше данных или времени, чем лучший лёгкий
  baseline, либо вызывает неприемлемое забывание.

## H4 Консолидация

**Утверждение.** Консолидация сжимает опыт без потери важных эпизодов и
корректных обновлений.

- Primary metric: `compression_ratio * critical_retention * update_accuracy`.
- Обязательный контроль: immutable raw event store.
- Условие продолжения: не менее 98% критических фактов и 95% task success после
  консолидации.
- Условие отказа: растёт ложная уверенность, ухудшается temporal reasoning или
  rollback не восстанавливает исходный результат.

## H5 Оценка исходов

**Утверждение.** Отдельный appraisal-слой повышает последовательность выбора
действий.

- Primary metrics: `pass^k`, regret и policy violations.
- Baseline: прямой выбор действия языковой моделью при том же контексте.
- Контроль: hard policy не входит в обучаемую reward/value модель.
- Условие отказа: слой меняет объяснение, но не конечное действие или результат.

## H6 Моделирование последствий

**Утверждение.** Явное моделирование нескольких исходов улучшает решение
многошаговых задач.

- Primary metrics: end-state success и calibrated outcome error.
- Baseline: один прямой план с равным общим числом LLM-вызовов.
- Условие отказа: нет выигрыша при равном бюджете или прогнозы исходов не
  калиброваны.

## H7 Групповое векторное связывание

**Утверждение.** Vector binding и резонанс позволяют извлекать композиции,
неизвестные на этапе обучения.

- Primary metric: held-out recovery against real-covariance null.
- Обязательная калибровка: planted group известного состава.
- Controls: sighted null, matched random directions, shuffled group membership.
- Условие отказа: эффект равен нулевым моделям или возникает только при заранее
  известном составе группы.

## H8 Причинный аудит

**Утверждение.** Причинные вмешательства отделяют реально используемую
структуру от геометрической тени.

- Primary metric: `delta_behavior` после on-target intervention.
- Controls: matched random intervention, mechanical floor, preserved norms и
  downstream guards.
- Условие подтверждения: on-target эффект сильнее matched control, а 95% CI не
  пересекает ноль.
- Условие отказа: probe score меняется без изменения поведения.

## Минимальные правила доказательства

1. До запуска фиксируются гипотеза, данные, разбиение, primary metric, baseline,
   random seed, допустимые исключения и порог решения.
2. Инструмент сначала обязан обнаружить специально внедрённый известный
   механизм. Отрицательный результат без planted-answer calibration не имеет
   силы.
3. Нулевая модель видит те же данные и реальную ковариацию, что и проверяемый
   метод, но не искомую структуру.
4. Threshold выбирается по dev/null до sealed test; после просмотра теста он не
   меняется.
5. Любое утверждение о механизме проходит behavioral gate: вмешательство меняет
   вероятность, действие или конечный результат, а не только probe score.
6. Post-hoc объяснение маркируется как rescue hypothesis, получает новый ID и
   проверяется на новом наборе.
7. Публикуются отрицательные результаты, сырые traces, версии кода и скрипт
   пересчёта итогового вердикта.
8. Для стохастических частей используется минимум три seed/rollout и paired
   bootstrap 95% CI.
9. Каждый результат содержит commit hash, версии модели и данных, конфигурацию и
   стоимость.
10. Решение `ship`, `research-only` или `remove` записывается до следующей
    архитектурной итерации.

## Шаблон записи эксперимента

```yaml
experiment_id: EXP-000
hypothesis_id: H1
claim: ""
data_version: ""
train_dev_test_split: ""
primary_metric: ""
secondary_metrics: []
baselines: []
planted_answer: ""
null_models: []
seeds: [11, 29, 47]
budget:
  context_tokens: 0
  storage_bytes: 0
  llm_calls: 0
  tool_calls: 0
decision_threshold: ""
allowed_exclusions: []
commit_sha: ""
result_artifacts: []
verdict: pending
```

## Журнал решений

| Гипотеза | Последний эксперимент | Вердикт | Действие |
|---|---|---|---|
| H1 | Не запущен | Pending | Построить E2-E4 |
| H2 | Не запущен | Pending | После решения H1 |
| H3 | Не запущен | Pending | После устойчивого event store |
| H4 | Не запущен | Pending | Не включать удаление до retention gate |
| H5 | Не запущен | Pending | После вертикального runtime |
| H6 | Не запущен | Pending | После базового tool-agent |
| H7 | Не запущен | Pending | Отдельная исследовательская ветка |
| H8 | Не запущен | Pending | Применять ко всем заявленным механизмам |
