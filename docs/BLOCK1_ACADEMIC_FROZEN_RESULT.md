# Block 1 — новый заранее замороженный academic holdout

Дата: 20 сентября 2026.

Ветка: `test/block1-quality-gate`

GitHub Actions run: `35517128783`

## Как была заморожена выборка

До запуска модели был зафиксирован manifest:

`docs/GUM_BLOCK1_ACADEMIC_FROZEN_MANIFEST.json`

Источник: GUM, commit
`22fdf87f9c71c96bcc771461d06e689b1f90020d`.

Правило выбора было определено до просмотра предсказаний:

- только жанр `academic`;
- первые 6 документов academic из официального train split;
- оба academic-документа из официального dev split как validation;
- оба academic-документа из официального test split как held-out test;
- выбор не зависит от score, prediction или ошибок модели.

Исходные conll/tsv/xml зафиксированы через Git blob SHA-1. Конвертер проверяет
blob hash локального checkout до преобразования корпуса.

## Результат

Frozen gate: **FAIL**.

Причины:

1. `accepted_link_decision_count`
2. `coreference_below_required_baseline_delta`

### Метрики

- Mention exact-span F1: **0.37335526315789475**
- End-to-end coreference F1: **0.0**
- Accepted links: **0**
- Candidate-bearing mentions: **622**
- Accepted-link coverage: **0.0**
- Accepted-link precision: **не определена**, потому что решений нет

### Новые поверхности

Held-out gold mentions с surface, отсутствующим в train: **529**.

Из них восстановлено exact-span: **187**.

Recall на unseen surfaces: **0.3534971644612476**.

Unsupported candidates: **18**.

### Представимость span

Unsupported gold rate: **0.030405405405405407**.

Это проходит текущий лимит 10%.

### Одинаковая поверхность — разные сущности

- рискованных surface: **11**
- gold mentions в этих группах: **32**
- выбранных link-решений по ним: **0**
- false merges: **0**

Ноль false merges здесь нельзя считать доказательством качества identity linking:
модель вообще не приняла ни одного link-решения.

## Интерпретация

Главный Block-1 стоп-фактор теперь подтверждён на новой заранее замороженной
выборке, а не только на старом news/interview pilot:

- candidate generation уже покрывает большую часть gold spans;
- unseen lexical surfaces частично переносятся;
- автоматическое связывание экземпляров остаётся неработающим при заданном
  консервативном gate;
- end-to-end coreference остаётся ниже контрольного маршрута.

Следовательно, переход к доказательству Block 2 пока не разрешён этим gate.
Следующая работа должна улучшать mention/link representation и learning на
train/validation evidence, а эти два academic test-документа после первого
открытия результата переводятся в regression-only.

Для следующего независимого утверждения о прогрессе потребуется другой новый
held-out manifest, замороженный до запуска улучшенной модели.
