# Продолжение работ с реальными данными

Ветка: `test/p01-real-data-pilot-contract`.
Основа: `implementation/open-mention-learning`, commit
`9bbae264eced86e6be2cccb565ea5c4162fc1e16`. Актуальный commit ветки:
`git rev-parse HEAD` в её checkout.

P01: технический черновик создан, проверен, но **не утверждён** без реальных
данных, прав и выбора условий пилота. Проверка неизвестных решений:

```bash
python scripts/validate_pilot_contract.py docs/real_data/pilot_contract.yaml --require-finalized
```

P02–P24: не начаты в этой ветке. Следующее действие — подтвердить входные
данные и условия из `reports/real-data/01_pilot_contract.md`, затем создать
отдельную поставку P02. Этот файл не содержит пользовательских текстов,
секретов, закрытых меток или весов. Текущие задачи обучения не запущены.
