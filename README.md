# AI2 Text Factors

Исследовательская реализация разреженной ассоциативной памяти для поиска
повторяющихся факторов в потоке текста.

Проект развивает идеи эксперимента
[«Факторы в тексте»](https://github.com/aldrd/aboutbrain/tree/master/%D0%A4%D0%B0%D0%BA%D1%82%D0%BE%D1%80%D1%8B%20%D0%B2%20%D1%82%D0%B5%D0%BA%D1%81%D1%82%D0%B5),
но написан с нуля на Python. Исходный VB.NET-код и приложенные к нему книжные
корпуса сюда не копируются.

## Что уже работает

- детерминированное разреженное кодирование пар «символ–позиция»;
- латиница и кириллица по умолчанию, произвольный алфавит через API/CLI;
- несколько циклических контекстных интерпретаций окна;
- случайные локальные рецептивные поля;
- создание, проверка, подрезание и консолидация кластеров памяти;
- режим поиска факторов без учителя;
- экспериментальный режим обучения преобразованию одного контекста в другой;
- объяснение выходного фактора через наиболее связанные символы и позиции;
- безопасное сохранение модели в сжатый NPZ без `pickle`;
- CLI и автоматические тесты.

## Установка

Требуется Python 3.10 или новее.

### Windows PowerShell

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e .
```

### Linux/macOS

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

## Быстрый запуск

```bash
text-factors demo --text "абракадабра абракадабра абракадабра"
```

Обучение на UTF-8-файле:

```bash
text-factors train \
  --input corpus.txt \
  --model artifacts/model.npz \
  --epochs 3
```

Анализ без изменения модели:

```bash
text-factors analyze \
  --model artifacts/model.npz \
  --text "абракадабра" \
  --top 10
```

Строки длиннее размера кадра автоматически анализируются как последовательность
перекрывающихся окон.

## Python API

```python
from text_factors import ModelConfig, TextFactorModel

config = ModelConfig(point_count=4_000, seed=42)
model = TextFactorModel(config)
model.fit_text("абракадабра абракадабра", epochs=5)

result = model.transform_window("абрак")
print(result.to_dict())

for factor in model.top_factors(5):
    print(factor.to_dict())
    print(model.explain_factor(factor.output_bit, limit=5))

model.save("artifacts/model.npz")
restored = TextFactorModel.load("artifacts/model.npz")
```

## Проверка

Тесты используют стандартную библиотеку и не требуют отдельного тестового
фреймворка:

```bash
python -m unittest discover -s tests -v
python -m compileall -q src tests
```

Полная локальная проверка:

```bash
python -m pip install -e ".[dev]"
ruff check .
ruff format --check .
pyright
```

CI выполняет тесты, форматирование, линтинг и проверку типов на Python 3.10,
3.11 и 3.12.

## Статус проекта

Версия `0.1.0` — проверяемый MVP. Она позволяет исследовать гипотезу, но сама
по себе не доказывает наличие семантических факторов. Следующая научная задача —
добавить корпус с контролируемой структурой, отрицательные примеры и метрики
устойчивости, точности восстановления и воспроизводимости факторов.

Подробное описание математики и отличий от прототипа находится в
[`docs/ALGORITHM.md`](docs/ALGORITHM.md).

## Исследовательская программа

Проект рассматривает идеи Алексея Редозубова как набор проверяемых гипотез, а не
как уже доказанную архитектуру сильного ИИ:

- [критический обзор корпуса работ и выводы для AI2](docs/REDOZUBOV_RESEARCH.md);
- [поэтапная дорожная карта на 12 месяцев](docs/ROADMAP.md);
- [реестр гипотез H1-H8 и правила эксперимента](docs/HYPOTHESES.md).

Полная версия обзора с таблицами и источниками доступна в
[`docs/AI2_Redozubov_research_roadmap.docx`](docs/AI2_Redozubov_research_roadmap.docx).
