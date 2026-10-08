# YOUTH_RETENTION

Анализ удержания молодежи на открытых онлайн-курсах (OULAD).

## Состав
- `analysis_oulad.py` — основной скрипт анализа
- `analysis_oulad.ipynb` — ноутбук с анализом
- `figures/` — графики
- `results_summary.json` — сводные результаты

## Данные
Папка `data/` исключена из репозитория (OULAD, большой объём).
Источник: Open University Learning Analytics Dataset (OULAD).
Ожидаемые файлы в `data/`: `assessments.csv`, `courses.csv`, `studentAssessment.csv`,
`studentInfo.csv`, `studentRegistration.csv`, `studentVle.csv`, `vle.csv`.

## Запуск
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
python analysis_oulad.py
# или
jupyter notebook analysis_oulad.ipynb
```
