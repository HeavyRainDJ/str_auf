# Лабораторная работа № 4

Вариант 2: обнаружение чашки и поиск параллельных прямых.

## Структура

- `data/cup` — пять изображений для проверки детекторов чашки;
- `data/lines` — шесть изображений со схемами и фасадами;
- `src/evaluate_cups.py` — локальный тест DETR, YOLOS-tiny и RT-DETR-R18;
- `src/evaluate_parallel_lines.py` — M-LSD-tiny и геометрический поиск параллельных пар;
- `results` — CSV, JSON и изображения с разметкой;
- `report/main.tex` — исходник отчёта;
- `output/pdf/lab4_huggingface_models.pdf` — итоговый отчёт.

## Повторный запуск

```powershell
.\.venv\Scripts\python.exe .\src\evaluate_cups.py
.\.venv\Scripts\python.exe .\src\evaluate_parallel_lines.py
```

Порог детекторов чашки — 0,30. Для параллельности используется допуск 3 градуса и минимальное нормальное расстояние 4 пикселя в масштабе 512 x 512.

