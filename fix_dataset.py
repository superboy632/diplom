import pandas as pd
import os

# Путь к твоему оригинальному файлу метаданных
INPUT_CSV = 'dataset_tiles/tiles_metadata.csv'
# Куда сохраним исправленный вариант
OUTPUT_CSV = 'dataset_tiles/tiles_metadata_fixed.csv'

print(f"Загружаем {INPUT_CSV}...")
df = pd.read_csv(INPUT_CSV)


def fix_coordinates(row):
    # Очищаем имя файла от кавычек
    filename = str(row['filename']).strip('"')

    # Пытаемся вытащить координаты из названия файла: X_min, Y_min, X_max, Y_max
    name_without_ext = filename.replace('.png', '')

    try:
        parts = name_without_ext.split(',')
        if len(parts) == 4:
            x_min, y_min, x_max, y_max = map(float, parts)

            # Вычисляем истинный центр тайла в метрах
            x_center = (x_min + x_max) / 2.0
            y_center = (y_min + y_max) / 2.0

            # Записываем правильно: Y -> lat (широта), X -> lon (долгота)
            row['lat_center'] = y_center
            row['lon_center'] = x_center
    except Exception as e:
        print(f"Пропущен файл {filename}: {e}")

    return row


# Применяем исправление ко всем строкам
print("Пересчитываем координаты...")
df = df.apply(fix_coordinates, axis=1)

# Сохраняем результат
df.to_csv(OUTPUT_CSV, index=False)
print(f"Готово! Исправленный датасет сохранен как: {OUTPUT_CSV}")