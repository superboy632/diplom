# Быстрый старт / Quick Start

## Установка и запуск / Installation and Running

### 1. Активируйте conda окружение / Activate conda environment

```bash
conda activate diplom
```

### 2. Установите зависимости / Install dependencies

```bash
pip install -r requirements.txt
```

Или вручную / Or manually:

```bash
pip install rasterio numpy Pillow tqdm
```

### 3. Запустите скрипт / Run the script

**Вариант 1: Используя bash-скрипт / Option 1: Using bash script**

```bash
./run_mosaic_processor.sh
```

**Вариант 2: Прямой запуск Python / Option 2: Direct Python execution**

```bash
python process_moon_mosaic.py
```

## Что будет создано / What will be created

После завершения работы скрипта в папке `dataset_tiles` будут:

After the script completes, in the `dataset_tiles` folder there will be:

- **PNG файлы**: Изображения 416×416 пикселей с именами в формате `lat_min,lon_min,lat_max,lon_max.png`
- **CSV файл**: `tiles_metadata.csv` с метаданными всех кадров

## Параметры по умолчанию / Default Parameters

- Размер окна / Window size: 416×416 пикселей
- Шаг / Stride: 208 пикселей (50% перекрытие / 50% overlap)
- Точность координат / Coordinate precision: 6 знаков после запятой / decimal places

## Изменение параметров / Changing Parameters

Откройте файл [`process_moon_mosaic.py`](process_moon_mosaic.py:118) и измените параметры в функции `main()`:

Open [`process_moon_mosaic.py`](process_moon_mosaic.py:118) and modify parameters in the `main()` function:

```python
window_size = 416  # Измените размер окна / Change window size
stride = 208       # Измените шаг / Change stride
coord_precision = 6  # Измените точность координат / Change coordinate precision
use_overlap = True  # True = с перекрытием, False = без перекрытия
```

## Пример времени выполнения / Example Execution Time

Для файла размером ~2 ГБ ожидаемое время выполнения:

For a ~2 GB file, expected execution time:

- С перекрытием 50% / With 50% overlap: ~10-15 минут
- Без перекрытия / Without overlap: ~5-10 минут

*Время зависит от производительности вашего компьютера*

*Time depends on your computer's performance*

## Проверка результата / Checking Results

```bash
# Посмотреть количество созданных файлов / Check number of created files
ls dataset_tiles/*.png | wc -l

# Посмотреть первые строки CSV / View first lines of CSV
head -n 5 dataset_tiles/tiles_metadata.csv
```

## Решение проблем / Troubleshooting

### Ошибка GDAL / GDAL Error

```bash
conda install -c conda-forge gdal
```

### Недостаточно памяти / Out of Memory

Уменьшите размер окна или увеличьте шаг:

Reduce window size or increase stride:

```python
window_size = 256  # Меньший размер / Smaller size
stride = 256       # Без перекрытия / No overlap
```

### Файл не найден / File Not Found

Убедитесь, что файл `Lunar_LRO_LROC-WAC_Mosaic_global_100m_June2013.tif` находится в текущей директории.

Ensure that `Lunar_LRO_LROC-WAC_Mosaic_global_100m_June2013.tif` is in the current directory.
