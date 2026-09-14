# SberAuto ML Service

Сервис для прогнозирования целевых действий пользователей на сайте «СберАвтоподписка» на основе данных о визитах и событиях.

## О проекте

Модель машинного обучения предсказывает, совершит ли пользователь целевое действие (оставит заявку, закажет звонок и т.д.) в рамках визита на сайт. Сервис построен на FastAPI, модель обучена с использованием LightGBM, ROC-AUC на тестовой выборке — 0.89.

## Архитектура
sberauto_ml/

├── data/ # Исходные данные

├── models/ # Обученная модель (best_pipeline.pkl)

├── src/

│└── preprocess.py # Предобработка данных

├── tests/ # Тесты и JSON-запросы

├── scripts/ # Скрипты для запуска

├── main.py # FastAPI сервис

├── requirements.txt # Зависимости

└── README.md # Описание проекта


## Запуск

### 1. Клонирование репозитория

```bash
git clone <repository-url>
cd sberauto_ml
```

## Создание виртуального окружения:

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```
Linux/Mac:
```bash
python3 -m venv venv
source venv/bin/activate
```

Установка зависимостей:
```bash
pip install -r requirements.txt
```
Запуск сервиса:

Windows:
```bash
scripts\run.bat
```

Linux/Mac:
```bash
bash scripts/run.sh
```

После запуска сервис будет доступен по адресу: http://127.0.0.1:8000

Тестирование API
Swagger UI:

text
http://127.0.0.1:8000/docs
Postman:

Используйте JSON-запросы из папки tests/:

test_request_low.json — низкая активность, prediction = 0

test_request_high.json — высокая активность, prediction = 1

test_request_boundary.json — пограничный случай

Тесты через скрипт:

Windows:
```bash
scripts\test_api.bat
```

Linux/Mac:
```bash
bash scripts/test_api.sh
```

Автоматические тесты (Python):
```bash
python tests/test_api.py
```

Пример запроса и ответа
Запрос (POST /predict):

```bash
{
{
  "session": {
    "session_id": "test_max",
    "client_id": "client_777",
    "visit_date": "2024-06-15",
    "visit_time": "13:00:00",
    "visit_number": 5,
    "utm_source": "google",
    "utm_medium": "organic",
    "utm_campaign": "spring_promo",
    "utm_adcontent": "banner_top",
    "utm_keyword": "bmw_x5",
    "device_category": "desktop",
    "device_os": "Windows",
    "device_brand": "Apple",
    "device_model": null,
    "device_screen_resolution": "1920x1080",
    "device_browser": "Chrome",
    "geo_country": "Russia",
    "geo_city": "Moscow"
  },
  "hits": [
    {"hit_page_path": "/cars/all/bmw/x5/123"},
    {"hit_page_path": "/cars/all/bmw/x5/123"},
    {"hit_page_path": "/cars/all/mercedes-benz/e-class/456"},
    {"hit_page_path": "/cars/all/audi/a6/789"},
    {"hit_page_path": "/cars/all/audi/a6/789"},
    {"hit_page_path": "/cars/all/audi/a6/789"},
    {"hit_page_path": "/cars/all/toyota/camry/111"},
    {"hit_page_path": "/cars/all/toyota/camry/111"},
    {"hit_page_path": "/cars/all/toyota/camry/111"},
    {"hit_page_path": "/cars/all/volkswagen/polo/222"}
  ]
}
```
Ответ:
```bash
{
  "session_id": "test_max",
  "prediction": 1,
  "probability": 0.7166
}
```