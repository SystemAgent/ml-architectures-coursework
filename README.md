# ml-architectures-coursework

Този проект предоставя машинен модел,
използващ регресия с поддържащи вектори за прогнозиране
на дневната консумация на електроенергия. 
Моделът е обучен с данни от [Kaggle](https://www.kaggle.com/datasets/uciml/electric-power-consumption-data-set).

## Инсталирайте зависимостите в virtualenv

``` virtualenv -p python3.10 venv```

```source venv/bin/activate```

```pip install -r requirements.txt```   


## Постройте Docker контейнера

```docker build -t electricity-prediction-api .```

## Стартирайте Docker контейнера

```docker run -p 8000:8000 electricity-prediction-api```    

След като API-то е стартирано, можете да правите прогнози,
като изпратите POST заявка до ендпойнта /predict/.

Примерна заявка с curl:

```bash
curl -X POST "http://localhost:8000/predict/" \
     -H "Content-Type: application/json" \
     -d '{"day": 10, "month": 11, "year": 2010, "global_active_power": 2.070}'