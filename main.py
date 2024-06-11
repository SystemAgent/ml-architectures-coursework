from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from joblib import load

app = FastAPI()
model = load('models/svr_model.joblib')
scaler = load('models/scaler.joblib')


class PredictionRequest(BaseModel):
    day: int
    month: int
    year: int
    global_active_power: float


@app.post('/predict/')
async def predict(request: PredictionRequest):
    try:
        features = np.array([[request.day, request.month, request.year, request.global_active_power]])
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)
        return {'prediction': prediction[0]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
