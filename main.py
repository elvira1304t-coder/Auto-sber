from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import pickle
import uvicorn
from src.preprocess import preprocess_input

app = FastAPI(title="SberAuto ML Service")

# Загрузка модели
try:
    with open('models/best_pipeline.pkl', 'rb') as f:
        model = pickle.load(f)
    print("Model loaded!")
except Exception as e:
    print(f"Error: {e}")
    model = None


class HitData(BaseModel):
    """Одно событие с сайта"""
    hit_page_path: str


class SessionData(BaseModel):
    """Данные одного визита"""
    session_id: str
    client_id: str
    visit_date: str
    visit_time: str
    visit_number: int
    utm_source: str
    utm_medium: str
    utm_campaign: Optional[str] = None
    utm_adcontent: Optional[str] = None
    utm_keyword: Optional[str] = None
    device_category: str
    device_os: Optional[str] = None
    device_brand: Optional[str] = None
    device_model: Optional[str] = None
    device_screen_resolution: str
    device_browser: str
    geo_country: str
    geo_city: str


class PredictRequest(BaseModel):
    """Запрос на предсказание"""
    session: SessionData
    hits: List[HitData] = []



@app.get("/")
def root():
    return {"message": "SberAuto ML Service is running!"}


@app.post("/predict")
def predict(request: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        session_dict = request.session.dict()
        hits_list = [h.dict() for h in request.hits]

        input_df = preprocess_input(session_dict, hits_list)

        prediction = int(model.predict(input_df)[0])
        probability = float(model.predict_proba(input_df)[0, 1])

        return {
            "session_id": request.session.session_id,
            "prediction": prediction,
            "probability": probability
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error: {str(e)}")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)