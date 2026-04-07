from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.schemas import PredictionRequest, PredictionResponse

from tensorflow.keras.models import load_model
import joblib

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from utils import clean_title

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR.parent / "models" / "tfidf_model.keras"
VECTORIZER_PATH = BASE_DIR.parent / "models" / "tfidf_vectorizer.joblib"

model = load_model(MODEL_PATH)
tfidf_vectorizer = joblib.load(VECTORIZER_PATH)

app = FastAPI(
    title="API NLP FAKE NEWS DETECTOR",
    description="API permettant l'interaction avec un modèle de détection de fake news",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def read_root() -> dict:
    return {"message": "API opérationnelle"}

@app.post("/predict", response_model=PredictionResponse)
def predict_intent(payload: PredictionRequest) -> PredictionResponse:
  
    tokens = clean_title(payload.text)
    cleaned_text = " ".join(tokens)

    vectorized_input = tfidf_vectorizer.transform([cleaned_text])
    vectorized_input = vectorized_input.toarray()

    proba = float(model.predict(vectorized_input)[0][0])

    predicted_label = "REAL" if proba >= 0.5 else "FAKE"
    confidence = round(proba if proba >= 0.5 else 1 - proba, 4)

    scores = {
        "FAKE": round(1 - proba, 4),
        "REAL": round(proba, 4)
    }

    return PredictionResponse(
        text=payload.text,
        predicted_label=predicted_label,
        confidence=confidence,
        scores=scores
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)