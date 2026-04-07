from typing import Dict

from pydantic import BaseModel, Field

class PredictionRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=500)

class PredictionResponse(BaseModel):
    text: str
    predicted_label: str
    confidence: float
    scores: Dict[str, float]
