from pydantic import BaseModel
from typing import List, Optional

class BBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int

class Prediction(BaseModel):
    label: str
    korean_label: str
    confidence: float
    is_recyclable: bool
    contamination_status: str
    bbox: BBox

class AnalysisData(BaseModel):
    detected: bool
    predictions: List[Prediction]
    inference_time_ms: float

class AnalysisResponse(BaseModel):
    status: str
    data: AnalysisData
