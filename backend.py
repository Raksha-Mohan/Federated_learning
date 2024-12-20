from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np
import uvicorn

app = FastAPI()

class PredictionRequest(BaseModel):
    features: List[float]

class PredictionResponse(BaseModel):
    prediction: float

@app.post("/api/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    print(f"Received prediction request with features: {request.features}")  # Debug print
    try:
        features = np.array(request.features).reshape(1, -1)
        prediction = mock_predict(features)
        print(f"Returning prediction: {prediction}")  # Debug print
        return PredictionResponse(prediction=prediction)
    except Exception as e:
        print(f"Error occurred: {str(e)}")  # Debug print
        raise HTTPException(status_code=500, detail=str(e))

def mock_predict(features):
    age, bmi, systolic_bp, creatinine, gfr, diabetes, hba1c, cholesterol, medication_adherence = features[0]
    risk_score = 0
    if age > 65: risk_score += 1
    if gfr < 60: risk_score += 1
    if systolic_bp > 140: risk_score += 1
    if diabetes == 1: risk_score += 1
    if hba1c > 6.5: risk_score += 1
    visits = min(max(risk_score + 1, 1), 4)
    return float(visits)

@app.get("/")
async def root():
    return {"message": "Backend server is running!"}

if __name__ == "__main__":
    print("Starting backend server on port 8001...")  # Debug print
    uvicorn.run(app, host="0.0.0.0", port=8001)