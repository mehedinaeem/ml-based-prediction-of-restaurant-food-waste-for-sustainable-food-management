from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class PredictionRequest(BaseModel):
    features: dict


@app.get("/")
def root():
    return {"message": "Restaurant Food Waste Prediction API"}


@app.post("/predict")
def predict(request: PredictionRequest):
    return {"prediction": None}
