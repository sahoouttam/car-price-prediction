import datetime

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

model = joblib.load('car_price_model.pkl')

app = FastAPI(title="Car Price Prediction")


class CarInput(BaseModel):
    Car_Name: str
    Year: int
    Present_Price: float
    Kms_Driven: float
    Fuel_Type: str
    Seller_Type: str
    Transmission: str
    Owner: int


@app.get("/")
def home():
    return {"message": "Car price prediction is running"}


@app.post("/predict")
def predict(car: CarInput):
    input_dict = car.dict()
    current_year = datetime.datetime.now().year
    input_dict["Car_Age"] = current_year - input_dict["Year"]

    input_dict.pop("Year")
    input_dict.pop("Car_Name")

    df = pd.DataFrame([input_dict])
    prediction = model.predict(df)[0]

    return {"predicted_price": round(prediction, 2)}
