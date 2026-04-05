from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

model = joblib.load("model.pkl")

@app.get("/")
def home():
    return {"message": "Auto MPG Prediction API is running"}

@app.post("/predict")
def predict(data: dict):
    features = np.array([[
        data['cylinders'],
        data['displacement'],
        data['horsepower'],
        data['weight'],
        data['acceleration'],
        data['model_year'],
        data['origin']
    ]])
    
    prediction = model.predict(features)[0]
    
    return {"mpg_prediction": round(prediction, 2)}