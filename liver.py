import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from liver_data import Liver
import joblib


app = FastAPI()
origins = ['*']
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

classifier = joblib.load('finalized_model.sav')


@app.post('/predict_liver')
def predict_liver(data:Liver):
    data = data.dict()
    Age = data['Age']
    Gender = data['Gender']
    Total_Bilirubin = data['Total_Bilirubin']
    Alkaline_Phosphotase = data['Alkaline_Phosphotase']
    Alamine_Aminotransferase = data['Alamine_Aminotransferase']
    Albumin_and_Globulin_Ratio = data['Albumin_and_Globulin_Ratio']
    cols = [[Age, Gender, Total_Bilirubin, Alkaline_Phosphotase, Alamine_Aminotransferase, Albumin_and_Globulin_Ratio]]
    prediction = classifier.predict_proba(cols)
    
    if(prediction[0][0]<0.5):
        prediction="No chances of getting Liver disease"
    else:
        prediction="You have high chances of getting Liver disease"
    return {'prediction': prediction}

