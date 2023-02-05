import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI
from liver_data import Liver
import joblib


app = FastAPI()
classifier = joblib.load('finalized_model.sav')


@app.post('/predict_liver')
def predict_diabetes(data:Liver):
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
        prediction="NO chances of getting Liver disease"
    else:
        prediction="YOU have chances of getting Liver disease"
    return {'prediction': prediction}

