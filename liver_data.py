from pydantic import BaseModel
class Liver(BaseModel):
    Age:int
    Gender:int
    Total_Bilirubin:float
    Alkaline_Phosphotase:float
    Alamine_Aminotransferase:float
    Albumin_and_Globulin_Ratio:float