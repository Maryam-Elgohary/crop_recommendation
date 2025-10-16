# fastapi_crop.py
# تشغيل السيرفر:
#pip install fastapi uvicorn joblib pandas scikit-learn

# python -m uvicorn app_backend:app --reload

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware

# انشاء تطبيق FastAPI
app = FastAPI(title="Crop Recommendation API")

# تفعيل CORS عشان Flutter يقدر يتواصل
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # أو حطي IP المحاكي/الهاتف بدل "*"
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# تحميل الموديل
model = joblib.load("crop_model.pkl")  # ملف النموذج عندك

# تعريف شكل البيانات اللي اليوزر هيدخلها
class Crop(BaseModel):
    N: int
    P: int
    K: int
    temperature: float
    humidity: float
    ph: float
    rainfall: float

# endpoint للتنبؤ بالمحصول
@app.post("/predict")
def predict_crop(crop: Crop):
    # تحويل البيانات ل DataFrame
    features = pd.DataFrame([crop.model_dump()])
    
    # التنبؤ بالمحصول
    prediction = model.predict(features)
    
    # ارجاع النتيجة
    return {"recommended_crop": str(prediction[0])}
