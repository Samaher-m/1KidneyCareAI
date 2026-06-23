from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import numpy as np
import xgboost as xgb
from datetime import datetime

app = FastAPI(title="DialyCare AI API", version="1.0.0")

# Allow requests from the HTML frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Train a simple XGBoost model on synthetic dialysis data ──────────────────
np.random.seed(42)
N = 500

# Features: age, diabetes, systolic_bp, hemoglobin, ktv, sessions_missed, potassium, albumin
X_train = np.column_stack([
    np.random.randint(30, 85, N),          # age
    np.random.randint(0, 2, N),            # diabetes (0/1)
    np.random.randint(100, 200, N),        # systolic_bp
    np.random.uniform(7.0, 13.5, N),       # hemoglobin (g/dL)
    np.random.uniform(0.8, 1.8, N),        # Kt/V
    np.random.randint(0, 6, N),            # sessions missed (last month)
    np.random.uniform(3.5, 7.0, N),        # potassium (mEq/L)
    np.random.uniform(2.5, 4.5, N),        # albumin (g/dL)
])

# Risk score 0–100 based on clinical logic
def compute_risk(X):
    score = np.zeros(len(X))
    score += (X[:, 0] - 30) * 0.4          # age
    score += X[:, 1] * 15                  # diabetes penalty
    score += np.clip((X[:, 2] - 140) * 0.3, 0, 20)   # high BP
    score += np.clip((10.5 - X[:, 3]) * 4, 0, 20)     # low hemoglobin
    score += np.clip((1.2 - X[:, 4]) * 25, 0, 20)     # low Kt/V
    score += X[:, 5] * 4                   # missed sessions
    score += np.clip((X[:, 6] - 5.0) * 6, 0, 10)      # high potassium
    score += np.clip((3.5 - X[:, 7]) * 8, 0, 10)      # low albumin
    return np.clip(score + np.random.normal(0, 3, len(X)), 0, 100)

y_train = compute_risk(X_train)

model = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.8,
    random_state=42,
    verbosity=0,
)
model.fit(X_train, y_train)

FEATURE_NAMES = ["age", "diabetes", "systolic_bp", "hemoglobin", "ktv",
                 "sessions_missed", "potassium", "albumin"]

# ── Request / Response schemas ───────────────────────────────────────────────
class PatientFeatures(BaseModel):
    age: int
    diabetes: int                  # 0 or 1
    systolic_bp: float
    hemoglobin: float              # g/dL
    ktv: float                     # Kt/V adequacy
    sessions_missed: int           # last 30 days
    potassium: float               # mEq/L
    albumin: float                 # g/dL
    patient_id: Optional[str] = None
    patient_name: Optional[str] = None

class RiskResponse(BaseModel):
    patient_id: Optional[str]
    patient_name: Optional[str]
    risk_score: float
    risk_level: str
    risk_factors: list[str]
    recommendations: list[str]
    feature_importances: dict
    calculated: str

# ── Helper functions ─────────────────────────────────────────────────────────
def risk_level(score: float) -> str:
    if score >= 80: return "Critical"
    if score >= 60: return "High"
    if score >= 35: return "Medium"
    return "Low"

def build_risk_factors(data: PatientFeatures) -> list[str]:
    factors = []
    if data.age >= 70:
        factors.append(f"Advanced age ({data.age} years)")
    if data.diabetes:
        factors.append("Diabetic patient — increased cardiovascular risk")
    if data.systolic_bp >= 160:
        factors.append(f"Severe hypertension (SBP {data.systolic_bp} mmHg)")
    elif data.systolic_bp >= 140:
        factors.append(f"Elevated blood pressure (SBP {data.systolic_bp} mmHg)")
    if data.hemoglobin < 9.0:
        factors.append(f"Severe anemia (Hgb {data.hemoglobin} g/dL)")
    elif data.hemoglobin < 10.5:
        factors.append(f"Anemia (Hgb {data.hemoglobin} g/dL)")
    if data.ktv < 1.2:
        factors.append(f"Inadequate dialysis adequacy (Kt/V {data.ktv:.2f})")
    if data.sessions_missed >= 3:
        factors.append(f"Frequent missed sessions ({data.sessions_missed} in last month)")
    elif data.sessions_missed >= 1:
        factors.append(f"Missed {data.sessions_missed} session(s) recently")
    if data.potassium > 5.5:
        factors.append(f"Hyperkalemia (K+ {data.potassium} mEq/L)")
    if data.albumin < 3.0:
        factors.append(f"Hypoalbuminemia (Albumin {data.albumin} g/dL)")
    return factors

def build_recommendations(data: PatientFeatures, score: float) -> list[str]:
    recs = []
    if data.ktv < 1.2:
        recs.append("Increase dialysis duration or blood flow rate to improve Kt/V")
    if data.hemoglobin < 10.5:
        recs.append("Review EPO/iron therapy; consider dose adjustment for anemia")
    if data.systolic_bp >= 140:
        recs.append("Optimize antihypertensive regimen and review fluid balance")
    if data.sessions_missed >= 1:
        recs.append("Reinforce attendance importance; schedule patient counseling")
    if data.potassium > 5.5:
        recs.append("Dietary potassium restriction and medication review")
    if data.albumin < 3.5:
        recs.append("Nutritional assessment and dietitian referral")
    if score >= 60:
        recs.append("Increased monitoring frequency — consider bi-weekly physician review")
    if not recs:
        recs.append("Continue routine monthly monitoring and quarterly lab panel")
    return recs

# ── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "ok", "service": "DialyCare AI API", "model": "XGBoost"}

@app.post("/predict", response_model=RiskResponse)
def predict(patient: PatientFeatures):
    X = np.array([[
        patient.age, patient.diabetes, patient.systolic_bp,
        patient.hemoglobin, patient.ktv, patient.sessions_missed,
        patient.potassium, patient.albumin,
    ]])

    score = float(np.clip(model.predict(X)[0], 0, 100))
    importances = dict(zip(FEATURE_NAMES, model.feature_importances_.tolist()))

    return RiskResponse(
        patient_id=patient.patient_id,
        patient_name=patient.patient_name,
        risk_score=round(score, 1),
        risk_level=risk_level(score),
        risk_factors=build_risk_factors(patient),
        recommendations=build_recommendations(patient, score),
        feature_importances=importances,
        calculated=datetime.now().strftime("%Y-%m-%d %H:%M"),
    )

@app.post("/predict/batch")
def predict_batch(patients: list[PatientFeatures]):
    return [predict(p) for p in patients]

@app.get("/model/info")
def model_info():
    return {
        "algorithm": "XGBoost Regressor",
        "n_estimators": 100,
        "max_depth": 4,
        "features": FEATURE_NAMES,
        "output": "risk_score (0–100)",
        "trained_on": "500 synthetic dialysis patient records",
    }
