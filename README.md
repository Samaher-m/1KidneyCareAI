# DialyCare AI Backend — FastAPI + XGBoost

## Setup

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| POST | `/predict` | Predict risk for one patient |
| POST | `/predict/batch` | Predict risk for multiple patients |
| GET | `/model/info` | XGBoost model details |
| GET | `/docs` | Swagger UI (auto-generated) |

## Example Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "patient_id": "P001",
    "patient_name": "Ahmed Al-Rashid",
    "age": 67,
    "diabetes": 1,
    "systolic_bp": 158,
    "hemoglobin": 9.8,
    "ktv": 1.15,
    "sessions_missed": 2,
    "potassium": 5.8,
    "albumin": 3.1
  }'
```

## Example Response

```json
{
  "patient_id": "P001",
  "patient_name": "Ahmed Al-Rashid",
  "risk_score": 74.3,
  "risk_level": "High",
  "risk_factors": [
    "Advanced age (67 years)",
    "Diabetic patient — increased cardiovascular risk",
    "Elevated blood pressure (SBP 158 mmHg)",
    "Anemia (Hgb 9.8 g/dL)",
    "Inadequate dialysis adequacy (Kt/V 1.15)",
    "Missed 2 session(s) recently",
    "Hyperkalemia (K+ 5.8 mEq/L)",
    "Hypoalbuminemia (Albumin 3.1 g/dL)"
  ],
  "recommendations": [
    "Increase dialysis duration or blood flow rate to improve Kt/V",
    "Review EPO/iron therapy; consider dose adjustment for anemia",
    "Optimize antihypertensive regimen and review fluid balance",
    "Reinforce attendance importance; schedule patient counseling",
    "Dietary potassium restriction and medication review",
    "Nutritional assessment and dietitian referral",
    "Increased monitoring frequency — consider bi-weekly physician review"
  ],
  "feature_importances": {
    "age": 0.12,
    "diabetes": 0.18,
    "systolic_bp": 0.09,
    "hemoglobin": 0.14,
    "ktv": 0.21,
    "sessions_missed": 0.11,
    "potassium": 0.08,
    "albumin": 0.07
  },
  "calculated": "2026-06-22 14:30"
}
```

## Connecting to the HTML Frontend

In `index.html`, the AI Risk tab now calls this API automatically.
Make sure the backend is running on `http://localhost:8000` before opening the HTML file.
