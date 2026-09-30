from fastapi import FastAPI
from pydantic import BaseModel
import mlflow
import mlflow.sklearn
import pandas as pd


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Insurance Claim Prediction API",
    description="MLOps API for Insurance Claim Prediction",
    version="1.0.0"
)


# ============================================================
# MLFLOW CONFIGURATION
# ============================================================

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_registry_uri("sqlite:///mlflow.db")

MODEL_URI = "./model"

model = mlflow.sklearn.load_model(MODEL_URI)


# ============================================================
# INPUT DATA MODEL
# ============================================================

class InsuranceInput(BaseModel):

    # ----------------------------
    # Numerical features
    # ----------------------------

    subscription_length: float
    vehicle_age: float
    customer_age: float
    region_density: float

    airbags: float

    displacement: float
    cylinder: float
    turning_radius: float
    length: float
    width: float
    gross_weight: float
    ncap_rating: float

    max_power_bhp: float
    max_power_rpm: float
    max_torque_nm: float
    max_torque_rpm: float

    # ----------------------------
    # Binary features
    # ----------------------------

    is_esc: int
    is_adjustable_steering: int
    is_tpms: int
    is_parking_sensors: int
    is_parking_camera: int
    is_front_fog_lights: int
    is_rear_window_wiper: int
    is_rear_window_washer: int
    is_rear_window_defogger: int
    is_brake_assist: int
    is_power_door_locks: int
    is_central_locking: int
    is_power_steering: int
    is_driver_seat_height_adjustable: int
    is_day_night_rear_view_mirror: int
    is_ecw: int
    is_speed_alert: int

    # ----------------------------
    # Categorical features
    # ----------------------------

    region_code: str
    segment: str
    model: str
    fuel_type: str
    engine_type: str
    rear_brakes_type: str
    transmission_type: str
    steering_type: str


# ============================================================
# HOME ENDPOINT
# ============================================================

@app.get("/")
def home():

    return {
        "message": "Insurance Claim Prediction API is running",
        "model": "InsuranceClaimPrediction",
        "version": "1"
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model": "InsuranceClaimPrediction",
        "model_version": "1"
    }


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
def predict(data: InsuranceInput):

    # Convert Pydantic input into dictionary
    input_dict = data.model_dump()

    # Convert to DataFrame
    input_df = pd.DataFrame([input_dict])

    # Prediction
    prediction = model.predict(input_df)

    # Probability
    probability = model.predict_proba(input_df)[:, 1]

    claim_prediction = int(prediction[0])
    claim_probability = float(probability[0])

    return {
        "claim_prediction": claim_prediction,
        "claim_probability": round(claim_probability, 4),
        "message": (
            "Claim likely"
            if claim_prediction == 1
            else "Claim unlikely"
        )
    }