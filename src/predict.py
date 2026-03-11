"""
=============================================================================
 Car Price Prediction — Prediction Function
=============================================================================
 Standalone prediction module that loads the saved model and returns
 a predicted selling price given input features.

 Usage:
   from src.predict import predict_price
   price = predict_price("corolla", 2015, 8.0, 30000, "Petrol", "Dealer", "Manual", 0)
=============================================================================
"""

import os
import numpy as np
import pandas as pd
import joblib

# Path to the saved model (relative to project root)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "model.pkl")


def load_model(path: str = MODEL_PATH):
    """
    Load the saved model artifact.

    The artifact contains:
      - model: the trained sklearn model
      - feature_names: list of expected feature column names
      - current_year: the year used during training for Car_Age calculation
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model not found at '{path}'. Run 'python train.py' first."
        )
    return joblib.load(path)


def predict_price(
    car_name: str,
    year: int,
    present_price: float,
    kms_driven: int,
    fuel_type: str,
    seller_type: str,
    transmission: str,
    owner: int,
    model_path: str = MODEL_PATH,
) -> float:
    """
    Predict the selling price of a car.

    Parameters
    ----------
    car_name : str
        Name/Model of the car (e.g. 'corolla').
    year : int
        Year the car was purchased (e.g. 2015).
    present_price : float
        Current showroom price in Lakhs (e.g. 8.0).
    kms_driven : int
        Total kilometres driven (e.g. 30000).
    fuel_type : str
        One of: "Petrol", "Diesel", "CNG".
    seller_type : str
        One of: "Dealer", "Individual".
    transmission : str
        One of: "Manual", "Automatic".
    owner : int
        Number of previous owners (0, 1, 2, 3).
    model_path : str
        Path to the saved model file.

    Returns
    -------
    float
        Predicted selling price in Lakhs (PKR).
    """
    artifact = load_model(model_path)
    model = artifact["model"]
    feature_names = artifact["feature_names"]
    current_year = artifact["current_year"]
    car_encoder = artifact["car_encoder"]

    # ── Build feature vector (same preprocessing as train.py) ────
    car_age = current_year - year

    # Encode Car_Name
    try:
        car_name_encoded = car_encoder.transform([car_name.lower()])[0]
    except ValueError:
        # If unseen car name, fallback to a default or 0
        car_name_encoded = 0

    # Encode Seller_Type: Dealer=0, Individual=1
    seller_encoded = 1 if seller_type == "Individual" else 0

    # Encode Transmission: Automatic=0, Manual=1
    trans_encoded = 1 if transmission == "Manual" else 0

    # One-hot encode Fuel_Type (drop_first=True → CNG is baseline)
    fuel_diesel = 1 if fuel_type == "Diesel" else 0
    fuel_petrol = 1 if fuel_type == "Petrol" else 0

    # Build DataFrame with EXACT column order used during training
    data = {
        "Car_Name": [car_name_encoded],
        "Present_Price": [present_price],
        "Kms_Driven": [kms_driven],
        "Seller_Type": [seller_encoded],
        "Transmission": [trans_encoded],
        "Owner": [owner],
        "Car_Age": [car_age],
        "Fuel_Type_Diesel": [fuel_diesel],
        "Fuel_Type_Petrol": [fuel_petrol],
    }

    input_df = pd.DataFrame(data)

    # Reorder columns to match training order
    input_df = input_df[feature_names]

    # Predict
    prediction = model.predict(input_df)[0]

    # Ensure non-negative price
    return max(0.0, round(prediction, 2))


import sys
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

# ── Quick Test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing prediction function...")
    try:
        price = predict_price(
            car_name="corolla",
            year=2015,
            present_price=60.0,
            kms_driven=30000,
            fuel_type="Petrol",
            seller_type="Dealer",
            transmission="Manual",
            owner=0,
        )
        print(f"✅ Predicted Selling Price: ₨{price:.2f} Lakhs PKR")
    except FileNotFoundError as e:
        print(f"⚠️  {e}")
