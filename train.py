"""
=============================================================================
 Car Price Prediction — Model Training Pipeline
=============================================================================
 This script handles the complete ML pipeline:
   1. Load and preprocess data
   2. Feature engineering
   3. Train 4 regression models
   4. Evaluate and compare models
   5. Hyperparameter-tune the best model
   6. Save the final model

 Usage:
   python train.py
=============================================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# Fix Windows console encoding for emoji/unicode characters
import sys
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

# ── Constants ────────────────────────────────────────────────────────────────
CURRENT_YEAR = 2026
DATA_PATH = os.path.join("data", "car_data.csv")
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_data(path: str) -> pd.DataFrame:
    """Load the dataset from CSV."""
    df = pd.read_csv(path)
    print(f"✅ Loaded dataset: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


# =============================================================================
# STEP 1 — Data Preprocessing & Feature Engineering
# =============================================================================
def preprocess(df: pd.DataFrame) -> tuple:
    """
    Preprocess the dataset:
      - Drop Car_Name (too many unique values for useful encoding)
      - Create Car_Age feature
      - Encode categorical variables
      - Split into train/test sets

    Returns:
        (X_train, X_test, y_train, y_test, feature_names)
    """
    print("\n" + "=" * 60)
    print("  STEP 1: Data Preprocessing & Feature Engineering")
    print("=" * 60)

    df = df.copy()

    # ── 1a. Encode Car_Name ──────────────────────────────────────
    le_car = LabelEncoder()
    df["Car_Name"] = le_car.fit_transform(df["Car_Name"])
    print(f"  🏷️  Label encoded 'Car_Name': {len(le_car.classes_)} unique models")

    # ── 1b. Feature Engineering: Car_Age ─────────────────────────
    # Why: The absolute year is less meaningful than HOW OLD the car is.
    df["Car_Age"] = CURRENT_YEAR - df["Year"]
    df.drop(columns=["Year"], inplace=True)
    print(f"  🔧 Created 'Car_Age' = {CURRENT_YEAR} - Year  (range: {df['Car_Age'].min()}–{df['Car_Age'].max()})")

    # ── 1c. Encode Categorical Variables ─────────────────────────
    # Fuel_Type: One-hot encoding (3 categories, no ordinal relationship)
    df = pd.get_dummies(df, columns=["Fuel_Type"], drop_first=True, dtype=int)
    print("  🏷️  One-hot encoded 'Fuel_Type' → Fuel_Type_Diesel, Fuel_Type_Petrol")

    # Seller_Type: Label encoding (binary: Dealer=0, Individual=1)
    le_seller = LabelEncoder()
    df["Seller_Type"] = le_seller.fit_transform(df["Seller_Type"])
    print(f"  🏷️  Label encoded 'Seller_Type': {dict(zip(le_seller.classes_, le_seller.transform(le_seller.classes_)))}")

    # Transmission: Label encoding (binary: Automatic=0, Manual=1)
    le_trans = LabelEncoder()
    df["Transmission"] = le_trans.fit_transform(df["Transmission"])
    print(f"  🏷️  Label encoded 'Transmission': {dict(zip(le_trans.classes_, le_trans.transform(le_trans.classes_)))}")

    # ── 1d. Train/Test Split ─────────────────────────────────────
    X = df.drop(columns=["Selling_Price"])
    y = df["Selling_Price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"\n  📊 Train/Test Split ({int((1-TEST_SIZE)*100)}/{int(TEST_SIZE*100)}):")
    print(f"     Train: {X_train.shape[0]} samples")
    print(f"     Test:  {X_test.shape[0]} samples")

    print(f"\n  Features ({X.shape[1]}): {list(X.columns)}")
    print(f"  Target: Selling_Price")

    return X_train, X_test, y_train, y_test, list(X.columns), le_car


# =============================================================================
# STEP 2 — Model Training & Evaluation
# =============================================================================
def train_and_evaluate(X_train, X_test, y_train, y_test) -> dict:
    """
    Train 4 regression models and evaluate each.

    Models:
      1. Linear Regression
      2. Decision Tree Regressor
      3. Random Forest Regressor
      4. Gradient Boosting Regressor

    Returns:
        dict of {model_name: (model, metrics_dict)}
    """
    print("\n" + "=" * 60)
    print("  STEP 2: Model Training & Evaluation")
    print("=" * 60)

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=RANDOM_STATE),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=RANDOM_STATE),
    }

    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = {
            "R2": r2_score(y_test, y_pred),
            "MAE": mean_absolute_error(y_test, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        }
        results[name] = (model, metrics)

    # Pretty-print comparison table
    print(f"\n  {'Model':<25} {'R² Score':>10} {'MAE':>10} {'RMSE':>10}")
    print("  " + "-" * 57)
    for name, (_, metrics) in results.items():
        print(f"  {name:<25} {metrics['R2']:>10.4f} {metrics['MAE']:>10.4f} {metrics['RMSE']:>10.4f}")

    # Find best model by R²
    best_name = max(results, key=lambda k: results[k][1]["R2"])
    best_r2 = results[best_name][1]["R2"]
    print(f"\n  🏆 Best Model: {best_name} (R² = {best_r2:.4f})")

    return results


# =============================================================================
# STEP 3 — Hyperparameter Tuning
# =============================================================================
def tune_best_model(best_name: str, X_train, y_train, X_test, y_test):
    """
    Apply RandomizedSearchCV to the best model for hyperparameter tuning.
    
    Why RandomizedSearchCV over GridSearchCV?
      - With small dataset (302 rows), RandomizedSearch is faster and explores
        the parameter space more efficiently than exhaustive Grid search.
    """
    print("\n" + "=" * 60)
    print("  STEP 3: Hyperparameter Tuning")
    print("=" * 60)
    print(f"  Tuning: {best_name}")

    if best_name == "Random Forest":
        base_model = RandomForestRegressor(random_state=RANDOM_STATE)
        param_distributions = {
            "n_estimators": [50, 100, 200, 300, 500],
            "max_depth": [None, 5, 10, 15, 20, 25],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", None],
        }
    elif best_name == "Gradient Boosting":
        base_model = GradientBoostingRegressor(random_state=RANDOM_STATE)
        param_distributions = {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [3, 5, 7, 10],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "subsample": [0.8, 0.9, 1.0],
        }
    elif best_name == "Decision Tree":
        base_model = DecisionTreeRegressor(random_state=RANDOM_STATE)
        param_distributions = {
            "max_depth": [None, 5, 10, 15, 20],
            "min_samples_split": [2, 5, 10, 20],
            "min_samples_leaf": [1, 2, 4, 8],
            "max_features": ["sqrt", "log2", None],
        }
    else:
        # Linear Regression has no meaningful hyperparameters
        print("  ⚠️  Linear Regression has no hyperparameters to tune.")
        print("  Returning default model.")
        model = LinearRegression()
        model.fit(X_train, y_train)
        return model

    search = RandomizedSearchCV(
        base_model,
        param_distributions=param_distributions,
        n_iter=50,
        cv=5,
        scoring="r2",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X_train, y_train)

    print(f"\n  Best Parameters:")
    for param, value in search.best_params_.items():
        print(f"    {param}: {value}")
    print(f"\n  Best CV R² Score: {search.best_score_:.4f}")

    # Evaluate tuned model on test set
    tuned_model = search.best_estimator_
    y_pred = tuned_model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"\n  Tuned Model Performance on Test Set:")
    print(f"    R² Score: {r2:.4f}")
    print(f"    MAE:      {mae:.4f}")
    print(f"    RMSE:     {rmse:.4f}")

    return tuned_model


# =============================================================================
# STEP 4 — Save Model
# =============================================================================
def save_model(model, path: str, feature_names: list, le_car: LabelEncoder):
    """
    Save the trained model and feature names using joblib.

    Why joblib over pickle?
      - joblib is optimized for large NumPy arrays (common in sklearn models)
      - Faster serialization/deserialization for ML models

    We save a dict containing both the model and the expected feature names,
    so the prediction function knows exactly what inputs to build.
    """
    print("\n" + "=" * 60)
    print("  STEP 4: Saving Model")
    print("=" * 60)

    os.makedirs(os.path.dirname(path), exist_ok=True)

    model_artifact = {
        "model": model,
        "feature_names": feature_names,
        "current_year": CURRENT_YEAR,
        "car_encoder": le_car,
        "car_names": list(le_car.classes_),
    }
    joblib.dump(model_artifact, path)
    file_size = os.path.getsize(path) / 1024
    print(f"  💾 Model saved to: {path} ({file_size:.1f} KB)")
    print(f"  📋 Features: {feature_names}")

    # Verify loading
    loaded = joblib.load(path)
    print(f"  ✅ Load verification: OK (type: {type(loaded['model']).__name__})")


# =============================================================================
# MAIN — Run Full Pipeline
# =============================================================================
def main():
    print("=" * 60)
    print("  🚗 Car Price Prediction — Training Pipeline")
    print("=" * 60)

    # Load data
    df = load_data(DATA_PATH)

    # Preprocess
    X_train, X_test, y_train, y_test, feature_names, le_car = preprocess(df)

    # Train & evaluate all models
    results = train_and_evaluate(X_train, X_test, y_train, y_test)

    # Find perfect model
    best_name = max(results, key=lambda k: results[k][1]["R2"])

    # Hyperparameter tuning on the best model
    tuned_model = tune_best_model(best_name, X_train, y_train, X_test, y_test)

    # Save the tuned model
    save_model(tuned_model, MODEL_PATH, feature_names, le_car)

    print("\n" + "=" * 60)
    print("  ✅  TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Model: {type(tuned_model).__name__}")
    print(f"  Saved: {MODEL_PATH}")
    print(f"  Next:  Run 'streamlit run app/app.py' to launch the prediction UI")
    print("=" * 60)


if __name__ == "__main__":
    main()
