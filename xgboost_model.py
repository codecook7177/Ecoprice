"""EcoPrice XGBoost Demand Forecaster.

Trains an XGBoost regressor on the feature matrix to predict daily
demand (quantity) for perishable goods. The trained model is used by
the PPO agent and the dashboard forecast page.

Usage:
    # Train and save model:
    python -m models.xgboost_model

    # From other modules:
    from models.xgboost_model import load_model, predict_demand
    model = load_model()
    pred = predict_demand(model, feature_dict)
"""

import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models" / "saved"
FEATURE_MATRIX = DATA_DIR / "feature_matrix.csv"
MODEL_PATH = MODEL_DIR / "xgboost_demand.joblib"
METRICS_PATH = MODEL_DIR / "xgboost_metrics.json"

# Feature columns (must match src/config.py FEATURE_COLS)
FEATURE_COLS = [
    "days_to_expiry",
    "temperature",
    "day_of_week",
    "is_holiday",
    "days_to_festival",
    "price",
    "rolling_avg_7d",
    "humidity",
    "precipitation",
    "month",
    "is_weekend",
]
TARGET_COL = "quantity"


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------
def load_data(path=None):
    """Load and validate the feature matrix CSV."""
    path = path or FEATURE_MATRIX
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Feature matrix not found at {path}. "
            "Run `python -m src.preprocess` first."
        )
    df = pd.read_csv(path)
    missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in feature matrix: {missing}")

    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()

    print(f"Loaded {len(X):,} rows × {len(FEATURE_COLS)} features")
    print(f"Target '{TARGET_COL}' range: {y.min()} – {y.max()}, mean: {y.mean():.2f}")
    return X, y


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(X, y, test_size=0.2, random_state=42):
    """Train XGBoost regressor with tuned hyperparameters.

    Returns: (model, X_test, y_test, metrics_dict)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"Train: {len(X_train):,} rows  |  Test: {len(X_test):,} rows")

    # Hyperparameters tuned for perishable demand data
    params = {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "reg_alpha": 0.1,        # L1 regularization
        "reg_lambda": 1.0,       # L2 regularization
        "random_state": random_state,
        "n_jobs": -1,
        "tree_method": "hist",   # fast histogram-based
    }

    print("\nTraining XGBoost with params:")
    for k, v in params.items():
        print(f"  {k}: {v}")

    model = xgb.XGBRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=50,  # print every 50 rounds
    )

    # Evaluate
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    metrics = evaluate(y_train, y_pred_train, y_test, y_pred_test)

    # Cross-validation for robustness check
    print("\nRunning 5-fold cross-validation...")
    cv_scores = cross_val_score(
        xgb.XGBRegressor(**params), X, y,
        cv=5, scoring="neg_mean_absolute_error", n_jobs=-1
    )
    metrics["cv_mae_mean"] = round(float(-cv_scores.mean()), 4)
    metrics["cv_mae_std"] = round(float(cv_scores.std()), 4)
    print(f"CV MAE: {metrics['cv_mae_mean']:.4f} ± {metrics['cv_mae_std']:.4f}")

    # Feature importance
    importance = dict(zip(FEATURE_COLS, model.feature_importances_.tolist()))
    importance = dict(sorted(importance.items(), key=lambda x: -x[1]))
    metrics["feature_importance"] = {k: round(v, 4) for k, v in importance.items()}

    print("\nFeature Importance (gain):")
    for feat, score in metrics["feature_importance"].items():
        bar = "#" * int(score * 50)
        print(f"  {feat:20s} {score:.4f}  {bar}")

    return model, X_test, y_test, metrics


def evaluate(y_train, y_pred_train, y_test, y_pred_test):
    """Compute regression metrics for train and test sets."""
    # Test metrics
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_test)))
    mae = float(mean_absolute_error(y_test, y_pred_test))
    r2 = float(r2_score(y_test, y_pred_test))

    # MAPE (avoid division by zero)
    mask = y_test != 0
    mape = float(np.mean(np.abs((y_test[mask] - y_pred_test[mask]) / y_test[mask])) * 100)

    # Train metrics for overfit check
    train_rmse = float(np.sqrt(mean_squared_error(y_train, y_pred_train)))
    train_mae = float(mean_absolute_error(y_train, y_pred_train))

    metrics = {
        "test_rmse": round(rmse, 4),
        "test_mae": round(mae, 4),
        "test_mape": round(mape, 2),
        "test_r2": round(r2, 4),
        "train_rmse": round(train_rmse, 4),
        "train_mae": round(train_mae, 4),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "n_features": len(FEATURE_COLS),
    }

    print(f"\n{'='*50}")
    print(f"  TEST METRICS")
    print(f"{'='*50}")
    print(f"  RMSE  : {rmse:.4f}")
    print(f"  MAE   : {mae:.4f}")
    print(f"  MAPE  : {mape:.2f}%")
    print(f"  R²    : {r2:.4f}")
    print(f"{'='*50}")
    print(f"  TRAIN METRICS (overfit check)")
    print(f"  RMSE  : {train_rmse:.4f}")
    print(f"  MAE   : {train_mae:.4f}")
    print(f"{'='*50}")

    return metrics


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------
def save_model(model, metrics):
    """Save trained model and metrics to disk."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")


def load_model(path=None):
    """Load a trained XGBoost model from disk."""
    path = path or MODEL_PATH
    if not Path(path).exists():
        raise FileNotFoundError(
            f"No trained model at {path}. Run `python -m models.xgboost_model` first."
        )
    model = joblib.load(path)
    print(f"Model loaded from {path}")
    return model


def load_metrics(path=None):
    """Load saved metrics from disk."""
    path = path or METRICS_PATH
    if not Path(path).exists():
        return None
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Prediction Helper
# ---------------------------------------------------------------------------
def predict_demand(model, features: dict) -> float:
    """Predict demand for a single product given its features.

    Args:
        model: Trained XGBoost model
        features: Dict with keys matching FEATURE_COLS
            e.g. {"days_to_expiry": 2, "temperature": 28.5, ...}

    Returns:
        Predicted quantity (float, clipped to >= 0)
    """
    row = pd.DataFrame([{col: features.get(col, 0) for col in FEATURE_COLS}])
    pred = float(model.predict(row)[0])
    return max(0.0, round(pred, 2))


def predict_batch(model, df: pd.DataFrame) -> np.ndarray:
    """Predict demand for multiple rows at once.

    Args:
        model: Trained XGBoost model
        df: DataFrame with FEATURE_COLS columns

    Returns:
        Array of predicted quantities
    """
    X = df[FEATURE_COLS].copy()
    preds = model.predict(X)
    return np.clip(preds, 0, None).round(2)


# ---------------------------------------------------------------------------
# Main (Train pipeline)
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  EcoPrice — XGBoost Demand Forecasting")
    print("=" * 60)
    print()

    # Load
    X, y = load_data()

    # Train
    model, X_test, y_test, metrics = train(X, y)

    # Save
    save_model(model, metrics)

    # Quick sanity test
    print("\n--- Sanity Check: Sample Predictions ---")
    sample = X_test.head(5)
    actuals = y_test.head(5)
    preds = model.predict(sample)
    for i, (idx, row) in enumerate(sample.iterrows()):
        print(
            f"  Expiry={int(row['days_to_expiry'])}d  "
            f"Temp={row['temperature']:.1f}C  "
            f"Price=Rs.{row['price']:.0f}  "
            f"Holiday={'Y' if row['is_holiday'] else 'N'}  "
            f"->  Predicted: {preds[i]:.1f}  |  Actual: {actuals.iloc[i]}"
        )

    print("\n✅ XGBoost model trained and saved successfully!")
    return model, metrics


if __name__ == "__main__":
    main()
