"""EcoPrice XGBoost v2 — Improved demand forecaster.

Key improvements over v1:
1. Daily aggregation per (sub_category, postal_code) — reduces noise
2. Lag features — yesterday's sales, 3d/7d/14d rolling averages
3. Time-series aware split (last 20% of dates, not random)
4. Category encoding + price normalization
5. More trees + early stopping to prevent overfitting

Usage:
    python -m models.xgboost_v2
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models" / "saved"
CLEAN_CSV = DATA_DIR / "ecoprice_clean.csv"
MODEL_PATH = MODEL_DIR / "xgboost_demand_v2.joblib"
METRICS_PATH = MODEL_DIR / "xgboost_metrics_v2.json"
ENCODER_PATH = MODEL_DIR / "label_encoders_v2.joblib"
FEATURE_MATRIX_V2 = DATA_DIR / "feature_matrix_v2.csv"


def build_daily_features():
    """Aggregate to daily level and engineer lag/rolling features."""
    print("Loading clean data...")
    df = pd.read_csv(CLEAN_CSV)
    df["sales_date"] = pd.to_datetime(df["sales_date"])

    # Filter perishables only
    df = df[df["is_perishable"] == 1].copy()
    print(f"Perishable rows: {len(df):,}")

    # Daily aggregation per (sub_category, postal_code)
    daily = (
        df.groupby(["sales_date", "sub_category", "postal_code"])
        .agg(
            daily_qty=("quantity", "sum"),
            avg_price=("price", "mean"),
            n_transactions=("order_id", "count"),
            avg_discount=("discount", "mean"),
            avg_days_to_expiry=("days_to_expiry", "mean"),
            temperature=("temperature", "first"),
            humidity=("humidity", "first"),
            precipitation=("precipitation", "first"),
            is_weekend=("is_weekend", "first"),
            is_holiday=("is_holiday", "first"),
            days_to_festival=("days_to_festival", "first"),
        )
        .reset_index()
        .sort_values(["sub_category", "postal_code", "sales_date"])
    )

    print(f"Daily aggregated: {len(daily):,} rows")

    # --- Temporal features ---
    daily["day_of_week"] = daily["sales_date"].dt.dayofweek
    daily["month"] = daily["sales_date"].dt.month
    daily["day_of_month"] = daily["sales_date"].dt.day
    daily["week_of_year"] = daily["sales_date"].dt.isocalendar().week.astype(int)
    daily["quarter"] = daily["sales_date"].dt.quarter

    # Is it start/end of month? (payday effect)
    daily["is_month_start"] = (daily["day_of_month"] <= 5).astype(int)
    daily["is_month_end"] = (daily["day_of_month"] >= 25).astype(int)

    # --- Lag features (per sub_category + postal_code group) ---
    group = daily.groupby(["sub_category", "postal_code"])["daily_qty"]

    # Yesterday's sales
    daily["lag_1d"] = group.shift(1)
    # 2 days ago
    daily["lag_2d"] = group.shift(2)
    # 7 days ago (same day last week)
    daily["lag_7d"] = group.shift(7)

    # Rolling averages (no leakage — shift first)
    daily["rolling_3d"] = group.transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).mean()
    )
    daily["rolling_7d"] = group.transform(
        lambda s: s.shift(1).rolling(7, min_periods=1).mean()
    )
    daily["rolling_14d"] = group.transform(
        lambda s: s.shift(1).rolling(14, min_periods=1).mean()
    )

    # Rolling std (volatility)
    daily["rolling_7d_std"] = group.transform(
        lambda s: s.shift(1).rolling(7, min_periods=2).std()
    )

    # --- Price features ---
    # Price relative to category average
    cat_avg_price = daily.groupby("sub_category")["avg_price"].transform("mean")
    daily["price_ratio"] = daily["avg_price"] / (cat_avg_price + 1e-6)

    # --- Encode sub_category ---
    le = LabelEncoder()
    daily["category_encoded"] = le.fit_transform(daily["sub_category"])

    # --- Drop rows with NaN from lag features (first few days per group) ---
    daily = daily.dropna(subset=["lag_1d", "rolling_7d"]).copy()

    print(f"After lag feature creation: {len(daily):,} rows")

    # Define feature columns
    feature_cols = [
        "category_encoded",
        "avg_price",
        "price_ratio",
        "avg_discount",
        "avg_days_to_expiry",
        "n_transactions",
        "temperature",
        "humidity",
        "precipitation",
        "day_of_week",
        "month",
        "day_of_month",
        "week_of_year",
        "quarter",
        "is_weekend",
        "is_holiday",
        "is_month_start",
        "is_month_end",
        "days_to_festival",
        "lag_1d",
        "lag_2d",
        "lag_7d",
        "rolling_3d",
        "rolling_7d",
        "rolling_14d",
        "rolling_7d_std",
    ]

    target_col = "daily_qty"

    # Save feature matrix v2
    save_df = daily[feature_cols + [target_col, "sales_date"]].copy()
    save_df.to_csv(FEATURE_MATRIX_V2, index=False)
    print(f"Saved feature matrix v2: {FEATURE_MATRIX_V2}")

    return daily, feature_cols, target_col, le


def train_v2():
    """Train XGBoost v2 with improved features and time-series split."""
    print("=" * 60)
    print("  EcoPrice -- XGBoost v2 (Improved)")
    print("=" * 60)

    daily, feature_cols, target_col, label_encoder = build_daily_features()

    X = daily[feature_cols]
    y = daily[target_col]

    print(f"\nFeatures: {len(feature_cols)}")
    print(f"Target range: {y.min()} - {y.max()}, mean: {y.mean():.2f}, std: {y.std():.2f}")

    # --- Time-series split (last 20% of dates, NOT random) ---
    dates = daily["sales_date"].sort_values().unique()
    split_date = dates[int(len(dates) * 0.8)]
    train_mask = daily["sales_date"] < split_date
    test_mask = daily["sales_date"] >= split_date

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    print(f"\nTime-series split at: {split_date}")
    print(f"Train: {len(X_train):,} rows (before {split_date})")
    print(f"Test:  {len(X_test):,} rows (from {split_date} onwards)")

    # --- XGBoost with early stopping ---
    params = {
        "n_estimators": 1000,
        "max_depth": 7,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
        "tree_method": "hist",
        "early_stopping_rounds": 50,
    }

    print("\nTraining XGBoost v2...")
    model = xgb.XGBRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=100,
    )

    best_iter = model.best_iteration
    print(f"Best iteration: {best_iter}")

    # --- Evaluate ---
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    test_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_test)))
    test_mae = float(mean_absolute_error(y_test, y_pred_test))
    test_r2 = float(r2_score(y_test, y_pred_test))
    mask = y_test != 0
    test_mape = float(np.mean(np.abs((y_test[mask] - y_pred_test[mask]) / y_test[mask])) * 100)

    train_rmse = float(np.sqrt(mean_squared_error(y_train, y_pred_train)))
    train_r2 = float(r2_score(y_train, y_pred_train))

    print(f"\n{'='*50}")
    print(f"  TEST METRICS (v2)")
    print(f"{'='*50}")
    print(f"  RMSE  : {test_rmse:.4f}")
    print(f"  MAE   : {test_mae:.4f}")
    print(f"  MAPE  : {test_mape:.2f}%")
    print(f"  R2    : {test_r2:.4f}")
    print(f"{'='*50}")
    print(f"  TRAIN METRICS")
    print(f"  RMSE  : {train_rmse:.4f}")
    print(f"  R2    : {train_r2:.4f}")
    print(f"{'='*50}")

    # --- Feature importance ---
    importance = dict(zip(feature_cols, model.feature_importances_.tolist()))
    importance = dict(sorted(importance.items(), key=lambda x: -x[1]))

    print("\nTop 10 Feature Importance:")
    for i, (feat, score) in enumerate(importance.items()):
        if i >= 10:
            break
        bar = "#" * int(score * 80)
        print(f"  {feat:22s} {score:.4f}  {bar}")

    # --- Save ---
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(label_encoder, ENCODER_PATH)

    metrics = {
        "version": "v2",
        "test_rmse": round(test_rmse, 4),
        "test_mae": round(test_mae, 4),
        "test_mape": round(test_mape, 2),
        "test_r2": round(test_r2, 4),
        "train_rmse": round(train_rmse, 4),
        "train_r2": round(train_r2, 4),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_features": len(feature_cols),
        "best_iteration": int(best_iter),
        "split_date": str(split_date),
        "feature_importance": {k: round(v, 4) for k, v in importance.items()},
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")

    # --- Comparison with v1 ---
    v1_path = MODEL_DIR / "xgboost_metrics.json"
    if v1_path.exists():
        with open(v1_path) as f:
            v1 = json.load(f)
        print(f"\n{'='*50}")
        print(f"  v1 vs v2 COMPARISON")
        print(f"{'='*50}")
        print(f"  {'Metric':<12s} {'v1':>10s} {'v2':>10s} {'Change':>10s}")
        print(f"  {'-'*42}")
        print(f"  {'R2':<12s} {v1['test_r2']:>10.4f} {test_r2:>10.4f} {test_r2-v1['test_r2']:>+10.4f}")
        print(f"  {'MAE':<12s} {v1['test_mae']:>10.4f} {test_mae:>10.4f} {test_mae-v1['test_mae']:>+10.4f}")
        print(f"  {'MAPE':<12s} {v1['test_mape']:>9.2f}% {test_mape:>9.2f}% {test_mape-v1['test_mape']:>+9.2f}%")
        print(f"  {'Features':<12s} {v1['n_features']:>10d} {len(feature_cols):>10d}")
        print(f"{'='*50}")

    return model, metrics


if __name__ == "__main__":
    train_v2()
