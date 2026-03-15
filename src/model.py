"""
Model training, evaluation, and prediction for hydropower generation forecasting.

Uses XGBoost with time-series aware cross-validation (no future data leakage).
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

from src.data_generator import (
    DISTRICTS,
    HYDRO_PLANTS,
    _river_flow,
    _hydro_generation,
)
from src.data_processing import (
    add_cumulative_rainfall,
    add_lag_features,
    add_rolling_features,
    add_temporal_features,
)


def _prepare_from_nepali_weather(
    dataset_path: str,
    target_col: str,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Build a training dataframe directly from the REAL weather dataset:
    `data/nepali_multi_district.csv`.

    We:
    - parse dates and districts
    - map key columns to a simpler schema (rainfall, temperature, humidity)
    - add temporal + lag + rolling features (time-series aware)
    """
    df = pd.read_csv(dataset_path, parse_dates=["Date"])

    # Standardize core columns used by existing feature pipeline
    rename_map = {
        "Date": "date",
        "District": "district",
        "Precip": "rainfall_mm",
        "Temp_2m": "temperature_c",
        "RH_2m": "humidity_pct",
    }
    df = df.rename(columns=rename_map)

    # ─── SYNTHETIC TARGET GENERATION ──────────────────────────────────────────
    # We must generate synthetic river flow and hydropower targets because
    # the real weather dataset does not have them.
    
    rows = []
    # Process each district separately to apply correct elevation/hydro params
    grid_districts = df["district"].unique()
    
    for district in grid_districts:
        # Skip districts we don't have metadata for
        if district not in DISTRICTS:
            continue
            
        subset = df[df["district"] == district].copy().sort_values("date")
        props = DISTRICTS[district]
        
        # Calculate synthetic physics-based flow
        flow = _river_flow(
            subset["rainfall_mm"].values,
            subset["temperature_c"].values,
            props["elevation"]
        )
        subset["river_flow_cumecs"] = flow
        
        # Calculate synthetic generation
        river_name = props["river"]
        plant = HYDRO_PLANTS[river_name]
        gen = _hydro_generation(flow, plant["capacity_mw"])
        subset["generation_mw"] = gen
        
        rows.append(subset)
        
    if not rows:
        raise ValueError("No matching districts found between CSV and DISTRICTS config")
        
    df = pd.concat(rows).reset_index(drop=True)
    # ──────────────────────────────────────────────────────────────────────────

    # Drop any rows missing the supervised target
    df = df.dropna(subset=[target_col]).reset_index(drop=True)

    # Reuse temporal + statistical feature engineering from data_processing
    df = add_temporal_features(df, date_col="date")
    df = add_lag_features(df, target_col="generation_mw")
    df = add_rolling_features(df)
    df = add_cumulative_rainfall(df)

    # Final clean-up
    df = df.dropna().reset_index(drop=True)

    exclude_cols = {
        "date",
        "district",
        target_col,
    }
    feature_cols: list[str] = []
    for col in df.columns:
        if col in exclude_cols:
            continue
        if np.issubdtype(df[col].dtype, np.number):
            feature_cols.append(col)

    return df, feature_cols


def train_model(
    dataset_path: str = "data/nepali_multi_district.csv",
    model_dir: str = "models",
    target_col: str = "generation_mw",
) -> dict:
    """
    Train XGBoost model using the NEW real-world dataset in `data/`.

    The model learns to predict `generation_mw` from current-day
    weather and engineered time-series features.
    """
    print("Preparing training data from data/nepali_multi_district.csv ...")
    df, feature_cols = _prepare_from_nepali_weather(dataset_path, target_col)
    print(f"  Features: {len(feature_cols)}, Samples: {len(df):,}")

    X = df[feature_cols].values
    y = df[target_col].values

    # Time-series split (no shuffling — prevents future leakage)
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        preds = model.predict(X_val)
        mae = mean_absolute_error(y_val, preds)
        cv_scores.append(mae)
        print(f"  Fold {fold + 1}: MAE = {mae:.3f}")

    print(f"  Mean CV MAE: {np.mean(cv_scores):.3f}")

    # Final model on all data
    print("Training final model on full dataset...")
    final_model = xgb.XGBRegressor(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
    )
    final_model.fit(X, y, verbose=False)

    y_pred = final_model.predict(X)
    metrics = {
        "mae": round(mean_absolute_error(y, y_pred), 4),
        "rmse": round(np.sqrt(mean_squared_error(y, y_pred)), 4),
        "r2": round(r2_score(y, y_pred), 4),
        "cv_mae_mean": round(np.mean(cv_scores), 4),
        "cv_mae_std": round(np.std(cv_scores), 4),
        "n_features": len(feature_cols),
        "n_samples": len(df),
    }

    # Feature importance
    importance = dict(zip(feature_cols, final_model.feature_importances_.tolist()))
    top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:15]

    out = Path(model_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, out / "xgb_hydro_model.joblib")
    joblib.dump(feature_cols, out / "feature_columns.joblib")

    with open(out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(out / "feature_importance.json", "w") as f:
        json.dump(dict(top_features), f, indent=2)

    print(f"\nModel saved to {out}/")
    print(f"  R²: {metrics['r2']:.4f}")
    print(f"  MAE: {metrics['mae']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print("\nTop 5 features:")
    for name, imp in top_features[:5]:
        print(f"  {name}: {imp:.4f}")

    return metrics


if __name__ == "__main__":
    train_model()
