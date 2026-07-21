"""
Project FORESIGHT – Forecasting Module
========================================
Implements 6 forecasting models with Rolling-Origin Cross Validation:

  1. Seasonal Naive Baseline
  2. Random Forest
  3. Gradient Boosting (sklearn)
  4. LightGBM
  5. XGBoost
  6. Prophet (per-SKU, top SKUs only)

Model selection: automatic, based on validation WAPE.
"""

import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import xgboost as xgb

from src.utils import logger, timer, safe_run, CONFIG, MODELS_DIR, DATA_PROCESSED, load_csv, save_csv
from src.evaluation import compute_all_metrics, select_best_model, compare_models

warnings.filterwarnings("ignore")

FEATURE_IMPORTANCE: Dict[str, pd.DataFrame] = {}


# ─────────────────────────────────────────────
# Rolling-Origin Cross Validation
# ─────────────────────────────────────────────
def rolling_origin_splits(
    df: pd.DataFrame,
    n_folds: int = 3,
    forecast_horizon: int = 28,
    min_train_days: int = 90,
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Generate (train, test) splits using rolling-origin evaluation.
    Each fold shifts the origin forward by one horizon.
    Avoids data leakage by only using past data for training.
    """
    dates = sorted(df["date"].unique())
    n_dates = len(dates)
    splits = []

    test_end_idx = n_dates - 1
    for fold in range(n_folds):
        test_start_idx = test_end_idx - forecast_horizon + 1
        train_end_idx  = test_start_idx - 1

        if train_end_idx < min_train_days:
            logger.warning(f"Fold {fold+1}: insufficient training data, skipping.")
            break

        train_dates = dates[:train_end_idx + 1]
        test_dates  = dates[test_start_idx:test_end_idx + 1]

        train = df[df["date"].isin(train_dates)].copy()
        test  = df[df["date"].isin(test_dates)].copy()

        splits.append((train, test))
        logger.debug(f"Fold {fold+1}: train {train_dates[0]}→{train_dates[-1]} | test {test_dates[0]}→{test_dates[-1]}")

        test_end_idx = test_start_idx - 1

    logger.info(f"Rolling-Origin CV: {len(splits)} folds")
    return list(reversed(splits))


# ─────────────────────────────────────────────
# Feature Preparation
# ─────────────────────────────────────────────
_EXCLUDE_COLS = {
    "stock_code", "date", "description", "quantity",
    "month_name", "day_name", "season", "abc_class",
    "category", "supplier", "warehouse", "invoice_no",
    "num_orders", "num_customers", "revenue",
    "first_sale_date", "last_sale_date",
}

_LABEL_ENCODERS: Dict[str, LabelEncoder] = {}


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Return X (features) and y (target) from the engineered DataFrame."""
    target = "quantity"
    feature_cols = [
        c for c in df.columns
        if c not in _EXCLUDE_COLS
        and df[c].dtype in [np.float64, np.float32, np.int64, np.int32, np.int8, bool, np.uint8]
    ]

    # Encode any remaining object columns
    for col in df.columns:
        if col not in _EXCLUDE_COLS and df[col].dtype == object:
            if col not in _LABEL_ENCODERS:
                le = LabelEncoder()
                df[col + "_enc"] = le.fit_transform(df[col].astype(str))
                _LABEL_ENCODERS[col] = le
            else:
                le = _LABEL_ENCODERS[col]
                df[col + "_enc"] = le.transform(df[col].astype(str))
            feature_cols.append(col + "_enc")

    X = df[feature_cols].fillna(0)
    y = df[target].clip(lower=0)
    return X, y


# ─────────────────────────────────────────────
# Model 1: Seasonal Naive Baseline
# ─────────────────────────────────────────────
class SeasonalNaive:
    """Seasonal Naive: predict = value from 7 days ago."""
    name = "SeasonalNaive"

    def fit(self, train: pd.DataFrame):
        self._history = (
            train.groupby(["stock_code", "day_of_week"])["quantity"].mean().reset_index()
        )
        self._sku_mean = train.groupby("stock_code")["quantity"].mean().to_dict()
        return self

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        df = test.merge(
            self._history.rename(columns={"quantity": "naive_pred"}),
            on=["stock_code", "day_of_week"],
            how="left",
        )
        df["naive_pred"] = df["naive_pred"].fillna(
            df["stock_code"].map(self._sku_mean).fillna(0)
        )
        return df["naive_pred"].clip(lower=0).values


# ─────────────────────────────────────────────
# Model 2–5: Tree-Based Models
# ─────────────────────────────────────────────
MODEL_CONFIGS = {
    "RandomForest": RandomForestRegressor(
        n_estimators=20,
        max_depth=6,
        min_samples_leaf=5,
        n_jobs=1,
        random_state=CONFIG["RANDOM_SEED"],
    ),
    "GradientBoosting": GradientBoostingRegressor(
        n_estimators=20,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        random_state=CONFIG["RANDOM_SEED"],
    ),
    "LightGBM": lgb.LGBMRegressor(
        n_estimators=20,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        n_jobs=1,
        random_state=CONFIG["RANDOM_SEED"],
        verbose=-1,
    ),
    "XGBoost": xgb.XGBRegressor(
        n_estimators=20,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        n_jobs=1,
        random_state=CONFIG["RANDOM_SEED"],
        verbosity=0,
    ),
}


# ─────────────────────────────────────────────
# Model 6: Prophet (top SKUs)
# ─────────────────────────────────────────────
def _fit_prophet_sku(
    sku_df: pd.DataFrame,
    forecast_horizon: int,
) -> Tuple[float, np.ndarray]:
    """Fit Prophet on a single SKU time series. Returns WAPE and predictions."""
    try:
        from prophet import Prophet
        prophet_df = sku_df[["date", "quantity"]].rename(columns={"date": "ds", "quantity": "y"})
        prophet_df["ds"] = pd.to_datetime(prophet_df["ds"])
        prophet_df["y"] = prophet_df["y"].clip(lower=0)

        if len(prophet_df) < 30:
            return float("inf"), np.zeros(forecast_horizon)

        split_idx = max(10, len(prophet_df) - forecast_horizon)
        train_p = prophet_df.iloc[:split_idx]
        test_p  = prophet_df.iloc[split_idx:]

        m = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10,
        )
        m.fit(train_p)
        future = m.make_future_dataframe(periods=len(test_p), freq="D")
        forecast = m.predict(future)
        preds = forecast.tail(len(test_p))["yhat"].clip(lower=0).values

        from src.evaluation import wape as _wape
        w = _wape(test_p["y"].values, preds)
        return w, preds
    except Exception as e:
        logger.warning(f"Prophet failed for SKU: {e}")
        return float("inf"), np.zeros(forecast_horizon)


# ─────────────────────────────────────────────
# Cross-Validation Runner
# ─────────────────────────────────────────────
@timer
def cross_validate_models(
    df: pd.DataFrame,
    n_folds: int = CONFIG["CV_FOLDS"],
    forecast_horizon: int = CONFIG["FORECAST_HORIZON"],
) -> Dict[str, Dict]:
    """
    Run rolling-origin cross-validation for all models.
    Returns dict: model_name → {metrics, predictions}.
    """
    from src.evaluation import compute_all_metrics, wape as _wape

    splits = rolling_origin_splits(df, n_folds=n_folds, forecast_horizon=forecast_horizon)
    results = {name: {"metrics_per_fold": [], "all_preds": [], "all_actuals": []}
               for name in list(MODEL_CONFIGS.keys()) + ["SeasonalNaive"]}

    for fold_idx, (train, test) in enumerate(splits):
        logger.info(f"━━ Fold {fold_idx+1}/{len(splits)} ━━")

        X_train, y_train = prepare_features(train)
        X_test, y_test   = prepare_features(test)

        # Seasonal Naive
        naive = SeasonalNaive().fit(train)
        naive_preds = naive.predict(test)
        results["SeasonalNaive"]["all_preds"].extend(naive_preds.tolist())
        results["SeasonalNaive"]["all_actuals"].extend(y_test.tolist())
        results["SeasonalNaive"]["metrics_per_fold"].append(
            compute_all_metrics(y_test.values, naive_preds)
        )
        logger.info(f"  SeasonalNaive WAPE: {_wape(y_test.values, naive_preds)*100:.2f}%")

        # Tree-based models
        for model_name, model in MODEL_CONFIGS.items():
            try:
                model.fit(X_train, y_train)
                preds = np.clip(model.predict(X_test), 0, None)
                metrics = compute_all_metrics(y_test.values, preds)
                results[model_name]["metrics_per_fold"].append(metrics)
                results[model_name]["all_preds"].extend(preds.tolist())
                results[model_name]["all_actuals"].extend(y_test.tolist())
                logger.info(f"  {model_name} WAPE: {metrics['WAPE']:.2f}%")
            except Exception as e:
                logger.error(f"  {model_name} failed: {e}")

    # Aggregate metrics across folds
    summary = {}
    for model_name, data in results.items():
        if not data["metrics_per_fold"]:
            continue
        agg = {}
        for metric in ["WAPE", "MAPE", "MAE", "RMSE", "Bias"]:
            vals = [f[metric] for f in data["metrics_per_fold"] if metric in f]
            agg[metric] = round(float(np.mean(vals)), 4) if vals else 0.0
        summary[model_name] = agg

    return summary


# ─────────────────────────────────────────────
# Final Training (Full Data)
# ─────────────────────────────────────────────
@timer
def train_best_model(
    df: pd.DataFrame,
    model_name: str,
) -> Any:
    """Train the selected model on the full dataset and persist it."""
    logger.info(f"Training final model: {model_name} on full dataset…")
    X, y = prepare_features(df)

    if model_name == "SeasonalNaive":
        model = SeasonalNaive().fit(df)
    else:
        model = MODEL_CONFIGS[model_name]
        model.fit(X, y)

        # Feature importance
        try:
            fi = pd.DataFrame({
                "feature": X.columns.tolist(),
                "importance": model.feature_importances_,
            }).sort_values("importance", ascending=False)
            FEATURE_IMPORTANCE[model_name] = fi
            fi_path = MODELS_DIR / f"feature_importance_{model_name}.csv"
            fi.to_csv(fi_path, index=False)
            logger.info(f"  Top feature: {fi.iloc[0]['feature']} ({fi.iloc[0]['importance']:.4f})")
        except AttributeError:
            pass

    # Save model
    model_path = MODELS_DIR / f"model_{model_name}.joblib"
    joblib.dump(model, model_path)
    logger.info(f"  Model saved: {model_path}")

    # Save feature list
    feat_list = X.columns.tolist() if model_name != "SeasonalNaive" else []
    joblib.dump(feat_list, MODELS_DIR / "feature_list.joblib")
    joblib.dump(_LABEL_ENCODERS, MODELS_DIR / "label_encoders.joblib")

    return model


# ─────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────
def predict(
    df: pd.DataFrame,
    model_name: str = None,
    model=None,
) -> pd.DataFrame:
    """Generate predictions on provided DataFrame."""
    if model is None:
        if model_name is None:
            model_name = _load_best_model_name()
        model_path = MODELS_DIR / f"model_{model_name}.joblib"
        model = joblib.load(model_path)

    if isinstance(model, SeasonalNaive):
        preds = model.predict(df)
    else:
        feature_list = joblib.load(MODELS_DIR / "feature_list.joblib")
        X, _ = prepare_features(df)
        # Align features
        missing = [f for f in feature_list if f not in X.columns]
        for f in missing:
            X[f] = 0
        X = X[feature_list]
        preds = np.clip(model.predict(X), 0, None)

    result = df[["stock_code", "date"]].copy()
    result["predicted_quantity"] = preds
    return result


def _load_best_model_name() -> str:
    """Load saved best model name."""
    name_path = MODELS_DIR / "best_model_name.txt"
    if name_path.exists():
        return name_path.read_text().strip()
    raise FileNotFoundError("best_model_name.txt not found. Run training first.")


# ─────────────────────────────────────────────
# Forecast Generation (future dates)
# ─────────────────────────────────────────────
@timer
def generate_forecast(
    df: pd.DataFrame,
    model,
    model_name: str,
    horizon: int = CONFIG["FORECAST_HORIZON"],
) -> pd.DataFrame:
    """
    Generate multi-step forecast by iteratively predicting future dates.
    Uses recursive strategy: each predicted value feeds into lag features.
    """
    logger.info(f"Generating {horizon}-day forecast…")
    last_date = pd.to_datetime(df["date"].max())
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")

    forecasts = []
    # For tree models, extend history iteratively
    df_extended = df.copy()
    if "quantity_lag1" in df_extended.columns:
        df_extended["quantity_lag1"] = df_extended["quantity_lag1"].astype("float64")

    for fdate in future_dates:
        # Create a row for each SKU at the future date
        from src.feature_engineering import run_feature_engineering
        # Simplified: use last available feature row + update date
        future_rows = df_extended.groupby("stock_code").last().reset_index()
        future_rows["date"] = fdate

        # Update temporal features
        future_rows["day_of_week"]  = fdate.dayofweek
        future_rows["day_of_month"] = fdate.day
        future_rows["month"]        = fdate.month
        future_rows["week_of_year"] = fdate.isocalendar()[1]
        future_rows["day_of_year"]  = fdate.timetuple().tm_yday
        future_rows["is_weekend"]   = int(fdate.weekday() >= 5)

        pred_df = predict(future_rows, model_name=model_name, model=model)
        pred_df["date"] = fdate
        forecasts.append(pred_df)

        # Update lag-1 with the new prediction
        lag1_update = pred_df.set_index("stock_code")["predicted_quantity"]
        df_extended.loc[df_extended["stock_code"].isin(lag1_update.index), "quantity_lag1"] = (
            df_extended["stock_code"].map(lag1_update.to_dict())
        )

    forecast_df = pd.concat(forecasts, ignore_index=True)
    forecast_df.columns = ["stock_code", "date", "forecast_quantity"]
    forecast_df["forecast_quantity"] = forecast_df["forecast_quantity"].clip(lower=0)

    out_path = DATA_PROCESSED / "forecast_output.csv"
    save_csv(forecast_df, out_path)
    logger.info(f"✅ Forecast saved: {len(forecast_df):,} rows")
    return forecast_df


# ─────────────────────────────────────────────
# Full Forecasting Pipeline
# ─────────────────────────────────────────────
@timer
@safe_run
def run_forecasting(
    features_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    End-to-end forecasting pipeline:
    1. Load features
    2. Run CV for all models
    3. Select best model
    4. Train on full data
    5. Generate future forecast
    """
    if features_df is None:
        features_df = load_csv(DATA_PROCESSED / "features_engineered.csv", parse_dates=["date"])

    features_df["date"] = pd.to_datetime(features_df["date"])
    
    # Memory optimization: Limit history for modeling to 120 days
    last_dt = features_df["date"].max()
    features_df = features_df[features_df["date"] >= last_dt - pd.Timedelta(days=120)]
    logger.info(f"Subsetting data for modeling to {len(features_df):,} rows (last 120 days)")

    # Filter SKUs with sufficient history
    sku_counts = features_df.groupby("stock_code")["date"].count()
    valid_skus = sku_counts[sku_counts >= CONFIG["MIN_HISTORY_DAYS"]].index
    logger.info(f"SKUs with ≥{CONFIG['MIN_HISTORY_DAYS']} days history: {len(valid_skus):,}/{sku_counts.shape[0]:,}")

    df_model = features_df[features_df["stock_code"].isin(valid_skus)].copy()

    # CV
    cv_results = cross_validate_models(df_model)

    # Compare and select
    comparison_df = compare_models(cv_results)
    comparison_df.to_csv(DATA_PROCESSED / "model_comparison.csv", index=False)
    logger.info(f"\n{comparison_df.to_string(index=False)}")

    best_model_name = select_best_model(cv_results)
    (MODELS_DIR / "best_model_name.txt").write_text(best_model_name)

    # Final training
    best_model = train_best_model(df_model, best_model_name)

    # Forecast
    forecast_df = generate_forecast(df_model, best_model, best_model_name)

    return {
        "cv_results": cv_results,
        "comparison_df": comparison_df,
        "best_model_name": best_model_name,
        "best_model": best_model,
        "forecast_df": forecast_df,
    }


if __name__ == "__main__":
    run_forecasting()
