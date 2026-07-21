"""
Project FORESIGHT – FastAPI Service
=====================================
REST API for Demand & Inventory Intelligence.

Endpoints
---------
GET  /health            → Service health & model status
POST /train             → Execute full ML pipeline
POST /predict           → Single-SKU forecast for a date range
POST /predict_batch     → Batch forecast for multiple SKUs
GET  /forecast          → Retrieve forecast_output.csv (filterable by stock_code)
GET  /risk              → Retrieve risk_scores.csv (filterable by risk_level / abc_class)
GET  /recommendation    → Top-N recommendations sorted by priority_score
GET  /model_metrics     → Retrieve model_comparison.csv

Run with:
    uvicorn service.api:app --reload --port 8000
"""

# ──────────────────────────────────────────────────────────────────────────────
# Standard Library
# ──────────────────────────────────────────────────────────────────────────────
import json
import logging
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, List, Optional

# ──────────────────────────────────────────────────────────────────────────────
# Third-Party
# ──────────────────────────────────────────────────────────────────────────────
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

# ──────────────────────────────────────────────────────────────────────────────
# Logging (stdlib for the API layer; loguru is used by src/ modules)
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("foresight.api")

# ──────────────────────────────────────────────────────────────────────────────
# Path Resolution
# service/api.py  →  go one level up  →  foresight root
# ──────────────────────────────────────────────────────────────────────────────
SERVICE_DIR   = Path(__file__).resolve().parent          # …/foresight/service
PROJECT_ROOT  = SERVICE_DIR.parent                       # …/foresight
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR     = PROJECT_ROOT / "models"
SRC_DIR        = PROJECT_ROOT / "src"

# Ensure project root is on sys.path so `src.*` imports work
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ──────────────────────────────────────────────────────────────────────────────
# Known processed files
# ──────────────────────────────────────────────────────────────────────────────
FORECAST_CSV        = DATA_PROCESSED / "forecast_output.csv"
RISK_CSV            = DATA_PROCESSED / "risk_scores.csv"
MODEL_COMPARISON_CSV = DATA_PROCESSED / "model_comparison.csv"
BUSINESS_IMPACT_JSON = DATA_PROCESSED / "business_impact.json"
BEST_MODEL_TXT      = MODELS_DIR / "best_model_name.txt"
FEATURE_LIST_JOB    = MODELS_DIR / "feature_list.joblib"
LABEL_ENC_JOB       = MODELS_DIR / "label_encoders.joblib"

# ──────────────────────────────────────────────────────────────────────────────
# App-Level State (populated during lifespan)
# ──────────────────────────────────────────────────────────────────────────────
_state: Dict[str, Any] = {
    "model": None,
    "model_name": None,
    "feature_list": None,
    "label_encoders": None,
    "model_loaded": False,
    "start_time": None,
}

VERSION = "1.0.0"


# ──────────────────────────────────────────────────────────────────────────────
# Helper: Load Models
# ──────────────────────────────────────────────────────────────────────────────
def _load_models() -> bool:
    """
    Attempt to load the best trained model and supporting artefacts from disk.

    Returns True if loading succeeded, False otherwise.
    """
    try:
        if not BEST_MODEL_TXT.exists():
            log.warning("best_model_name.txt not found – run /train first.")
            return False

        model_name = BEST_MODEL_TXT.read_text().strip()
        model_path = MODELS_DIR / f"model_{model_name}.joblib"

        if not model_path.exists():
            log.warning(f"Model file not found: {model_path}")
            return False

        log.info(f"Loading model '{model_name}' from {model_path} …")
        model = joblib.load(model_path)

        feature_list: Optional[List[str]] = None
        label_encoders: Optional[Dict] = None

        if FEATURE_LIST_JOB.exists():
            feature_list = joblib.load(FEATURE_LIST_JOB)
            log.info(f"Feature list loaded: {len(feature_list)} features")

        if LABEL_ENC_JOB.exists():
            label_encoders = joblib.load(LABEL_ENC_JOB)
            log.info(f"Label encoders loaded: {list(label_encoders.keys())}")

        _state["model"]          = model
        _state["model_name"]     = model_name
        _state["feature_list"]   = feature_list
        _state["label_encoders"] = label_encoders
        _state["model_loaded"]   = True
        log.info(f"✅ Model '{model_name}' ready.")
        return True

    except Exception as exc:
        log.exception(f"Model loading failed: {exc}")
        _state["model_loaded"] = False
        return False


# ──────────────────────────────────────────────────────────────────────────────
# Lifespan Context Manager
# ──────────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup / shutdown lifecycle handler.

    On startup:
      - Records boot timestamp.
      - Attempts to load the trained model from models/ directory.
      - Logs a summary of available processed data files.

    On shutdown:
      - Logs service termination.
    """
    _state["start_time"] = time.time()
    log.info("=" * 60)
    log.info("PROJECT FORESIGHT – API Service starting …")
    log.info(f"  Project root : {PROJECT_ROOT}")
    log.info(f"  Version      : {VERSION}")
    log.info("=" * 60)

    # Check data availability
    for label, path in [
        ("Forecast CSV",         FORECAST_CSV),
        ("Risk CSV",             RISK_CSV),
        ("Model Comparison CSV", MODEL_COMPARISON_CSV),
        ("Business Impact JSON", BUSINESS_IMPACT_JSON),
    ]:
        status_sym = "✅" if path.exists() else "❌"
        log.info(f"  {status_sym} {label}: {path.name}")

    # Load model
    _load_models()

    log.info("Service startup complete. Accepting requests.")
    log.info("-" * 60)

    yield  # ← application runs here

    log.info("PROJECT FORESIGHT – API Service shutting down.")


# ──────────────────────────────────────────────────────────────────────────────
# FastAPI Application
# ──────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Project FORESIGHT – Demand & Inventory Intelligence API",
    description=(
        "REST API for the FORESIGHT ML pipeline. "
        "Provides demand forecasting, inventory risk scoring, and "
        "actionable reorder recommendations for NorthBay Living."
    ),
    version=VERSION,
    contact={
        "name": "FORESIGHT Engineering",
        "email": "foresight@northbayliving.example.com",
    },
    license_info={"name": "Proprietary"},
    lifespan=lifespan,
)

# ──────────────────────────────────────────────────────────────────────────────
# CORS Middleware
# ──────────────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # restrict to specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────────────────────────────────────
# Pydantic Request / Response Models
# ──────────────────────────────────────────────────────────────────────────────

# ── Health ────────────────────────────────────────────────────────────────────
class HealthResponse(BaseModel):
    status: str = Field(..., description="'ok' when the service is running normally.")
    version: str = Field(..., description="API semantic version.")
    uptime_seconds: float = Field(..., description="Seconds since service start.")
    model_loaded: bool = Field(..., description="True if a trained model is in memory.")
    model_name: Optional[str] = Field(None, description="Name of the loaded model.")
    timestamp: str = Field(..., description="ISO-8601 timestamp of this response.")


# ── Train ─────────────────────────────────────────────────────────────────────
class TrainRequest(BaseModel):
    skip_training: bool = Field(
        default=False,
        description=(
            "If True, skips model training and uses existing artefacts. "
            "Runs preprocessing → feature engineering → risk scoring only."
        ),
    )


class TrainResponse(BaseModel):
    status: str
    elapsed_seconds: float
    completed_at: str
    best_model: Optional[str] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)
    pipeline_summary: Dict[str, Any] = Field(default_factory=dict)


# ── Predict (single SKU) ──────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    stock_code: str = Field(..., description="SKU / stock code to forecast.", min_length=1)
    start_date: date = Field(..., description="Forecast start date (inclusive). Format: YYYY-MM-DD.")
    end_date: date = Field(..., description="Forecast end date (inclusive). Format: YYYY-MM-DD.")

    @field_validator("end_date")
    @classmethod
    def end_after_start(cls, v: date, info) -> date:
        start = info.data.get("start_date")
        if start and v < start:
            raise ValueError("end_date must be >= start_date.")
        max_horizon = 365
        if start and (v - start).days > max_horizon:
            raise ValueError(f"Forecast horizon cannot exceed {max_horizon} days.")
        return v


class PredictedDay(BaseModel):
    date: str
    stock_code: str
    forecast_quantity: float


class PredictResponse(BaseModel):
    stock_code: str
    start_date: str
    end_date: str
    model_used: str
    forecast_days: int
    predictions: List[PredictedDay]


# ── Predict Batch ─────────────────────────────────────────────────────────────
class BatchPredictRequest(BaseModel):
    stock_codes: List[str] = Field(
        ...,
        description="List of SKUs to forecast.",
        min_length=1,
    )
    start_date: date = Field(..., description="Forecast start date (inclusive).")
    end_date: date = Field(..., description="Forecast end date (inclusive).")

    @field_validator("end_date")
    @classmethod
    def end_after_start(cls, v: date, info) -> date:
        start = info.data.get("start_date")
        if start and v < start:
            raise ValueError("end_date must be >= start_date.")
        return v

    @field_validator("stock_codes")
    @classmethod
    def limit_batch_size(cls, v: List[str]) -> List[str]:
        if len(v) > 500:
            raise ValueError("Batch size cannot exceed 500 SKUs per request.")
        return v


class BatchPredictResponse(BaseModel):
    model_used: str
    start_date: str
    end_date: str
    sku_count: int
    total_rows: int
    predictions: List[Dict[str, Any]]


# ── Forecast ──────────────────────────────────────────────────────────────────
class ForecastResponse(BaseModel):
    stock_code_filter: Optional[str]
    total_rows: int
    records: List[Dict[str, Any]]


# ── Risk ──────────────────────────────────────────────────────────────────────
class RiskResponse(BaseModel):
    risk_level_filter: Optional[str]
    abc_class_filter: Optional[str]
    total_rows: int
    records: List[Dict[str, Any]]


# ── Recommendation ────────────────────────────────────────────────────────────
class RecommendationItem(BaseModel):
    rank: int
    stock_code: str
    description: Optional[str]
    abc_class: Optional[str]
    risk_level: str
    risk_score: float
    priority_score: float
    recommendation: str
    revenue_at_risk: Optional[float]
    capital_locked: Optional[float]
    coverage_days: Optional[float]
    avg_daily_demand: Optional[float]


class RecommendationResponse(BaseModel):
    top_n: int
    total_skus_in_system: int
    records: List[RecommendationItem]


# ── Model Metrics ─────────────────────────────────────────────────────────────
class ModelMetricsResponse(BaseModel):
    best_model: Optional[str]
    total_models: int
    records: List[Dict[str, Any]]


# ──────────────────────────────────────────────────────────────────────────────
# Custom Exception Handlers
# ──────────────────────────────────────────────────────────────────────────────

@app.exception_handler(404)
async def not_found_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Return a consistent JSON body for 404 Not Found errors."""
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={
            "status": "error",
            "code": 404,
            "message": "The requested resource was not found.",
            "detail": str(exc.detail) if exc.detail else "Not found.",
            "path": str(request.url.path),
        },
    )


@app.exception_handler(422)
async def validation_error_handler(request: Request, exc) -> JSONResponse:
    """Return a consistent JSON body for 422 Unprocessable Entity errors."""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "status": "error",
            "code": 422,
            "message": "Request validation failed. Check the request body or parameters.",
            "detail": exc.errors() if hasattr(exc, "errors") else str(exc),
            "path": str(request.url.path),
        },
    )


@app.exception_handler(500)
async def internal_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Return a consistent JSON body for 500 Internal Server Error."""
    log.exception(f"Unhandled exception at {request.url.path}: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "status": "error",
            "code": 500,
            "message": "An unexpected internal server error occurred.",
            "path": str(request.url.path),
        },
    )


# ──────────────────────────────────────────────────────────────────────────────
# Internal Utility Functions
# ──────────────────────────────────────────────────────────────────────────────

def _assert_file_exists(path: Path, label: str) -> None:
    """Raise HTTP 404 if the given file does not exist."""
    if not path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"{label} not found at '{path.name}'. "
                "Please run POST /train first to generate pipeline outputs."
            ),
        )


def _read_csv_safe(path: Path, label: str, **kwargs) -> pd.DataFrame:
    """Load a CSV file, raising HTTP 404 if missing and HTTP 500 on parse error."""
    _assert_file_exists(path, label)
    try:
        df = pd.read_csv(path, **kwargs)
        return df
    except Exception as exc:
        log.exception(f"Failed to read {label}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to parse {label}: {exc}",
        )


def _df_to_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Convert a DataFrame to a JSON-serialisable list of dicts.
    Handles NaN → None, numpy scalars → Python natives, datetime → ISO strings.
    """
    records = []
    for row in df.to_dict(orient="records"):
        clean: Dict[str, Any] = {}
        for k, v in row.items():
            if isinstance(v, float) and np.isnan(v):
                clean[k] = None
            elif isinstance(v, (np.integer,)):
                clean[k] = int(v)
            elif isinstance(v, (np.floating,)):
                clean[k] = float(v)
            elif isinstance(v, (np.bool_,)):
                clean[k] = bool(v)
            elif isinstance(v, (pd.Timestamp, datetime)):
                clean[k] = v.isoformat()
            else:
                clean[k] = v
        records.append(clean)
    return records


def _assert_model_loaded() -> None:
    """Raise HTTP 503 if no model has been loaded into memory."""
    if not _state["model_loaded"] or _state["model"] is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "No trained model is loaded. "
                "Run POST /train to train and persist a model, "
                "then restart the service or call /train again."
            ),
        )


# ──────────────────────────────────────────────────────────────────────────────
# Prediction Helper (used by /predict and /predict_batch)
# ──────────────────────────────────────────────────────────────────────────────

def _build_prediction_dataframe(
    stock_codes: List[str],
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """
    Build a minimal feature DataFrame for inference.

    Strategy:
      1. If features_engineered.csv exists, use the last known feature row for
         each SKU as the base, then update temporal columns for each future date.
      2. If features file is absent (rare), create a skeleton DataFrame with
         temporal features only.

    Returns a DataFrame with columns aligned to the trained model's feature list.
    """
    model      = _state["model"]
    model_name = _state["model_name"]
    feat_list  = _state["feature_list"]

    date_range = pd.date_range(start=start_date, end=end_date, freq="D")
    features_path = DATA_PROCESSED / "features_engineered.csv"

    # ── Attempt to load historical feature rows as seed ───────────────────────
    if features_path.exists():
        try:
            hist = pd.read_csv(features_path, parse_dates=["date"])
            hist["date"] = pd.to_datetime(hist["date"])
            hist_filtered = hist[hist["stock_code"].isin(stock_codes)]

            if hist_filtered.empty:
                log.warning(
                    f"No historical feature rows found for SKUs: {stock_codes}. "
                    "Using a zero-feature fallback."
                )
                return _build_skeleton_df(stock_codes, date_range, feat_list)

            # Take the latest feature row per SKU as the base template
            seed = hist_filtered.sort_values("date").groupby("stock_code").last().reset_index()

            rows = []
            for _, base_row in seed.iterrows():
                for fdate in date_range:
                    row = base_row.copy()
                    row["date"]         = fdate
                    row["day_of_week"]  = fdate.dayofweek
                    row["day_of_month"] = fdate.day
                    row["month"]        = fdate.month
                    row["week_of_year"] = fdate.isocalendar()[1]
                    row["day_of_year"]  = fdate.timetuple().tm_yday
                    row["is_weekend"]   = int(fdate.weekday() >= 5)
                    row["quantity"]     = 0.0   # target placeholder; not used for inference
                    rows.append(row)

            if not rows:
                return _build_skeleton_df(stock_codes, date_range, feat_list)

            df_future = pd.DataFrame(rows).reset_index(drop=True)
            return df_future

        except Exception as exc:
            log.warning(f"Feature file loading failed ({exc}), using skeleton fallback.")
            return _build_skeleton_df(stock_codes, date_range, feat_list)

    return _build_skeleton_df(stock_codes, date_range, feat_list)


def _build_skeleton_df(
    stock_codes: List[str],
    date_range: pd.DatetimeIndex,
    feat_list: Optional[List[str]],
) -> pd.DataFrame:
    """
    Construct a minimal skeleton DataFrame when historical features are unavailable.
    All numeric features default to 0 so the model can still produce a result.
    """
    rows = []
    for sc in stock_codes:
        for fdate in date_range:
            rows.append({
                "stock_code":   sc,
                "date":         fdate,
                "day_of_week":  fdate.dayofweek,
                "day_of_month": fdate.day,
                "month":        fdate.month,
                "week_of_year": fdate.isocalendar()[1],
                "day_of_year":  fdate.timetuple().tm_yday,
                "is_weekend":   int(fdate.weekday() >= 5),
                "quantity":     0.0,
            })
    df = pd.DataFrame(rows)

    # Pad any feature columns the model expects
    if feat_list:
        for col in feat_list:
            if col not in df.columns:
                df[col] = 0.0

    return df


def _run_model_predict(df_future: pd.DataFrame) -> np.ndarray:
    """
    Run inference using the in-memory model.

    Handles:
      - SeasonalNaive (duck-typed predict with DataFrame)
      - sklearn/LightGBM/XGBoost estimators (need aligned feature matrix)
    """
    from src.forecasting import SeasonalNaive, prepare_features  # lazy import

    model      = _state["model"]
    feat_list  = _state["feature_list"]

    if isinstance(model, SeasonalNaive):
        # SeasonalNaive needs 'day_of_week' and 'stock_code' columns
        preds = model.predict(df_future)
    else:
        X, _ = prepare_features(df_future.copy())
        if feat_list:
            for col in feat_list:
                if col not in X.columns:
                    X[col] = 0.0
            # Keep only trained feature columns, in the trained order
            available = [c for c in feat_list if c in X.columns]
            X = X[available]
        preds = np.clip(model.predict(X), 0.0, None)

    return preds


# ──────────────────────────────────────────────────────────────────────────────
# ENDPOINT 1 – GET /health
# ──────────────────────────────────────────────────────────────────────────────

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Service Health Check",
    tags=["Operations"],
)
async def health() -> HealthResponse:
    """
    Return the current health status of the FORESIGHT API service.

    This endpoint is designed for liveness and readiness probes.
    It reports:

    - **status**: Always `"ok"` when the service is up.
    - **version**: Semantic version of the API.
    - **uptime_seconds**: Seconds elapsed since the service started.
    - **model_loaded**: Whether a trained ML model is currently in memory.
    - **model_name**: The name of the loaded model (e.g. `"LightGBM"`), or `null`.
    - **timestamp**: ISO-8601 UTC timestamp of this response.

    This endpoint never raises an exception – if the service responds at all,
    it is considered healthy.
    """
    uptime = (
        round(time.time() - _state["start_time"], 2)
        if _state["start_time"] is not None
        else 0.0
    )
    return HealthResponse(
        status="ok",
        version=VERSION,
        uptime_seconds=uptime,
        model_loaded=_state["model_loaded"],
        model_name=_state["model_name"],
        timestamp=datetime.utcnow().isoformat() + "Z",
    )


# ──────────────────────────────────────────────────────────────────────────────
# ENDPOINT 2 – POST /train
# ──────────────────────────────────────────────────────────────────────────────

@app.post(
    "/train",
    response_model=TrainResponse,
    summary="Run Full ML Pipeline",
    tags=["Training"],
    status_code=status.HTTP_200_OK,
)
async def train(request: TrainRequest = TrainRequest()) -> TrainResponse:
    """
    Execute the complete FORESIGHT ML pipeline end-to-end.

    Pipeline steps:
    1. **Data Preprocessing** – cleans and normalises raw sales transactions.
    2. **Feature Engineering** – builds lag features, rolling statistics, calendar
       features, and ABC classification.
    3. **Model Training & CV** – runs rolling-origin cross-validation across 5
       models (Seasonal Naive, Random Forest, Gradient Boosting, LightGBM, XGBoost)
       and selects the best by WAPE.  Skipped if `skip_training=true`.
    4. **Risk Scoring** – classifies every SKU by stockout / overstock risk.
    5. **Business Impact** – computes revenue-at-risk and capital-locked metrics.

    After a successful run the service automatically reloads the newly trained
    model so subsequent `/predict` calls use it immediately.

    Returns a summary of pipeline metrics and the best model name.

    **Note**: This is a long-running operation (typically 2–15 minutes depending
    on dataset size). Consider running it as a background task in production.
    """
    try:
        from src.pipeline import run_pipeline  # lazy import keeps startup fast
    except ImportError as exc:
        log.exception("Cannot import pipeline module.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline import failed: {exc}",
        )

    log.info(f"Starting pipeline (skip_training={request.skip_training}) …")
    t0 = time.time()

    try:
        results: Dict[str, Any] = run_pipeline(skip_training=request.skip_training)
    except Exception as exc:
        log.exception(f"Pipeline execution failed: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline execution failed: {exc}",
        )

    elapsed = round(time.time() - t0, 2)
    log.info(f"Pipeline completed in {elapsed}s.")

    # Hot-reload the newly trained model into memory
    reload_ok = _load_models()
    if not reload_ok:
        log.warning("Model reload after training failed – /predict may be unavailable.")

    # Extract metrics for the response
    forecasting_info = results.get("forecasting", {})
    best_model = forecasting_info.get("best_model", _state.get("model_name"))
    cv_metrics = forecasting_info.get("cv_metrics", {})

    return TrainResponse(
        status="success",
        elapsed_seconds=elapsed,
        completed_at=results.get("completed_at", datetime.utcnow().isoformat()),
        best_model=best_model,
        metrics=cv_metrics,
        pipeline_summary=results,
    )


# ──────────────────────────────────────────────────────────────────────────────
# ENDPOINT 3 – POST /predict
# ──────────────────────────────────────────────────────────────────────────────

@app.post(
    "/predict",
    response_model=PredictResponse,
    summary="Single-SKU Demand Forecast",
    tags=["Inference"],
    status_code=status.HTTP_200_OK,
)
async def predict_single(request: PredictRequest) -> PredictResponse:
    """
    Generate a day-by-day demand forecast for a **single SKU** over a specified
    date range.

    The model used is the best model selected during the last training run
    (retrieved from `models/best_model_name.txt`).

    **Request body**:
    - `stock_code`: The SKU identifier (e.g. `"84029E"`).
    - `start_date`: First date of the forecast window (YYYY-MM-DD).
    - `end_date`: Last date of the forecast window (YYYY-MM-DD, inclusive).
      Maximum horizon is 365 days.

    **Response**:
    Returns a list of `{date, stock_code, forecast_quantity}` dicts, one entry
    per day in the requested window.

    Raises:
    - **503** if no trained model is loaded.
    - **500** if inference fails.
    """
    _assert_model_loaded()

    date_range = pd.date_range(start=request.start_date, end=request.end_date, freq="D")
    log.info(
        f"Predict SKU='{request.stock_code}' "
        f"{request.start_date} → {request.end_date} ({len(date_range)} days)"
    )

    try:
        df_future = _build_prediction_dataframe(
            stock_codes=[request.stock_code],
            start_date=request.start_date,
            end_date=request.end_date,
        )

        preds = _run_model_predict(df_future)

        predictions: List[PredictedDay] = []
        for i, (_, row) in enumerate(df_future.iterrows()):
            predictions.append(
                PredictedDay(
                    date=pd.Timestamp(row["date"]).strftime("%Y-%m-%d"),
                    stock_code=str(row["stock_code"]),
                    forecast_quantity=round(float(preds[i]), 4),
                )
            )

    except HTTPException:
        raise
    except Exception as exc:
        log.exception(f"Prediction failed for SKU '{request.stock_code}': {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error: {exc}",
        )

    return PredictResponse(
        stock_code=request.stock_code,
        start_date=str(request.start_date),
        end_date=str(request.end_date),
        model_used=_state["model_name"] or "unknown",
        forecast_days=len(predictions),
        predictions=predictions,
    )


# ──────────────────────────────────────────────────────────────────────────────
# ENDPOINT 4 – POST /predict_batch
# ──────────────────────────────────────────────────────────────────────────────

@app.post(
    "/predict_batch",
    response_model=BatchPredictResponse,
    summary="Batch Demand Forecast for Multiple SKUs",
    tags=["Inference"],
    status_code=status.HTTP_200_OK,
)
async def predict_batch(request: BatchPredictRequest) -> BatchPredictResponse:
    """
    Generate day-by-day demand forecasts for a **list of SKUs** over a specified
    date range in a single request.

    Supports up to 500 SKUs per request. For very large batches (>500 SKUs),
    split into multiple calls.

    **Request body**:
    - `stock_codes`: List of SKU identifiers.
    - `start_date`: First date of the forecast window (YYYY-MM-DD).
    - `end_date`: Last date of the forecast window (YYYY-MM-DD, inclusive).

    **Response**:
    Returns a flat list of `{stock_code, date, forecast_quantity}` records
    covering all requested SKUs and dates.

    Raises:
    - **503** if no trained model is loaded.
    - **500** if inference fails.
    """
    _assert_model_loaded()

    unique_skus = list(dict.fromkeys(request.stock_codes))  # deduplicate, preserve order
    date_range  = pd.date_range(start=request.start_date, end=request.end_date, freq="D")

    log.info(
        f"Batch predict: {len(unique_skus)} SKUs × {len(date_range)} days "
        f"({request.start_date} → {request.end_date})"
    )

    try:
        df_future = _build_prediction_dataframe(
            stock_codes=unique_skus,
            start_date=request.start_date,
            end_date=request.end_date,
        )

        preds = _run_model_predict(df_future)

        records: List[Dict[str, Any]] = []
        for i, (_, row) in enumerate(df_future.iterrows()):
            records.append({
                "stock_code":        str(row["stock_code"]),
                "date":              pd.Timestamp(row["date"]).strftime("%Y-%m-%d"),
                "forecast_quantity": round(float(preds[i]), 4),
            })

    except HTTPException:
        raise
    except Exception as exc:
        log.exception(f"Batch prediction failed: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction error: {exc}",
        )

    return BatchPredictResponse(
        model_used=_state["model_name"] or "unknown",
        start_date=str(request.start_date),
        end_date=str(request.end_date),
        sku_count=len(unique_skus),
        total_rows=len(records),
        predictions=records,
    )


# ──────────────────────────────────────────────────────────────────────────────
# ENDPOINT 5 – GET /forecast
# ──────────────────────────────────────────────────────────────────────────────

@app.get(
    "/forecast",
    response_model=ForecastResponse,
    summary="Retrieve Full Forecast Output",
    tags=["Data"],
    status_code=status.HTTP_200_OK,
)
async def get_forecast(
    stock_code: Optional[str] = Query(
        default=None,
        description="Filter results to a specific SKU. If omitted, all SKUs are returned.",
    ),
) -> ForecastResponse:
    """
    Return the contents of `data/processed/forecast_output.csv` as JSON.

    This file is generated by the pipeline's forecasting step and contains
    28-day-ahead demand forecasts for every SKU that had sufficient history.

    **Query parameters**:
    - `stock_code` *(optional)*: Filter to a single SKU identifier.

    **Response fields per record**:
    - `stock_code`: SKU identifier.
    - `date`: Forecast date (YYYY-MM-DD).
    - `forecast_quantity`: Predicted demand (units, non-negative).

    Raises:
    - **404** if `forecast_output.csv` has not yet been generated (run `/train`).
    - **500** if the CSV cannot be parsed.
    """
    df = _read_csv_safe(FORECAST_CSV, "forecast_output.csv", parse_dates=["date"])

    if stock_code is not None:
        df = df[df["stock_code"].astype(str).str.upper() == stock_code.upper()]

    records = _df_to_records(df)

    return ForecastResponse(
        stock_code_filter=stock_code,
        total_rows=len(records),
        records=records,
    )


# ──────────────────────────────────────────────────────────────────────────────
# ENDPOINT 6 – GET /risk
# ──────────────────────────────────────────────────────────────────────────────

@app.get(
    "/risk",
    response_model=RiskResponse,
    summary="Retrieve Inventory Risk Scores",
    tags=["Data"],
    status_code=status.HTTP_200_OK,
)
async def get_risk(
    risk_level: Optional[str] = Query(
        default=None,
        description=(
            "Filter by risk level. Valid values: "
            "'Critical Stockout Risk', 'High Stockout Risk', "
            "'Moderate Stockout Risk', 'Overstock Risk', "
            "'Mild Overstock Risk', 'Volatile Demand', 'Healthy'."
        ),
    ),
    abc_class: Optional[str] = Query(
        default=None,
        description="Filter by ABC class. Valid values: 'A', 'B', 'C'.",
    ),
) -> RiskResponse:
    """
    Return the contents of `data/processed/risk_scores.csv` as JSON.

    This file is produced by the risk scoring step and includes every SKU's:
    composite risk score, risk classification, inventory coverage, revenue at
    risk, capital locked, and recommended action.

    **Query parameters** (all optional, combinable):
    - `risk_level`: Filter by risk classification string (case-insensitive).
    - `abc_class`: Filter by Pareto class (`A`, `B`, or `C`).

    Results are returned sorted by `priority_score` descending (highest urgency
    first), matching the sort order of the source file.

    Raises:
    - **404** if `risk_scores.csv` has not yet been generated (run `/train`).
    - **500** if the CSV cannot be parsed.
    """
    df = _read_csv_safe(RISK_CSV, "risk_scores.csv")

    if risk_level is not None:
        mask = df["risk_level"].str.lower() == risk_level.lower()
        df = df[mask]

    if abc_class is not None:
        mask = df["abc_class"].astype(str).str.upper() == abc_class.upper()
        df = df[mask]

    records = _df_to_records(df)

    return RiskResponse(
        risk_level_filter=risk_level,
        abc_class_filter=abc_class,
        total_rows=len(records),
        records=records,
    )


# ──────────────────────────────────────────────────────────────────────────────
# ENDPOINT 7 – GET /recommendation
# ──────────────────────────────────────────────────────────────────────────────

@app.get(
    "/recommendation",
    response_model=RecommendationResponse,
    summary="Top-N Inventory Recommendations",
    tags=["Intelligence"],
    status_code=status.HTTP_200_OK,
)
async def get_recommendation(
    top_n: int = Query(
        default=20,
        ge=1,
        le=500,
        description="Number of top recommendations to return (sorted by priority_score desc).",
    ),
    risk_level: Optional[str] = Query(
        default=None,
        description="Pre-filter by risk level before applying top_n.",
    ),
    abc_class: Optional[str] = Query(
        default=None,
        description="Pre-filter by ABC class before applying top_n.",
    ),
) -> RecommendationResponse:
    """
    Return the top-N SKUs ranked by `priority_score` (descending) with their
    associated recommendations.

    `priority_score` = `risk_score` × ABC weight (A=3×, B=2×, C=1×), ensuring
    high-revenue SKUs with supply risk surface first.

    **Query parameters**:
    - `top_n` *(default: 20, max: 500)*: Number of SKUs to return.
    - `risk_level` *(optional)*: Pre-filter to a specific risk category.
    - `abc_class` *(optional)*: Pre-filter to A, B, or C SKUs.

    **Response fields per record**:
    - `rank`: Position in the sorted list (1 = highest urgency).
    - `stock_code`, `description`, `abc_class`: SKU identifiers.
    - `risk_level`: Human-readable risk classification.
    - `risk_score`: Composite risk score (0–100).
    - `priority_score`: Weighted score used for ranking.
    - `recommendation`: Actionable text (e.g. "🚨 Reorder Immediately").
    - `revenue_at_risk`: Estimated revenue exposure (£).
    - `capital_locked`: Estimated capital tied up in excess stock (£).
    - `coverage_days`: Days of stock remaining at average demand.
    - `avg_daily_demand`: Average units sold per day.

    Raises:
    - **404** if `risk_scores.csv` has not yet been generated (run `/train`).
    - **500** if the CSV cannot be parsed.
    """
    df = _read_csv_safe(RISK_CSV, "risk_scores.csv")

    # Apply optional filters
    if risk_level is not None:
        df = df[df["risk_level"].str.lower() == risk_level.lower()]

    if abc_class is not None:
        df = df[df["abc_class"].astype(str).str.upper() == abc_class.upper()]

    total_in_system = len(df)

    # Sort and slice
    if "priority_score" in df.columns:
        df = df.sort_values("priority_score", ascending=False)
    df_top = df.head(top_n).reset_index(drop=True)

    def _safe_float(val) -> Optional[float]:
        try:
            f = float(val)
            return None if np.isnan(f) else round(f, 4)
        except (TypeError, ValueError):
            return None

    items: List[RecommendationItem] = []
    for rank, (_, row) in enumerate(df_top.iterrows(), start=1):
        items.append(
            RecommendationItem(
                rank=rank,
                stock_code=str(row.get("stock_code", "")),
                description=str(row["description"]) if pd.notna(row.get("description")) else None,
                abc_class=str(row["abc_class"]) if pd.notna(row.get("abc_class")) else None,
                risk_level=str(row.get("risk_level", "Unknown")),
                risk_score=_safe_float(row.get("risk_score")) or 0.0,
                priority_score=_safe_float(row.get("priority_score")) or 0.0,
                recommendation=str(row.get("recommendation", "No recommendation available.")),
                revenue_at_risk=_safe_float(row.get("revenue_at_risk")),
                capital_locked=_safe_float(row.get("capital_locked")),
                coverage_days=_safe_float(row.get("coverage_days")),
                avg_daily_demand=_safe_float(row.get("avg_daily_demand")),
            )
        )

    return RecommendationResponse(
        top_n=top_n,
        total_skus_in_system=total_in_system,
        records=items,
    )


# ──────────────────────────────────────────────────────────────────────────────
# ENDPOINT 8 – GET /model_metrics
# ──────────────────────────────────────────────────────────────────────────────

@app.get(
    "/model_metrics",
    response_model=ModelMetricsResponse,
    summary="Model Comparison Metrics",
    tags=["Training"],
    status_code=status.HTTP_200_OK,
)
async def get_model_metrics() -> ModelMetricsResponse:
    """
    Return the cross-validation performance metrics for all evaluated models.

    The metrics are sourced from `data/processed/model_comparison.csv`, which
    is written by the forecasting pipeline after rolling-origin CV is complete.

    **Response fields per model record**:
    - `model`: Model name (e.g. `"LightGBM"`, `"XGBoost"`, `"SeasonalNaive"`).
    - `WAPE`: Weighted Absolute Percentage Error (primary selection metric, %).
    - `MAPE`: Mean Absolute Percentage Error (%).
    - `MAE`: Mean Absolute Error (units).
    - `RMSE`: Root Mean Squared Error (units).
    - `Bias`: Signed forecast bias (+ = over-forecast, − = under-forecast).
    - `rank`: Rank by WAPE (1 = best).

    The `best_model` field in the response body indicates which model was
    selected by the pipeline (from `models/best_model_name.txt`).

    Raises:
    - **404** if `model_comparison.csv` has not yet been generated (run `/train`).
    - **500** if the CSV cannot be parsed.
    """
    df = _read_csv_safe(MODEL_COMPARISON_CSV, "model_comparison.csv")

    # Determine which model was selected
    best_model: Optional[str] = None
    if BEST_MODEL_TXT.exists():
        best_model = BEST_MODEL_TXT.read_text().strip() or None

    records = _df_to_records(df)

    return ModelMetricsResponse(
        best_model=best_model,
        total_models=len(records),
        records=records,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Root redirect
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root() -> JSONResponse:
    """Redirect root path to the API documentation."""
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "message": "Project FORESIGHT API is running.",
            "docs": "/docs",
            "redoc": "/redoc",
            "health": "/health",
            "version": VERSION,
        },
    )
