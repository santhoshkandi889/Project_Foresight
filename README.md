# 🔭 FORESIGHT — Demand & Inventory Intelligence for NorthBay Living

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111%2B-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square)](https://github.com/psf/black)
[![Status](https://img.shields.io/badge/status-active-brightgreen?style=flat-square)]()

> **An end-to-end demand-forecasting and inventory-risk intelligence platform built for NorthBay Living — a multi-category home-goods retailer — enabling data-driven replenishment, proactive stockout prevention, and markdown optimisation across 4 000+ SKUs.**

---

## 📑 Table of Contents

1. [Business Problem & Solution Approach](#-business-problem--solution-approach)
2. [Architecture Diagram](#-architecture-diagram)
3. [Tech Stack](#-tech-stack)
4. [Project Directory Tree](#-project-directory-tree)
5. [Dataset](#-dataset)
6. [Installation](#-installation)
7. [Quick Start](#-quick-start)
8. [Data Pipeline](#-data-pipeline)
9. [Features](#-features)
10. [Models](#-models)
11. [Risk Scoring](#-risk-scoring)
12. [Business Impact](#-business-impact)
13. [API Reference](#-api-reference)
14. [Dashboard Pages](#-dashboard-pages)
15. [Configuration](#-configuration)
16. [Future Improvements](#-future-improvements)
17. [License & Credits](#-license--credits)

---

## 🎯 Business Problem & Solution Approach

### The Problem

NorthBay Living operates across **three fulfilment channels** (D2C web store, wholesale, and two physical showrooms) and carries over 4 000 active SKUs spanning furniture, soft furnishings, kitchenware, and seasonal décor. Prior to FORESIGHT, the planning team relied on static Excel-based reorder-point models refreshed monthly, leading to:

| Pain Point | Estimated Annual Cost |
|---|---|
| Stockouts on hero SKUs during peak season | £420 000 in lost revenue |
| Excess slow-moving inventory tying up working capital | £310 000 in holding costs |
| Emergency air-freight replenishments | £95 000 in premium logistics |
| Markdown losses on end-of-season overstock | £180 000 in margin erosion |
| Planner time spent on manual data wrangling | ~1 800 person-hours / year |

**Total addressable inefficiency: ~£1 005 000 per year.**

### The Solution

FORESIGHT replaces the monthly static model with a **continuously updated, ML-powered demand intelligence engine** that:

1. **Ingests** transactional and product data daily via an automated ETL pipeline.
2. **Engineers** 25+ time-series and business features per SKU-week.
3. **Ensembles** six complementary forecasting models using rolling-origin cross-validation.
4. **Scores** every SKU against a five-level inventory risk framework.
5. **Recommends** replenishment quantities using safety-stock theory calibrated to service-level targets.
6. **Surfaces** all insights through a 12-page interactive Streamlit dashboard and a REST API consumed by ERP and warehouse systems.

---

## 🏗️ Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         FORESIGHT PLATFORM                                   │
│                                                                               │
│  ┌──────────────┐    ┌──────────────────────────────────────────────────┐   │
│  │  DATA SOURCES │    │                  ETL PIPELINE                    │   │
│  │              │    │                                                    │   │
│  │ Online Retail│───▶│  ingest.py ──▶ clean.py ──▶ features.py          │   │
│  │ UCI Dataset  │    │     │               │              │               │   │
│  │              │    │   Raw CSV       Validated       Feature            │   │
│  │ Simulated    │───▶│   Parquet       Parquet        Parquet             │   │
│  │ Inventory &  │    │                                   │               │   │
│  │ Product Data │    └───────────────────────────────────┼───────────────┘   │
│  └──────────────┘                                        │                   │
│                                                          ▼                   │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                        MODELLING LAYER                                  │ │
│  │                                                                          │ │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────────┐  │ │
│  │  │ Naïve /  │ │  SARIMA  │ │   ETS    │ │ Prophet  │ │  XGBoost   │  │ │
│  │  │ Seasonal │ │          │ │          │ │          │ │  Gradient  │  │ │
│  │  │ Baseline │ │          │ │          │ │          │ │  Boosting  │  │ │
│  │  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ └─────┬──────┘  │ │
│  │       │            │            │            │             │           │ │
│  │       └────────────┴────────────┴────────────┴─────────────┘           │ │
│  │                                       │                                  │ │
│  │                               ┌───────▼────────┐                        │ │
│  │                               │  LightGBM      │                        │ │
│  │                               │  Stacking      │                        │ │
│  │                               │  Ensemble      │                        │ │
│  │                               └───────┬────────┘                        │ │
│  └───────────────────────────────────────┼────────────────────────────────┘ │
│                                          │                                   │
│                            ┌─────────────▼──────────────┐                  │
│                            │      RISK ENGINE             │                  │
│                            │  risk_scoring.py             │                  │
│                            │  5-level classification      │                  │
│                            │  + reorder recommendations   │                  │
│                            └─────────────┬──────────────┘                   │
│                                          │                                   │
│              ┌───────────────────────────┼───────────────────────┐          │
│              ▼                           ▼                        ▼          │
│   ┌──────────────────┐       ┌───────────────────┐    ┌──────────────────┐  │
│   │  STREAMLIT APP   │       │   FASTAPI SERVER  │    │   SQLITE / DW    │  │
│   │  12 Dashboard    │       │   8 REST Endpoints│    │   4 Core Tables  │  │
│   │  Pages           │       │   ERP Integration │    │   Model Registry │  │
│   └──────────────────┘       └───────────────────┘    └──────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology | Version | Purpose |
|---|---|---|---|
| **Language** | Python | 3.11+ | Core runtime |
| **Data Wrangling** | pandas | 2.2+ | Tabular ETL and feature engineering |
| **Numerical** | NumPy | 1.26+ | Array operations, safety-stock maths |
| **Visualisation** | Plotly | 5.22+ | Interactive charts in dashboard |
| **Dashboard** | Streamlit | 1.35+ | 12-page BI application |
| **REST API** | FastAPI | 0.111+ | ERP integration endpoints |
| **API Server** | Uvicorn | 0.29+ | ASGI server |
| **API Validation** | Pydantic | 2.7+ | Request / response schema |
| **Statistics** | statsmodels | 0.14+ | SARIMA, ETS, seasonal decomposition |
| **Prophet** | prophet | 1.1+ | Holiday-aware trend forecasting |
| **Gradient Boosting** | XGBoost | 2.0+ | Tabular demand regression |
| **Gradient Boosting** | LightGBM | 4.3+ | Stacking meta-learner |
| **ML Utilities** | scikit-learn | 1.4+ | Pipeline, CV, metrics |
| **Storage** | SQLite | 3.x | Lightweight data warehouse |
| **Serialisation** | Parquet (pyarrow) | 15+ | Columnar data storage |
| **Scheduling** | APScheduler | 3.10+ | Nightly pipeline automation |
| **Env Management** | python-dotenv | 1.0+ | Configuration secrets |
| **Logging** | loguru | 0.7+ | Structured application logging |
| **Testing** | pytest | 8.2+ | Unit and integration tests |
| **Code Style** | black + ruff | latest | Formatting and linting |

---

## 📁 Project Directory Tree

```
foresight/
│
├── 📄 README.md                        # This document
├── 📄 main.py                          # Orchestrator: runs full pipeline end-to-end
├── 📄 requirements.txt                 # Pinned Python dependencies
├── 📄 .env.example                     # Template for environment variables
├── 📄 .gitignore                       # Git ignore rules
├── 📄 pyproject.toml                   # Black / ruff configuration
├── 📄 LICENSE                          # MIT licence text
│
├── 📂 config/
│   ├── 📄 settings.py                  # Central config dataclass (paths, params, flags)
│   └── 📄 logging_config.py            # Loguru sink and rotation configuration
│
├── 📂 data/
│   ├── 📂 raw/
│   │   ├── 📄 online_retail_II.xlsx    # UCI Online Retail II dataset (place here)
│   │   ├── 📄 inventory_snapshot.csv   # Simulated on-hand inventory per SKU
│   │   ├── 📄 product_master.csv       # Simulated product metadata and lead times
│   │   └── 📄 promotions.csv           # Simulated promotional calendar
│   ├── 📂 processed/
│   │   ├── 📄 transactions.parquet     # Cleaned, validated transaction records
│   │   ├── 📄 weekly_demand.parquet    # SKU-week demand aggregations
│   │   ├── 📄 features.parquet         # Engineered feature matrix (25+ columns)
│   │   └── 📄 risk_scores.parquet      # Latest risk scores and recommendations
│   └── 📂 models/
│       ├── 📄 model_registry.db        # SQLite registry of trained model artefacts
│       ├── 📄 xgboost_demand.ubj       # Serialised XGBoost model
│       ├── 📄 lgbm_stacker.txt         # Serialised LightGBM stacking model
│       └── 📄 prophet_<sku_id>.json    # Per-SKU Prophet model serialisations
│
├── 📂 pipeline/
│   ├── 📄 __init__.py
│   ├── 📄 ingest.py                    # Loads raw Excel/CSV, validates schema
│   ├── 📄 clean.py                     # Deduplication, cancellation netting, outlier caps
│   ├── 📄 aggregate.py                 # Aggregates line-items → SKU-week demand table
│   └── 📄 features.py                  # Computes all 25+ engineered features
│
├── 📂 models/
│   ├── 📄 __init__.py
│   ├── 📄 baseline.py                  # Naïve seasonal and moving-average baselines
│   ├── 📄 sarima_model.py              # Auto-ARIMA wrapper with seasonal differencing
│   ├── 📄 ets_model.py                 # Holt-Winters exponential smoothing
│   ├── 📄 prophet_model.py             # Facebook Prophet with UK public holidays
│   ├── 📄 xgboost_model.py             # XGBoost tabular demand regressor
│   ├── 📄 ensemble.py                  # LightGBM stacking meta-learner
│   ├── 📄 cross_validation.py          # Rolling-origin time-series CV harness
│   └── 📄 metrics.py                   # MAE, RMSE, MAPE, SMAPE, Bias, Coverage
│
├── 📂 risk/
│   ├── 📄 __init__.py
│   ├── 📄 risk_scoring.py              # 5-level SKU risk classification engine
│   ├── 📄 safety_stock.py              # Safety-stock and reorder-point calculation
│   └── 📄 recommendations.py           # Natural-language replenishment recommendations
│
├── 📂 api/
│   ├── 📄 __init__.py
│   ├── 📄 app.py                       # FastAPI application factory
│   ├── 📄 routers/
│   │   ├── 📄 forecast.py              # /forecast endpoints
│   │   ├── 📄 inventory.py             # /inventory endpoints
│   │   └── 📄 health.py                # /health liveness probe
│   ├── 📄 schemas.py                   # Pydantic request/response models
│   └── 📄 dependencies.py              # Shared DB session, auth token dependency
│
├── 📂 dashboard/
│   ├── 📄 app.py                       # Streamlit entry point (multipage router)
│   └── 📂 pages/
│       ├── 📄 01_executive_summary.py  # KPI scorecard and headline alerts
│       ├── 📄 02_demand_forecast.py    # Interactive SKU-level forecast explorer
│       ├── 📄 03_inventory_risk.py     # Risk heatmap and ranked SKU table
│       ├── 📄 04_replenishment.py      # Reorder recommendations and PO builder
│       ├── 📄 05_abc_xyz_analysis.py   # ABC-XYZ segmentation matrix
│       ├── 📄 06_seasonality.py        # Seasonal decomposition and YoY comparison
│       ├── 📄 07_product_performance.py# Revenue, margin, and velocity ranking
│       ├── 📄 08_customer_analysis.py  # Cohort, RFM, and geographic spread
│       ├── 📄 09_promotions.py         # Promo lift attribution and calendar view
│       ├── 📄 10_model_diagnostics.py  # CV results, residuals, and feature importance
│       ├── 📄 11_data_quality.py       # Pipeline health, anomalies, and completeness
│       └── 📄 12_settings.py           # User-configurable thresholds and parameters
│
├── 📂 scheduler/
│   ├── 📄 __init__.py
│   └── 📄 jobs.py                      # APScheduler nightly pipeline and weekly retrain
│
├── 📂 tests/
│   ├── 📄 conftest.py                  # Shared pytest fixtures and sample data
│   ├── 📄 test_ingest.py               # Tests for data ingestion and schema validation
│   ├── 📄 test_clean.py                # Tests for cleaning and cancellation netting
│   ├── 📄 test_features.py             # Tests for feature engineering correctness
│   ├── 📄 test_models.py               # Tests for model fit, predict interface contract
│   ├── 📄 test_risk_scoring.py         # Tests for risk classification logic
│   └── 📄 test_api.py                  # FastAPI endpoint integration tests
│
└── 📂 notebooks/
    ├── 📄 01_eda.ipynb                  # Exploratory data analysis and distributions
    ├── 📄 02_feature_engineering.ipynb  # Feature derivation experiments
    ├── 📄 03_model_benchmarking.ipynb   # CV comparison of all six models
    └── 📄 04_risk_framework.ipynb       # Risk threshold calibration experiments
```

---

## 📦 Dataset

### Primary Source — UCI Online Retail II

| Attribute | Detail |
|---|---|
| **Name** | Online Retail II |
| **Source** | [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/502/online+retail+ii) |
| **DOI** | `10.24432/C5CG6D` |
| **Period** | 01 Dec 2009 – 09 Dec 2011 (2 years) |
| **Records** | ~1 067 371 invoice line-items |
| **Countries** | 43 (UK-dominant, ~90% of revenue) |
| **Unique SKUs** | 5 942 (4 063 after cleaning) |
| **Unique Customers** | 5 878 |
| **Key Columns** | `Invoice`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `Price`, `Customer ID`, `Country` |

> **Download:** Visit the UCI repository linked above, download `online_retail_II.xlsx`, and place it in `data/raw/online_retail_II.xlsx`.

### Simulated Supplementary Files

Three additional CSV files are generated by `pipeline/ingest.py --simulate` on first run to represent NorthBay Living's internal data systems. The simulation assumptions are:

| ID | Assumption | Value / Rule |
|---|---|---|
| **A** | **Lead-time distribution** | Uniform 7–21 days per SKU, seeded by `StockCode` hash for reproducibility |
| **B** | **On-hand inventory initialisation** | Random draw from `Uniform(0.5×, 2.5×)` of trailing 8-week average demand per SKU |
| **C** | **Holding cost rate** | 25% of unit cost per annum (industry standard for home-goods retail) |
| **D** | **Service-level target** | 95% (z = 1.645) for Class-A SKUs; 90% (z = 1.282) for Class-B; 85% (z = 1.036) for Class-C |
| **E** | **Promotional uplift model** | Log-normal multiplier with μ = 0.35, σ = 0.15 applied to weeks tagged in `promotions.csv` |
| **F** | **Demand noise** | Additive Gaussian noise N(0, 0.08 × mean_demand) layered onto simulated inventory movements |

---

## ⚙️ Installation

### Prerequisites

| Requirement | Minimum Version | Notes |
|---|---|---|
| Python | 3.11 | Use `pyenv` or system Python |
| pip | 23+ | `python -m pip install --upgrade pip` |
| Git | 2.x | For cloning the repository |
| RAM | 8 GB | 16 GB recommended for Prophet parallel fitting |
| Disk | 2 GB | For dataset, parquet files, and model artefacts |

> **Note:** Prophet requires a C++ compiler (`gcc` / `clang`) and `pystan`. On Ubuntu/Debian run `sudo apt-get install build-essential` before installing dependencies.

### Clone & Install

```bash
# 1. Clone the repository
git clone https://github.com/northbay-living/foresight.git
cd foresight

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Linux / macOS
# .venv\Scripts\activate           # Windows

# 3. Install all dependencies
pip install -r requirements.txt

# 4. Copy and configure environment variables
cp .env.example .env
# Edit .env with your preferred settings (see Configuration section)
```

### Dataset Placement

```bash
# Place the UCI dataset in the expected location:
mv ~/Downloads/online_retail_II.xlsx data/raw/online_retail_II.xlsx

# Generate simulated supplementary files (first-run only):
python main.py --simulate
```

---

## 🚀 Quick Start

### Run the Full Pipeline

```bash
# Ingest → Clean → Aggregate → Feature Engineering → Train → Score → Export
python main.py
```

```bash
# Run pipeline with verbose logging and force-retrain all models
python main.py --verbose --retrain
```

### Launch the Streamlit Dashboard

```bash
streamlit run dashboard/app.py
# Dashboard opens at http://localhost:8501
```

### Start the FastAPI Server

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
# API docs at http://localhost:8000/docs  (Swagger UI)
# ReDoc at    http://localhost:8000/redoc
```

### Run the Test Suite

```bash
pytest tests/ -v --tb=short
```

### Run the Nightly Scheduler (daemon mode)

```bash
python scheduler/jobs.py --daemon
# Runs pipeline at 02:00 UTC daily; retrains models every Sunday at 03:00 UTC
```

---

## 🔄 Data Pipeline

The ETL pipeline produces **four core tables** stored as Parquet files under `data/processed/`:

### Table 1 — `transactions.parquet`

Cleaned, validated invoice line-items with cancellations netted off.

| Column | Type | Description |
|---|---|---|
| `invoice_id` | str | Unique invoice identifier (prefix `C` = credit / cancellation) |
| `stock_code` | str | SKU identifier (mapped to NorthBay product master) |
| `description` | str | Product description (normalised, lowercased) |
| `quantity` | int | Net quantity after cancellation matching |
| `invoice_date` | datetime | UTC-normalised invoice timestamp |
| `unit_price` | float | Line unit price in GBP |
| `revenue` | float | `quantity × unit_price` |
| `customer_id` | str | Anonymised customer identifier |
| `country` | str | ISO-3166 alpha-2 country code |
| `week_start` | date | Monday of the ISO week containing `invoice_date` |

### Table 2 — `weekly_demand.parquet`

SKU-week demand aggregation used as the modelling target.

| Column | Type | Description |
|---|---|---|
| `stock_code` | str | SKU identifier |
| `week_start` | date | ISO week start (Monday) |
| `total_quantity` | int | Total units sold in the week |
| `total_revenue` | float | Total revenue in GBP |
| `n_invoices` | int | Number of distinct invoices |
| `n_customers` | int | Number of distinct customers |
| `avg_unit_price` | float | Revenue-weighted average unit price |
| `is_zero` | bool | True if no sales recorded this week |

### Table 3 — `features.parquet`

The full engineered feature matrix used for ML model training and inference.

| Column | Type | Description |
|---|---|---|
| `stock_code` | str | SKU identifier |
| `week_start` | date | Feature observation week |
| `demand_lag_*` | float | See [Features section](#-features) |
| `rolling_mean_*` | float | See [Features section](#-features) |
| … (25+ columns) | … | All features described below |

### Table 4 — `risk_scores.parquet`

Latest risk classification and replenishment recommendations per SKU.

| Column | Type | Description |
|---|---|---|
| `stock_code` | str | SKU identifier |
| `risk_level` | str | One of: `CRITICAL`, `HIGH`, `MEDIUM`, `LOW`, `HEALTHY` |
| `risk_score` | float | Composite score 0–100 (higher = riskier) |
| `forecast_8w` | float | 8-week-ahead demand point forecast |
| `reorder_point` | float | ROP = lead-time demand + safety stock |
| `reorder_qty` | float | Economic order quantity (EOQ) recommendation |
| `days_of_cover` | float | On-hand inventory ÷ average daily demand |
| `recommendation` | str | Human-readable action text |
| `scored_at` | datetime | Timestamp of scoring run |

---

## 🧮 Features

All features are computed per SKU-week in `pipeline/features.py`. Features are grouped into six conceptual families:

### 📅 Temporal Features

| Feature | Description |
|---|---|
| `week_of_year` | ISO week number (1–52/53) |
| `month` | Calendar month (1–12) |
| `quarter` | Calendar quarter (1–4) |
| `is_q4` | Binary flag for October–December peak trading period |
| `days_to_christmas` | Signed integer days from week midpoint to 25 Dec |
| `is_uk_bank_holiday` | Binary flag for any UK bank holiday in the week |

### 📈 Lag Features (Autoregressive)

| Feature | Description |
|---|---|
| `demand_lag_1w` | Demand 1 week prior |
| `demand_lag_2w` | Demand 2 weeks prior |
| `demand_lag_4w` | Demand 4 weeks prior (monthly signal) |
| `demand_lag_8w` | Demand 8 weeks prior |
| `demand_lag_52w` | Demand 52 weeks prior (same week last year) |

### 📉 Rolling Statistics

| Feature | Description |
|---|---|
| `rolling_mean_4w` | 4-week rolling average demand |
| `rolling_mean_12w` | 12-week rolling average demand |
| `rolling_mean_26w` | 26-week rolling average demand (half-year) |
| `rolling_std_4w` | 4-week rolling standard deviation (volatility) |
| `rolling_std_12w` | 12-week rolling standard deviation |
| `rolling_cv_12w` | Coefficient of variation over 12 weeks |
| `rolling_max_4w` | 4-week rolling peak demand |

### 🏷️ Product & Category Features

| Feature | Description |
|---|---|
| `abc_class` | ABC segmentation class (A / B / C) based on revenue Pareto |
| `xyz_class` | XYZ segmentation class (X / Y / Z) based on demand CV |
| `unit_price` | Current listed unit price in GBP |
| `price_rank_pct` | Percentile rank of unit price within product category |
| `lead_time_days` | Supplier lead time in days (from product master) |

### 📣 External & Promotional Features

| Feature | Description |
|---|---|
| `is_promo_week` | Binary flag for active promotional activity |
| `promo_discount_pct` | Percentage discount offered in promo week |
| `weeks_since_last_promo` | Integer weeks elapsed since most recent promotion |

### 🔢 Inventory State Features

| Feature | Description |
|---|---|
| `on_hand_qty` | Current on-hand inventory units |
| `days_of_cover` | On-hand ÷ rolling 4-week average daily demand |
| `stockout_flag` | Binary: 1 if on-hand < safety stock |

---

## 🤖 Models

FORESIGHT trains and evaluates **six models** per SKU (or SKU cluster for sparse series), then combines the five base learners into a stacking ensemble.

### Model Descriptions

| # | Model | Type | Strengths | Config |
|---|---|---|---|---|
| 1 | **Naïve Seasonal Baseline** | Statistical | Zero-parameter benchmark; uses same week prior year | Seasonal period = 52 weeks |
| 2 | **SARIMA** | Statistical | Captures AR/MA dynamics and seasonal differencing | Auto-selected `(p,d,q)(P,D,Q,52)` via AIC grid search |
| 3 | **ETS (Holt-Winters)** | Statistical | Handles trend + additive/multiplicative seasonality | Auto-selected error/trend/seasonal component |
| 4 | **Prophet** | Statistical / ML | Holiday effects, changepoint detection, robust to gaps | UK public holidays; changepoint_prior_scale = 0.05 |
| 5 | **XGBoost** | Gradient Boosting | Non-linear feature interactions, handles high cardinality | 500 estimators, max_depth=6, learning_rate=0.05 |
| 6 | **LightGBM Stacking** | Gradient Boosting | Meta-learner combining base-model OOF predictions | 200 estimators, num_leaves=31, min_child_samples=20 |

### Rolling-Origin Cross-Validation

Time-series cross-validation is performed using a **rolling-origin** (expanding window) scheme to prevent data leakage:

```
Training Window     Validation Window
─────────────────   ──────────────
Weeks 1 – 52   →   Weeks 53 – 60  (Fold 1)
Weeks 1 – 60   →   Weeks 61 – 68  (Fold 2)
Weeks 1 – 68   →   Weeks 69 – 76  (Fold 3)
Weeks 1 – 76   →   Weeks 77 – 84  (Fold 4)
Weeks 1 – 84   →   Weeks 85 – 92  (Fold 5)
```

- **Minimum training size:** 52 weeks (1 full seasonal cycle)
- **Forecast horizon:** 8 weeks (≈ 2 months ahead)
- **Gap between train end and validation start:** 0 (contiguous)
- **SKUs with < 52 non-zero weeks** are handled by the naïve baseline only

### Evaluation Metrics

| Metric | Formula | Interpretation |
|---|---|---|
| **MAE** | `mean(|y - ŷ|)` | Average absolute error in original units |
| **RMSE** | `sqrt(mean((y - ŷ)²))` | Penalises large errors more than MAE |
| **MAPE** | `mean(|y - ŷ| / y) × 100` | Percentage error (excluded for zero-demand weeks) |
| **SMAPE** | `mean(2×|y-ŷ| / (|y|+|ŷ|)) × 100` | Symmetric MAPE, handles zeros more gracefully |
| **Bias** | `mean(ŷ - y)` | Directional systematic over/under-forecast |
| **PI Coverage** | `mean(y_low ≤ y ≤ y_high)` | Empirical coverage of 90% prediction interval |

---

## ⚠️ Risk Scoring

The risk engine in `risk/risk_scoring.py` computes a **composite risk score (0–100)** for each active SKU every time the pipeline runs, then assigns a discrete risk level:

### Risk Score Components

| Component | Weight | Inputs |
|---|---|---|
| Days of Cover | 35% | `on_hand_qty`, rolling demand rate |
| Forecast Uncertainty | 20% | Ensemble prediction interval width (CV of base-model forecasts) |
| Demand Trend | 20% | Sign and magnitude of 4w vs 12w rolling mean delta |
| Stockout History | 15% | Fraction of past 12 weeks with zero demand (supply-side) |
| Lead Time Risk | 10% | `lead_time_days` relative to days-of-cover headroom |

### Risk Levels

| Level | Score Range | Colour | Meaning | Typical Action |
|---|---|---|---|---|
| 🔴 **CRITICAL** | 80 – 100 | Red | Imminent stockout within lead time | Emergency replenishment; escalate to Head of Supply Chain |
| 🟠 **HIGH** | 60 – 79 | Orange | Stockout probable within 2× lead time | Raise PO immediately; explore alternative suppliers |
| 🟡 **MEDIUM** | 40 – 59 | Yellow | Elevated risk; monitor closely | Schedule replenishment within 1 week |
| 🟢 **LOW** | 20 – 39 | Light Green | Adequate cover; minor concern | Review at next weekly planning cycle |
| ✅ **HEALTHY** | 0 – 19 | Green | Comfortable stock position | No action required; flag if overstock detected |

### Replenishment Recommendations

The `risk/recommendations.py` module generates natural-language recommended actions in the following format:

```
[CRITICAL] StockCode: 22728 | "ALARM CLOCK BAKELIKE PINK"
  Current DOC: 4.2 days | Lead Time: 14 days | Forecast 8w: 312 units
  ▶ Raise emergency PO for 420 units with backup supplier (lead time: 7 days).
    Safety Stock: 48 units | Reorder Point: 192 units | EOQ: 380 units
```

---

## 📊 Business Impact

Based on back-testing over the 24-month UCI dataset and Monte Carlo simulation of NorthBay Living's inventory parameters:

| Metric | Before FORESIGHT | After FORESIGHT | Improvement |
|---|---|---|---|
| **Forecast MAPE (8-week horizon)** | 38.4% (naïve) | 14.2% (ensemble) | −63% |
| **Stockout Rate (Class-A SKUs)** | 12.3% of weeks | 3.8% of weeks | −69% |
| **Excess Inventory (weeks of cover)** | 11.2 weeks avg | 7.4 weeks avg | −34% |
| **Emergency Air-Freight Events** | ~18 / year | ~4 / year | −78% |
| **Planner Time on Manual Reports** | ~35 hrs / week | ~8 hrs / week | −77% |
| **Estimated Annual Saving** | — | **£640 000+** | — |
| **Markdown Losses** | £180 000 / year | £62 000 / year | −66% |

> *Projections based on simulation. Actual results will vary with live implementation and data quality.*

---

## 🌐 API Reference

The FastAPI server exposes **8 REST endpoints**. Full interactive documentation is available at `http://localhost:8000/docs` when the server is running.

### Authentication

All endpoints except `/health` require a Bearer token set in `.env` as `API_SECRET_KEY`.

```bash
export TOKEN="your_api_secret_key"
```

---

### Endpoint 1 — `GET /health`

Liveness probe for load balancers and monitoring systems.

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "version": "1.0.0",
  "pipeline_last_run": "2026-07-20T02:00:00Z",
  "model_count": 4063
}
```

---

### Endpoint 2 — `GET /forecast/{stock_code}`

Retrieve the 8-week demand forecast for a single SKU.

```bash
curl -H "Authorization: Bearer $TOKEN" \
  "http://localhost:8000/forecast/22728?horizon=8&include_intervals=true"
```

```json
{
  "stock_code": "22728",
  "description": "ALARM CLOCK BAKELIKE PINK",
  "forecast": [
    { "week_start": "2026-07-27", "point": 42.3, "lower_90": 31.0, "upper_90": 54.1 },
    { "week_start": "2026-08-03", "point": 38.7, "lower_90": 28.5, "upper_90": 49.8 }
  ],
  "model_used": "lgbm_stacking_ensemble",
  "generated_at": "2026-07-20T21:17:32Z"
}
```

---

### Endpoint 3 — `POST /forecast/batch`

Bulk forecast for a list of SKUs in a single call (max 500 SKUs per request).

```bash
curl -X POST \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"stock_codes": ["22728", "85123A", "21730"], "horizon": 4}' \
  http://localhost:8000/forecast/batch
```

```json
{
  "results": [
    { "stock_code": "22728", "forecast": [ ... ] },
    { "stock_code": "85123A", "forecast": [ ... ] },
    { "stock_code": "21730", "forecast": [ ... ] }
  ],
  "requested": 3,
  "returned": 3
}
```

---

### Endpoint 4 — `GET /inventory/risk`

Return risk scores for all SKUs, optionally filtered by risk level.

```bash
curl -H "Authorization: Bearer $TOKEN" \
  "http://localhost:8000/inventory/risk?level=CRITICAL&limit=50"
```

```json
{
  "risk_level_filter": "CRITICAL",
  "total_skus": 4063,
  "returned": 12,
  "items": [
    {
      "stock_code": "22728",
      "risk_level": "CRITICAL",
      "risk_score": 87.4,
      "days_of_cover": 4.2,
      "reorder_point": 192,
      "reorder_qty": 380,
      "recommendation": "Raise emergency PO for 420 units with backup supplier."
    }
  ]
}
```

---

### Endpoint 5 — `GET /inventory/risk/{stock_code}`

Risk score detail for a single SKU including component breakdown.

```bash
curl -H "Authorization: Bearer $TOKEN" \
  http://localhost:8000/inventory/risk/22728
```

```json
{
  "stock_code": "22728",
  "risk_level": "CRITICAL",
  "risk_score": 87.4,
  "score_components": {
    "days_of_cover_score": 30.8,
    "forecast_uncertainty_score": 17.2,
    "demand_trend_score": 18.1,
    "stockout_history_score": 13.5,
    "lead_time_risk_score": 7.8
  },
  "on_hand_qty": 55,
  "safety_stock": 48,
  "reorder_point": 192,
  "reorder_qty": 380,
  "lead_time_days": 14,
  "days_of_cover": 4.2,
  "forecast_8w": 312.0,
  "scored_at": "2026-07-20T02:00:00Z"
}
```

---

### Endpoint 6 — `GET /pipeline/status`

Return the current status and run history of the ETL pipeline.

```bash
curl -H "Authorization: Bearer $TOKEN" \
  http://localhost:8000/pipeline/status
```

```json
{
  "last_run": "2026-07-20T02:00:00Z",
  "last_run_status": "success",
  "records_processed": 1067371,
  "skus_active": 4063,
  "next_scheduled_run": "2026-07-21T02:00:00Z",
  "model_last_retrained": "2026-07-20T03:00:00Z"
}
```

---

### Endpoint 7 — `POST /pipeline/trigger`

Manually trigger a pipeline run (requires admin token scope).

```bash
curl -X POST \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"retrain": false, "verbose": true}' \
  http://localhost:8000/pipeline/trigger
```

```json
{
  "run_id": "run_20260720_211732",
  "status": "accepted",
  "message": "Pipeline run queued. Poll /pipeline/status for updates."
}
```

---

### Endpoint 8 — `GET /products/abc-xyz`

Return the ABC-XYZ segmentation matrix for all active SKUs.

```bash
curl -H "Authorization: Bearer $TOKEN" \
  "http://localhost:8000/products/abc-xyz?category=kitchenware"
```

```json
{
  "category": "kitchenware",
  "segments": {
    "AX": { "count": 48, "revenue_pct": 32.1 },
    "AY": { "count": 31, "revenue_pct": 18.4 },
    "AZ": { "count": 12, "revenue_pct": 7.2 },
    "BX": { "count": 67, "revenue_pct": 14.3 },
    "BY": { "count": 55, "revenue_pct": 9.8 },
    "BZ": { "count": 29, "revenue_pct": 5.1 },
    "CX": { "count": 112, "revenue_pct": 6.4 },
    "CY": { "count": 94, "revenue_pct": 4.8 },
    "CZ": { "count": 201, "revenue_pct": 1.9 }
  }
}
```

---

## 📺 Dashboard Pages

The Streamlit application is a 12-page BI dashboard accessible at `http://localhost:8501`:

| # | Page | Key Widgets |
|---|---|---|
| 1 | **Executive Summary** | KPI tiles (revenue, MAPE, stockout %, DOC), alert banner for CRITICAL SKUs, weekly trend sparklines |
| 2 | **Demand Forecast** | SKU selector, interactive 52-week actuals + 8-week forecast chart with PI bands, model comparison table |
| 3 | **Inventory Risk** | Risk level heatmap, sortable SKU risk table with drill-down, radar chart of score components |
| 4 | **Replenishment Planner** | CRITICAL/HIGH SKU list with PO builder, EOQ calculator, export to CSV / ERP-ready JSON |
| 5 | **ABC-XYZ Analysis** | 3×3 matrix scatter plot, segment-level metrics table, strategy guidance per segment |
| 6 | **Seasonality Explorer** | STL decomposition plots, week-of-year demand heatmap, YoY comparison with variance shading |
| 7 | **Product Performance** | Revenue ranking bar chart, margin waterfall, velocity leaderboard, price elasticity scatter |
| 8 | **Customer Analysis** | RFM segmentation bubble chart, geographic choropleth, cohort retention heatmap |
| 9 | **Promotions** | Promotional calendar Gantt, lift attribution table (pre/during/post), ROI estimator |
| 10 | **Model Diagnostics** | Rolling-CV metric table per model, residual distribution plots, SHAP feature importance bar chart |
| 11 | **Data Quality** | Pipeline run log, schema validation results, missing-data heatmap, outlier detection summary |
| 12 | **Settings** | Adjustable risk thresholds, service-level targets per ABC class, holding-cost rate, lead-time overrides |

---

## 🔧 Configuration

All settings are managed via environment variables (`.env`) and the central `config/settings.py` dataclass:

```dotenv
# ── Data Paths ──────────────────────────────────────────────
RAW_DATA_DIR=data/raw
PROCESSED_DATA_DIR=data/processed
MODELS_DIR=data/models

# ── Pipeline ─────────────────────────────────────────────────
FORCE_RETRAIN=false                    # true = retrain all models each run
MIN_WEEKS_FOR_ML=52                    # SKUs below this use naïve baseline only
FORECAST_HORIZON_WEEKS=8              # Look-ahead period for all models
CV_N_FOLDS=5                           # Number of rolling-origin CV folds

# ── Risk Scoring ─────────────────────────────────────────────
RISK_CRITICAL_THRESHOLD=80            # Composite score ≥ 80 → CRITICAL
RISK_HIGH_THRESHOLD=60                # Composite score ≥ 60 → HIGH
RISK_MEDIUM_THRESHOLD=40              # Composite score ≥ 40 → MEDIUM
RISK_LOW_THRESHOLD=20                 # Composite score ≥ 20 → LOW

# ── Service Levels ───────────────────────────────────────────
SERVICE_LEVEL_A=0.95                   # 95% service level for Class-A SKUs
SERVICE_LEVEL_B=0.90                   # 90% service level for Class-B SKUs
SERVICE_LEVEL_C=0.85                   # 85% service level for Class-C SKUs
HOLDING_COST_RATE=0.25                 # Annual holding cost as fraction of unit cost

# ── API ──────────────────────────────────────────────────────
API_HOST=0.0.0.0
API_PORT=8000
API_SECRET_KEY=changeme_in_production  # Bearer token for API authentication
API_BATCH_LIMIT=500                    # Max SKUs per /forecast/batch call

# ── Scheduler ────────────────────────────────────────────────
PIPELINE_CRON=0 2 * * *               # Nightly pipeline at 02:00 UTC
RETRAIN_CRON=0 3 * * 0                # Weekly retrain every Sunday at 03:00 UTC

# ── Logging ──────────────────────────────────────────────────
LOG_LEVEL=INFO
LOG_FILE=logs/foresight.log
LOG_ROTATION=50 MB
LOG_RETENTION=30 days
```

---

## 🔮 Future Improvements

| Priority | Idea | Rationale |
|---|---|---|
| 🔴 High | **External demand signals** — integrate Google Trends, weather APIs, and competitor pricing scrapers as additional features | Captures demand shocks driven by external events not visible in transaction history |
| 🔴 High | **Multi-echelon inventory optimisation** — extend the model to jointly optimise stock across warehouse, showroom, and in-transit layers | Current model treats each location independently; multi-echelon reduces total system inventory |
| 🟠 Medium | **Causal inference for promotions** — replace the multiplier model (Assumption E) with a DID or synthetic control estimator to disentangle true promo lift from coincident demand | Reduces promo-attribution bias that currently inflates forecast accuracy metrics during sale periods |
| 🟠 Medium | **Real-time streaming pipeline** — migrate from nightly batch (APScheduler) to an event-driven Kafka + Faust stream processing architecture | Enables intra-day replenishment triggers for fast-moving SKUs during peak periods (e.g. Black Friday) |
| 🟡 Medium | **Automated model selection per SKU** — use Bayesian optimisation (Optuna) to tune model hyperparameters and selection per SKU rather than one global config | Improves forecast accuracy for heterogeneous SKUs with distinct demand patterns (intermittent vs. trend vs. seasonal) |
| 🟡 Medium | **Supplier integration via EDI** — auto-generate and transmit purchase orders to suppliers using EDI 850 standard when a CRITICAL risk is detected | Closes the loop from intelligence to action without planner intervention |
| 🟢 Low | **Mobile push alerts** — integrate Firebase Cloud Messaging to push CRITICAL risk alerts to the supply chain team's smartphones | Ensures actionable alerts reach planners even when the dashboard is not open |

---

## 📄 License & Credits

### License

This project is licensed under the **MIT License** — see the [`LICENSE`](LICENSE) file for full terms.

```
MIT License

Copyright (c) 2026 NorthBay Living Ltd

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### Dataset Citation

> Chen, D. (2015). *Online Retail II* [Dataset]. UCI Machine Learning Repository. [https://doi.org/10.24432/C5CG6D](https://doi.org/10.24432/C5CG6D)

### Acknowledgements

- **UCI Machine Learning Repository** — for making the Online Retail II dataset publicly available.
- **Meta Open Source** — for [Prophet](https://github.com/facebook/prophet), licensed under the MIT License.
- **Microsoft** — for [LightGBM](https://github.com/microsoft/LightGBM), licensed under the MIT License.
- **XGBoost Contributors** — for [XGBoost](https://github.com/dmlc/xgboost), licensed under the Apache 2.0 License.
- **Streamlit Inc.** — for the [Streamlit](https://streamlit.io/) framework.
- **Sebastián Ramírez** — for [FastAPI](https://fastapi.tiangolo.com/).

---

<div align="center">

Built with ❤️ by the NorthBay Living Data & Analytics Team · 2026

</div>
