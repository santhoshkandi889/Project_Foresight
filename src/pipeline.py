"""
Project FORESIGHT – Pipeline Orchestrator
==========================================
Single-command execution of the full ML pipeline:

  python main.py

Steps:
  1. Data Preprocessing    → 4 FORESIGHT tables
  2. Feature Engineering   → features_engineered.csv
  3. Forecasting           → model training + forecast_output.csv
  4. Risk Scoring          → risk_scores.csv
  5. Business Impact       → impact summary
  6. Report Generation     → reports/
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime

from src.utils import logger, timer, safe_run, DATA_RAW, DATA_PROCESSED, MODELS_DIR, REPORTS_DIR


@timer
@safe_run
def run_pipeline(skip_training: bool = False) -> dict:
    """Execute the complete FORESIGHT pipeline."""
    start_time = time.time()
    results = {}

    logger.info("""
╔══════════════════════════════════════════════════════════╗
║       PROJECT FORESIGHT – NorthBay Living                ║
║       Demand & Inventory Intelligence Pipeline           ║
╚══════════════════════════════════════════════════════════╝
""")

    # ─── Step 1: Preprocessing ───────────────────────────────────
    logger.info("┌─ STEP 1: Data Preprocessing ─────────────────────────────")
    from src.preprocessing import run_preprocessing
    sales_daily, sku_master, calendar, inventory = run_preprocessing()
    results["preprocessing"] = {
        "sales_daily_rows": len(sales_daily),
        "sku_count": sku_master["stock_code"].nunique(),
        "date_range": f"{sales_daily['date'].min()} → {sales_daily['date'].max()}",
        "calendar_rows": len(calendar),
    }
    logger.info("└─ Step 1 complete ✓")

    # ─── Step 2: Feature Engineering ─────────────────────────────
    logger.info("┌─ STEP 2: Feature Engineering ────────────────────────────")
    from src.feature_engineering import run_feature_engineering
    features_df = run_feature_engineering(sales_daily, sku_master, calendar, inventory)
    results["feature_engineering"] = {
        "shape": list(features_df.shape),
        "feature_count": features_df.shape[1],
    }
    logger.info("└─ Step 2 complete ✓")

    # ─── Step 3: Forecasting ──────────────────────────────────────
    if not skip_training:
        logger.info("┌─ STEP 3: Model Training & Forecasting ───────────────────")
        from src.forecasting import run_forecasting
        forecast_results = run_forecasting(features_df)
        results["forecasting"] = {
            "best_model": forecast_results["best_model_name"],
            "cv_metrics": {
                m: forecast_results["cv_results"][m]
                for m in forecast_results["cv_results"]
            },
            "forecast_rows": len(forecast_results["forecast_df"]),
        }
        logger.info("└─ Step 3 complete ✓")
        best_model = forecast_results["best_model"]
        best_model_name = forecast_results["best_model_name"]
        forecast_df = forecast_results["forecast_df"]
    else:
        logger.info("⚡ Skipping model training (skip_training=True)")
        import pandas as pd
        forecast_file = DATA_PROCESSED / "forecast_output.csv"
        forecast_df = pd.read_csv(forecast_file, parse_dates=["date"]) if forecast_file.exists() else None
        best_model_name = "LightGBM"

    # ─── Step 4: Risk Scoring ──────────────────────────────────────
    logger.info("┌─ STEP 4: Risk Scoring ───────────────────────────────────")
    from src.risk_scoring import run_risk_scoring, calculate_business_impact
    risk_df = run_risk_scoring(
        features_df=features_df,
        inventory_df=inventory,
        sku_master=sku_master,
        forecast_df=forecast_df,
    )
    results["risk_scoring"] = {
        "total_skus": len(risk_df),
        "critical_stockout": int((risk_df["risk_level"] == "Critical Stockout Risk").sum()),
        "overstock": int(risk_df["risk_level"].str.contains("Overstock").sum()),
        "healthy": int((risk_df["risk_level"] == "Healthy").sum()),
    }
    logger.info("└─ Step 4 complete ✓")

    # ─── Step 5: Business Impact ───────────────────────────────────
    logger.info("┌─ STEP 5: Business Impact Calculation ────────────────────")
    impact = calculate_business_impact(risk_df)
    results["business_impact"] = impact

    # Save impact as JSON
    impact_path = DATA_PROCESSED / "business_impact.json"
    impact_path.write_text(json.dumps(impact, indent=2))
    logger.info("└─ Step 5 complete ✓")

    # ─── Step 6: Pipeline Summary ──────────────────────────────────
    elapsed = time.time() - start_time
    results["elapsed_seconds"] = round(elapsed, 2)
    results["completed_at"] = datetime.now().isoformat()

    summary_path = DATA_PROCESSED / "pipeline_summary.json"
    summary_path.write_text(json.dumps(results, indent=2, default=str))

    logger.info(f"""
╔══════════════════════════════════════════════════════════╗
║  ✅ PIPELINE COMPLETE in {elapsed:.1f}s                        ║
╠══════════════════════════════════════════════════════════╣
║  SKUs processed   : {results['preprocessing']['sku_count']:>8,}                          ║
║  Best model       : {results.get('forecasting', {}).get('best_model', 'N/A'):<20}           ║
║  Revenue at risk  : £{impact.get('revenue_at_risk', 0):>12,.2f}                 ║
║  Capital locked   : £{impact.get('capital_locked', 0):>12,.2f}                 ║
╚══════════════════════════════════════════════════════════╝

Next steps:
  📊 Dashboard  : streamlit run app/streamlit_app.py
  🌐 API Server : uvicorn service.api:app --reload --port 8000
""")

    return results


if __name__ == "__main__":
    # Parse CLI args
    skip_train = "--skip-training" in sys.argv
    results = run_pipeline(skip_training=skip_train)
