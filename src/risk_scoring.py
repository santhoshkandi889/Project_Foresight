"""
Project FORESIGHT – Risk Scoring Module
=========================================
Transparent business-rule risk engine that classifies every SKU into:

  • Critical Stockout Risk
  • High Stockout Risk
  • Overstock Risk
  • Volatile Demand
  • Healthy Inventory

Outputs per SKU:
  - risk_score (0–100)
  - risk_level
  - revenue_at_risk
  - capital_locked
  - inventory_coverage
  - priority_score
  - recommendation
  - business_impact
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
from pathlib import Path

from src.utils import logger, timer, safe_run, CONFIG, DATA_PROCESSED, MODELS_DIR, load_csv, save_csv


# ─────────────────────────────────────────────
# Risk Scoring Rules
# ─────────────────────────────────────────────
STOCKOUT_CRITICAL_DAYS  = 7
STOCKOUT_HIGH_DAYS      = 14
OVERSTOCK_DAYS          = 90
VOLATILE_CV_THRESHOLD   = 1.5    # coefficient of variation
HOLDING_COST_RATE       = 0.20   # 20% annual
STOCKOUT_PENALTY_MULT   = 2.0    # brand impact multiplier


def _risk_score_stockout(coverage_days: float, lead_time: float, demand_vol: float) -> float:
    """
    Stockout risk score [0–100].
    Higher = more risk.
    """
    # Base score: inverse coverage relative to lead time
    if coverage_days <= 0:
        base = 100.0
    elif coverage_days < lead_time:
        base = 90 - (coverage_days / lead_time) * 40
    elif coverage_days < STOCKOUT_CRITICAL_DAYS:
        base = 70 - (coverage_days / STOCKOUT_CRITICAL_DAYS) * 30
    elif coverage_days < STOCKOUT_HIGH_DAYS:
        base = 40 - ((coverage_days - STOCKOUT_CRITICAL_DAYS) / (STOCKOUT_HIGH_DAYS - STOCKOUT_CRITICAL_DAYS)) * 20
    elif coverage_days < 30:
        base = 20 - (coverage_days / 30) * 15
    else:
        base = 5.0

    # Volatility adjustment: volatile demand increases risk
    vol_adj = min(demand_vol * 10, 20)
    return min(max(base + vol_adj, 0), 100)


def _risk_score_overstock(coverage_days: float, demand_vol: float) -> float:
    """
    Overstock risk score [0–100].
    Higher = more capital locked.
    """
    if coverage_days > 365:
        base = 90.0
    elif coverage_days > OVERSTOCK_DAYS:
        base = 60 + ((coverage_days - OVERSTOCK_DAYS) / (365 - OVERSTOCK_DAYS)) * 30
    elif coverage_days > 60:
        base = 30 + ((coverage_days - 60) / (OVERSTOCK_DAYS - 60)) * 30
    else:
        base = max(0, (coverage_days / 60) * 15)
    return min(max(base, 0), 100)


def _assign_risk_level(stockout_score: float, overstock_score: float, demand_vol: float) -> str:
    """Assign human-readable risk level."""
    if stockout_score >= 75:
        return "Critical Stockout Risk"
    elif stockout_score >= 50:
        return "High Stockout Risk"
    elif stockout_score >= 30:
        return "Moderate Stockout Risk"
    elif overstock_score >= 70:
        return "Overstock Risk"
    elif overstock_score >= 40:
        return "Mild Overstock Risk"
    elif demand_vol > VOLATILE_CV_THRESHOLD:
        return "Volatile Demand"
    else:
        return "Healthy"


def _assign_recommendation(
    risk_level: str,
    coverage_days: float,
    demand_mom: float,
    abc_class: str,
) -> str:
    """Business recommendation based on risk level and demand trends."""
    if risk_level == "Critical Stockout Risk":
        return "🚨 Reorder Immediately"
    elif risk_level == "High Stockout Risk":
        return "📦 Increase Purchase"
    elif risk_level == "Moderate Stockout Risk":
        if demand_mom > 1.2:
            return "📦 Increase Purchase"
        return "📊 Monitor Closely"
    elif risk_level == "Overstock Risk":
        if abc_class == "C":
            return "🏷 Clearance Sale"
        elif coverage_days > 180:
            return "🔗 Bundle Offer"
        else:
            return "📉 Markdown Promotion"
    elif risk_level == "Mild Overstock Risk":
        return "📉 Reduce Purchase"
    elif risk_level == "Volatile Demand":
        return "📊 Monitor Closely"
    else:
        return "✅ Healthy – No Action"


def _calculate_revenue_at_risk(
    avg_daily_demand: float,
    unit_price: float,
    coverage_days: float,
    risk_level: str,
) -> float:
    """Estimated revenue at risk due to stockout."""
    if "Stockout" not in risk_level:
        return 0.0
    days_at_risk = max(0, STOCKOUT_HIGH_DAYS - coverage_days)
    return days_at_risk * avg_daily_demand * unit_price * STOCKOUT_PENALTY_MULT


def _calculate_capital_locked(
    closing_inventory: float,
    unit_price: float,
    coverage_days: float,
    risk_level: str,
) -> float:
    """Capital locked in excess inventory."""
    if "Overstock" not in risk_level:
        return 0.0
    excess_days = max(0, coverage_days - 60)
    excess_units = (excess_days / max(coverage_days, 1)) * closing_inventory
    return excess_units * unit_price


# ─────────────────────────────────────────────
# Main Risk Scoring
# ─────────────────────────────────────────────
@timer
@safe_run
def run_risk_scoring(
    features_df: Optional[pd.DataFrame] = None,
    inventory_df: Optional[pd.DataFrame] = None,
    sku_master: Optional[pd.DataFrame] = None,
    forecast_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Compute risk scores for every SKU.
    Uses the most recent inventory snapshot + demand statistics.
    """
    if features_df is None:
        features_df = load_csv(DATA_PROCESSED / "features_engineered.csv", parse_dates=["date"])
    if inventory_df is None:
        inventory_df = load_csv(DATA_PROCESSED / "inventory_snapshots.csv", parse_dates=["date"])
    if sku_master is None:
        sku_master = load_csv(DATA_PROCESSED / "sku_master.csv")

    features_df["date"] = pd.to_datetime(features_df["date"])
    inventory_df["date"] = pd.to_datetime(inventory_df["date"])

    # Latest snapshot per SKU
    latest_inv = inventory_df.sort_values("date").groupby("stock_code").last().reset_index()

    # Demand statistics from features
    demand_stats = (
        features_df.sort_values("date")
        .groupby("stock_code")
        .agg(
            avg_daily_demand=("quantity", "mean"),
            std_daily_demand=("quantity", "std"),
            demand_momentum=("demand_momentum", "last") if "demand_momentum" in features_df.columns else ("quantity", "last"),
            total_revenue=("revenue", "sum") if "revenue" in features_df.columns else ("quantity", "sum"),
            avg_unit_price=("unit_price", "mean") if "unit_price" in features_df.columns else ("avg_unit_price_x", "mean"),
        )
        .reset_index()
    )

    # Drop overlapping columns to prevent _x, _y suffixes
    overlap = [c for c in demand_stats.columns if c in latest_inv.columns and c != "stock_code"]
    if overlap:
        latest_inv = latest_inv.drop(columns=overlap)

    # Merge
    risk_df = latest_inv.merge(demand_stats, on="stock_code", how="left")
    risk_df = risk_df.merge(
        sku_master[["stock_code", "category", "abc_class", "lead_time_days", "description"]],
        on="stock_code", how="left"
    )

    # Fill price from sku_master if missing
    if "avg_unit_price" not in risk_df.columns or risk_df["avg_unit_price"].isna().all():
        price_map = sku_master.set_index("stock_code")["avg_unit_price"].to_dict()
        risk_df["avg_unit_price"] = risk_df["stock_code"].map(price_map)

    risk_df["avg_unit_price"]     = risk_df["avg_unit_price"].fillna(1.0)
    risk_df["avg_daily_demand"]   = risk_df["avg_daily_demand"].fillna(0)
    risk_df["std_daily_demand"]   = risk_df["std_daily_demand"].fillna(0)
    risk_df["coverage_days"]      = risk_df["coverage_days"].fillna(0)
    risk_df["closing_inventory"]  = risk_df["closing_inventory"].fillna(0)
    risk_df["lead_time_days"]     = risk_df["lead_time_days"].fillna(7)
    risk_df["demand_momentum"]    = risk_df.get("demand_momentum", pd.Series(1.0, index=risk_df.index)).fillna(1.0)
    risk_df["abc_class"]          = risk_df["abc_class"].fillna("C")

    # Demand volatility (CV)
    risk_df["demand_cv"] = (
        risk_df["std_daily_demand"] / risk_df["avg_daily_demand"].clip(lower=0.001)
    ).fillna(0)

    # Scores
    risk_df["stockout_score"] = risk_df.apply(
        lambda r: _risk_score_stockout(r["coverage_days"], r["lead_time_days"], r["demand_cv"]),
        axis=1,
    )
    risk_df["overstock_score"] = risk_df.apply(
        lambda r: _risk_score_overstock(r["coverage_days"], r["demand_cv"]),
        axis=1,
    )

    # Composite risk score (weighted: stockout 60%, overstock 40%)
    risk_df["risk_score"] = (
        0.60 * risk_df["stockout_score"] + 0.40 * risk_df["overstock_score"]
    ).round(2)

    # Risk level
    risk_df["risk_level"] = risk_df.apply(
        lambda r: _assign_risk_level(r["stockout_score"], r["overstock_score"], r["demand_cv"]),
        axis=1,
    )

    # Recommendation
    risk_df["recommendation"] = risk_df.apply(
        lambda r: _assign_recommendation(
            r["risk_level"], r["coverage_days"],
            r["demand_momentum"], r["abc_class"]
        ),
        axis=1,
    )

    # Revenue at risk
    risk_df["revenue_at_risk"] = risk_df.apply(
        lambda r: _calculate_revenue_at_risk(
            r["avg_daily_demand"], r["avg_unit_price"],
            r["coverage_days"], r["risk_level"]
        ),
        axis=1,
    ).round(2)

    # Capital locked
    risk_df["capital_locked"] = risk_df.apply(
        lambda r: _calculate_capital_locked(
            r["closing_inventory"], r["avg_unit_price"],
            r["coverage_days"], r["risk_level"]
        ),
        axis=1,
    ).round(2)

    # Priority score = risk_score × abc_weight
    abc_weight = {"A": 3.0, "B": 2.0, "C": 1.0}
    risk_df["priority_score"] = (
        risk_df["risk_score"] * risk_df["abc_class"].map(abc_weight).fillna(1.0)
    ).round(2)

    # Forecast revenue (28-day)
    if forecast_df is not None:
        forecast_df["date"] = pd.to_datetime(forecast_df["date"])
        fcast_rev = forecast_df.groupby("stock_code").agg(
            forecast_qty=("forecast_quantity", "sum")
        ).reset_index()
        fcast_rev = fcast_rev.merge(
            risk_df[["stock_code", "avg_unit_price"]], on="stock_code", how="left"
        )
        fcast_rev["forecast_revenue"] = (fcast_rev["forecast_qty"] * fcast_rev["avg_unit_price"]).round(2)
        risk_df = risk_df.merge(
            fcast_rev[["stock_code", "forecast_qty", "forecast_revenue"]],
            on="stock_code", how="left"
        )
    else:
        risk_df["forecast_qty"] = risk_df["avg_daily_demand"] * CONFIG["FORECAST_HORIZON"]
        risk_df["forecast_revenue"] = (risk_df["forecast_qty"] * risk_df["avg_unit_price"]).round(2)

    # Sort by priority
    risk_df = risk_df.sort_values("priority_score", ascending=False).reset_index(drop=True)

    # Save
    out_cols = [
        "stock_code", "description", "category", "abc_class",
        "closing_inventory", "coverage_days", "reorder_point", "eoq",
        "avg_daily_demand", "avg_unit_price", "lead_time_days",
        "demand_cv", "demand_momentum",
        "stockout_score", "overstock_score", "risk_score", "risk_level",
        "revenue_at_risk", "capital_locked", "priority_score",
        "recommendation", "forecast_qty", "forecast_revenue",
        "inventory_value",
    ]
    out_cols = [c for c in out_cols if c in risk_df.columns]
    save_csv(risk_df[out_cols], DATA_PROCESSED / "risk_scores.csv")

    # Summary
    logger.info(f"✅ Risk scoring complete: {len(risk_df):,} SKUs")
    for level in risk_df["risk_level"].value_counts().index:
        count = (risk_df["risk_level"] == level).sum()
        logger.info(f"   {level}: {count}")

    return risk_df[out_cols]


# ─────────────────────────────────────────────
# Business Impact Calculator
# ─────────────────────────────────────────────
@timer
def calculate_business_impact(risk_df: pd.DataFrame) -> Dict:
    """Aggregate business impact metrics from risk scores."""
    total_inv_value    = risk_df["inventory_value"].sum()
    total_rev_at_risk  = risk_df["revenue_at_risk"].sum()
    total_cap_locked   = risk_df["capital_locked"].sum()
    forecast_revenue   = risk_df["forecast_revenue"].sum()

    # Stockout cost = revenue at risk × penalty multiplier
    stockout_cost = total_rev_at_risk * 1.5

    # Overstock cost = holding cost on locked capital
    overstock_cost = total_cap_locked * HOLDING_COST_RATE / 12  # monthly

    # Working capital saved if overstock eliminated
    working_capital_saved = total_cap_locked * 0.70  # assume 70% reducible

    # Potential sales increase from eliminating stockouts
    potential_sales_increase = total_rev_at_risk * 0.85  # 85% recovery rate

    # Expected profit improvement
    profit_improvement = (potential_sales_increase - stockout_cost) + working_capital_saved

    impact = {
        "revenue_at_risk":         round(total_rev_at_risk, 2),
        "forecast_revenue_28d":    round(forecast_revenue, 2),
        "inventory_value":         round(total_inv_value, 2),
        "capital_locked":          round(total_cap_locked, 2),
        "working_capital_saved":   round(working_capital_saved, 2),
        "potential_sales_increase":round(potential_sales_increase, 2),
        "overstock_cost_monthly":  round(overstock_cost, 2),
        "stockout_cost":           round(stockout_cost, 2),
        "expected_profit_improvement": round(profit_improvement, 2),
        # SKU counts
        "critical_stockout_count": int((risk_df["risk_level"] == "Critical Stockout Risk").sum()),
        "high_stockout_count":     int((risk_df["risk_level"] == "High Stockout Risk").sum()),
        "overstock_count":         int(risk_df["risk_level"].str.contains("Overstock").sum()),
        "healthy_count":           int((risk_df["risk_level"] == "Healthy").sum()),
        "total_skus":              len(risk_df),
    }

    logger.info(f"💰 Business Impact Summary:")
    for k, v in impact.items():
        logger.info(f"   {k}: {v:,.2f}" if isinstance(v, float) else f"   {k}: {v:,}")

    return impact


if __name__ == "__main__":
    df = run_risk_scoring()
    impact = calculate_business_impact(df)
    print(impact)
