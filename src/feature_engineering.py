"""
Project FORESIGHT – Feature Engineering Module
================================================
Generates 25+ predictive features for SKU-level demand forecasting.

Feature Groups:
  1. Temporal / Calendar
  2. Lag Features (1, 7, 14, 21, 28, 35 days)
  3. Rolling Statistics (7, 14, 28, 56 day windows)
  4. Price & Promotion
  5. Inventory & Supply Chain
  6. Demand Momentum & Volatility
  7. Revenue Features
  8. Lifecycle Features
  9. Encoding
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional

from src.utils import (
    logger, timer, safe_run, CONFIG,
    DATA_PROCESSED, UK_HOLIDAYS, save_csv, load_csv, reduce_mem_usage
)

LAG_DAYS = [1, 7, 14, 21, 28, 35]
ROLL_WINDOWS = [7, 14, 28, 56]


# ─────────────────────────────────────────────
# Helper: lag & rolling
# ─────────────────────────────────────────────
def _add_lags(df: pd.DataFrame, col: str, lags: List[int]) -> pd.DataFrame:
    """Add lag features grouped by SKU."""
    for lag in lags:
        df[f"{col}_lag{lag}"] = df.groupby("stock_code")[col].shift(lag)
    return df


def _add_rolling(df: pd.DataFrame, col: str, windows: List[int]) -> pd.DataFrame:
    """Add rolling mean, median, std features."""
    for w in windows:
        grp = df.groupby("stock_code")[col]
        df[f"{col}_roll_mean_{w}"] = grp.transform(
            lambda x: x.shift(1).rolling(w, min_periods=max(1, w // 2)).mean()
        )
        df[f"{col}_roll_median_{w}"] = grp.transform(
            lambda x: x.shift(1).rolling(w, min_periods=max(1, w // 2)).median()
        )
        df[f"{col}_roll_std_{w}"] = grp.transform(
            lambda x: x.shift(1).rolling(w, min_periods=max(1, w // 2)).std()
        )
    return df


# ─────────────────────────────────────────────
# Main Feature Engineering
# ─────────────────────────────────────────────
@timer
@safe_run
def run_feature_engineering(
    sales_daily: Optional[pd.DataFrame] = None,
    sku_master: Optional[pd.DataFrame] = None,
    calendar: Optional[pd.DataFrame] = None,
    inventory: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Merge all four tables and engineer comprehensive features.
    Returns feature-rich DataFrame ready for modeling.
    """
    # Load processed tables if not passed
    if sales_daily is None:
        sales_daily = load_csv(DATA_PROCESSED / "sales_daily.csv", parse_dates=["date"])
    if sku_master is None:
        sku_master = load_csv(DATA_PROCESSED / "sku_master.csv", parse_dates=["first_sale_date", "last_sale_date"])
    if calendar is None:
        calendar = load_csv(DATA_PROCESSED / "calendar.csv")
    if inventory is None:
        inventory = load_csv(DATA_PROCESSED / "inventory_snapshots.csv")

    # Ensure date types
    sales_daily["date"] = pd.to_datetime(sales_daily["date"])
    calendar["date"] = pd.to_datetime(calendar["date"])
    inventory["date"] = pd.to_datetime(inventory["date"])

    logger.info("Merging tables…")
    df = sales_daily.copy()

    # ── Merge Calendar ──────────────────────────────────────────────
    df = df.merge(calendar, on="date", how="left")

    # ── Merge SKU Master ─────────────────────────────────────────────
    sku_cols = [
        "stock_code", "category", "lead_time_days", "moq",
        "avg_unit_price", "abc_class", "days_since_launch",
        "supplier", "warehouse", "shelf_life_days"
    ]
    df = df.merge(sku_master[sku_cols], on="stock_code", how="left")

    # ── Merge Inventory ──────────────────────────────────────────────
    inv_cols = [
        "stock_code", "date", "closing_inventory", "coverage_days",
        "reorder_triggered", "reorder_point", "eoq", "inventory_value",
        "avg_daily_demand"
    ]
    df = df.merge(inventory[inv_cols], on=["stock_code", "date"], how="left")

    df = df.sort_values(["stock_code", "date"]).reset_index(drop=True)

    # ══════════════════════════════════════════════════════════════════
    # GROUP 1: Temporal Features
    # ══════════════════════════════════════════════════════════════════
    logger.info("Engineering temporal features…")
    # Already have: year, month, quarter, week_of_year, day_of_week,
    # is_weekend, is_holiday, is_black_friday, is_cyber_monday,
    # days_to_christmas, season, is_christmas_season

    # Fourier terms for weekly seasonality
    df["sin_week"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["cos_week"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # Fourier terms for yearly seasonality
    df["sin_year"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["cos_year"] = np.cos(2 * np.pi * df["day_of_year"] / 365)

    # ══════════════════════════════════════════════════════════════════
    # GROUP 2: Lag Features
    # ══════════════════════════════════════════════════════════════════
    logger.info("Engineering lag features…")
    df = _add_lags(df, "quantity", LAG_DAYS)
    df = _add_lags(df, "revenue", [7, 14, 28])

    # Same-weekday last week lag
    df["qty_lag7_same_dow"] = df.groupby(["stock_code", "day_of_week"])["quantity"].shift(1)

    # ══════════════════════════════════════════════════════════════════
    # GROUP 3: Rolling Statistics
    # ══════════════════════════════════════════════════════════════════
    logger.info("Engineering rolling features…")
    df = _add_rolling(df, "quantity", ROLL_WINDOWS)
    df = _add_rolling(df, "revenue", [7, 28])

    # Exponentially weighted moving average (α=0.3)
    df["qty_ewma_7"] = df.groupby("stock_code")["quantity"].transform(
        lambda x: x.shift(1).ewm(span=7, min_periods=3).mean()
    )
    df["qty_ewma_28"] = df.groupby("stock_code")["quantity"].transform(
        lambda x: x.shift(1).ewm(span=28, min_periods=7).mean()
    )

    # ══════════════════════════════════════════════════════════════════
    # GROUP 4: Price & Promotion Features
    # ══════════════════════════════════════════════════════════════════
    logger.info("Engineering price/promotion features…")
    df["price_change_7d"] = df.groupby("stock_code")["avg_unit_price_x"].transform(
        lambda x: x.pct_change(7).fillna(0)
    )
    df["price_vs_avg"] = df.groupby("stock_code")["avg_unit_price_x"].transform(
        lambda x: x / x.expanding().mean().shift(1)
    ).fillna(1.0)

    # Rename price column if duplicated
    if "avg_unit_price_x" in df.columns:
        df = df.rename(columns={"avg_unit_price_x": "unit_price", "avg_unit_price_y": "sku_avg_price"})
    elif "avg_unit_price" in df.columns:
        df = df.rename(columns={"avg_unit_price": "unit_price"})

    df["promotion_lag1"] = df.groupby("stock_code")["is_promotion"].shift(1).fillna(0)
    df["promotion_roll7"] = df.groupby("stock_code")["is_promotion"].transform(
        lambda x: x.shift(1).rolling(7, min_periods=1).sum()
    )

    # ══════════════════════════════════════════════════════════════════
    # GROUP 5: Inventory & Supply Chain Features
    # ══════════════════════════════════════════════════════════════════
    logger.info("Engineering inventory features…")
    # Stock turnover rate (sales / avg inventory over 28 days)
    df["stock_turnover_28"] = (
        df["quantity_roll_mean_28"] * 28 /
        (df["closing_inventory"].clip(lower=1))
    )

    # Inventory coverage vs lead time ratio
    df["coverage_vs_lead"] = df["coverage_days"] / df["lead_time_days"].clip(lower=1)

    # Reorder point gap
    df["inv_above_rop"] = (df["closing_inventory"] - df["reorder_point"]).fillna(0)

    # Days of supply
    df["days_of_supply"] = df["coverage_days"].fillna(0).clip(upper=365)

    # ══════════════════════════════════════════════════════════════════
    # GROUP 6: Demand Momentum & Volatility
    # ══════════════════════════════════════════════════════════════════
    logger.info("Engineering momentum/volatility features…")
    # Momentum: recent 7-day avg vs 28-day avg
    df["demand_momentum"] = (
        df["quantity_roll_mean_7"] / df["quantity_roll_mean_28"].clip(lower=0.001)
    ).fillna(1.0)

    # Volatility: coefficient of variation over 28 days
    df["demand_volatility"] = (
        df["quantity_roll_std_28"] / df["quantity_roll_mean_28"].clip(lower=0.001)
    ).fillna(0.0)

    # Demand acceleration (momentum derivative)
    df["demand_acceleration"] = df.groupby("stock_code")["demand_momentum"].diff().fillna(0)

    # Trend slope (linear regression slope over 14 days)
    def rolling_slope(series, window=14):
        def slope(y):
            if len(y) < 3:
                return 0.0
            x = np.arange(len(y))
            return np.polyfit(x, y, 1)[0]
        return series.rolling(window, min_periods=3).apply(slope, raw=True)

    df["demand_trend_slope"] = df.groupby("stock_code")["quantity"].transform(
        lambda x: rolling_slope(x.shift(1))
    ).fillna(0)

    # ══════════════════════════════════════════════════════════════════
    # GROUP 7: Revenue Features
    # ══════════════════════════════════════════════════════════════════
    logger.info("Engineering revenue features…")
    df["revenue_trend"] = df.groupby("stock_code")["revenue"].transform(
        lambda x: rolling_slope(x.shift(1))
    ).fillna(0)

    df["revenue_vs_28d_avg"] = (
        df["revenue"] / df["revenue_roll_mean_28"].clip(lower=0.001)
    ).fillna(1.0)

    # ══════════════════════════════════════════════════════════════════
    # GROUP 8: Lifecycle Features
    # ══════════════════════════════════════════════════════════════════
    logger.info("Engineering lifecycle features…")
    df["is_new_product"] = (df["days_since_launch"] <= 30).astype(int)
    df["is_mature_product"] = (df["days_since_launch"] > 180).astype(int)

    # Cumulative sales (running total per SKU)
    df["cumulative_qty"] = df.groupby("stock_code")["quantity"].cumsum()

    # ══════════════════════════════════════════════════════════════════
    # GROUP 9: Categorical Encoding
    # ══════════════════════════════════════════════════════════════════
    logger.info("Encoding categorical features…")
    # Ordinal encode ABC class
    abc_map = {"A": 3, "B": 2, "C": 1}
    df["abc_encoded"] = df["abc_class"].map(abc_map).fillna(1)

    # One-hot encode season (4 dummies)
    season_dummies = pd.get_dummies(df["season"], prefix="season", drop_first=False)
    df = pd.concat([df, season_dummies], axis=1)

    # Label encode category
    categories = df["category"].unique()
    cat_map = {c: i for i, c in enumerate(sorted(categories))}
    df["category_encoded"] = df["category"].map(cat_map).fillna(0).astype(int)

    # Day-of-week encoded
    df["dow_encoded"] = df["day_of_week"].astype(int)

    # ══════════════════════════════════════════════════════════════════
    # Finalize
    # ══════════════════════════════════════════════════════════════════
    # Fill any remaining NaNs in feature columns
    feature_cols = _get_feature_columns(df)
    df[feature_cols] = df[feature_cols].fillna(0)

    # Save
    df = reduce_mem_usage(df)
    out_path = DATA_PROCESSED / "features_engineered.csv"
    save_csv(df, out_path)
    logger.info(f"✅ Feature engineering complete: {df.shape[0]:,} rows × {df.shape[1]} columns")
    logger.info(f"   Feature columns: {len(feature_cols)}")

    return df


def _get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Return list of numeric feature columns (excludes identifiers and target)."""
    exclude = {
        "stock_code", "date", "invoice_date", "description",
        "quantity",  # target
        "month_name", "day_name", "season", "abc_class",
        "category", "supplier", "warehouse", "invoice_no",
        "num_orders", "num_customers",
    }
    return [
        c for c in df.columns
        if c not in exclude and df[c].dtype in [np.float64, np.float32, np.int64, np.int32, np.int8, bool]
    ]


def get_feature_list(df: pd.DataFrame) -> List[str]:
    """Public accessor for model-ready feature list."""
    return _get_feature_columns(df)


if __name__ == "__main__":
    run_feature_engineering()
