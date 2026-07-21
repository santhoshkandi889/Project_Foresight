"""
Project FORESIGHT – Preprocessing Module
==========================================
Transforms raw Online Retail II CSV into the four Project FORESIGHT tables:

  1. sales_daily.csv          – daily SKU-level sales aggregations
  2. sku_master.csv           – product catalog + simulated attributes
  3. calendar.csv             – date dimension with holiday/seasonal flags
  4. inventory_snapshots.csv  – simulated daily inventory levels

DOCUMENTED ASSUMPTIONS
======================
A. Categories
   - Mapped from StockCode character-prefix patterns.
   - StockCodes starting with digits → sub-categories based on numeric range.
   - POST, DOT, BANK, PADS etc. → "Services/Other".
   - Unknown → "Miscellaneous".

B. Inventory Levels
   - Starting inventory = 12 weeks of average weekly demand per SKU.
   - Each day's closing inventory = previous day's inventory + replenishments − sales.
   - Replenishment triggered when inventory ≤ reorder point; qty = EOQ.

C. Reorder Points
   - ROP = avg_daily_demand × (lead_time_days + safety_days)
   - safety_days = 1.5 × std_daily_demand / avg_daily_demand (capped 2–14).

D. Lead Times
   - Randomly sampled from U(3, 14) per SKU, seeded at 42 for reproducibility.

E. Promotions
   - Flagged when unit_price < 0.85 × 30-day rolling average price (>15% discount).

F. EOQ (Economic Order Quantity)
   - EOQ = sqrt(2 × annual_demand × ordering_cost / holding_cost_per_unit)
   - ordering_cost = £50, holding_cost_rate = 20% of unit price.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple

from src.utils import (
    logger, timer, safe_run, CONFIG,
    DATA_RAW, DATA_PROCESSED, UK_HOLIDAYS,
    save_csv, load_csv
)

# ─────────────────────────────────────────────
# Category Mapping
# ─────────────────────────────────────────────
CATEGORY_MAP = {
    "home_decor":    ["HOME", "DECOR", "FRAME", "SIGN", "WALL", "MIRROR", "CANDLE", "LAMP"],
    "kitchen":       ["MUG", "CUP", "PLATE", "BOWL", "KITCHEN", "COOK", "CAKE", "BAKING"],
    "garden":        ["GARDEN", "PLANT", "FLOWER", "SEED", "OUTDOOR"],
    "gifts":         ["GIFT", "WRAP", "BOX", "BAG", "BIRTHDAY", "CHRISTM", "XMAS", "PARTY"],
    "stationery":    ["PAPER", "CARD", "BOOK", "PEN", "PENCIL", "NOTE", "DIARY"],
    "toys":          ["TOY", "GAME", "DOLL", "PLAY", "PUZZLE", "KIDS"],
    "fashion":       ["SCARF", "HAT", "BAG", "PURSE", "JEWEL", "NECKLACE", "BRACELET"],
    "storage":       ["BOX", "TIN", "BASKET", "STORAGE", "RACK", "SHELF", "ORGANIS"],
    "services":      ["POST", "DOT", "BANK", "PADS", "ADJUST", "M", "S", "AMAZFIT"],
}

NON_PRODUCT_CODES = {"POST", "DOT", "BANK", "PADS", "M", "S", "ADJUST", "AMAZFIT", "gift"}


def _assign_category(description: str, stock_code: str) -> str:
    """Assign product category based on description keywords."""
    if not isinstance(description, str):
        return "miscellaneous"
    desc_upper = description.upper()
    for category, keywords in CATEGORY_MAP.items():
        for kw in keywords:
            if kw in desc_upper:
                return category
    # Fallback by stock_code prefix
    sc = str(stock_code).strip()
    if sc.isdigit():
        n = int(sc[:2]) if len(sc) >= 2 else 0
        if n < 22:
            return "home_decor"
        elif n < 50:
            return "gifts"
        elif n < 75:
            return "kitchen"
        else:
            return "storage"
    return "miscellaneous"


# ─────────────────────────────────────────────
# Step 1: Load & Clean Raw Data
# ─────────────────────────────────────────────
@timer
def load_and_clean_raw(filepath: Path) -> pd.DataFrame:
    """Load raw Online Retail II CSV and apply cleaning rules."""
    logger.info(f"Loading raw data from: {filepath}")

    # Try multiple encodings
    for enc in ["utf-8", "latin-1", "cp1252"]:
        try:
            df = pd.read_csv(filepath, encoding=enc, low_memory=False)
            logger.info(f"Loaded with encoding={enc}: {len(df):,} rows, {len(df.columns)} cols")
            break
        except UnicodeDecodeError:
            continue

    # Standardise column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Rename to FORESIGHT schema
    rename_map = {
        "invoiceno":     "invoice_no",
        "invoice_no":    "invoice_no",
        "stockcode":     "stock_code",
        "stock_code":    "stock_code",
        "description":   "description",
        "quantity":      "quantity",
        "invoicedate":   "invoice_date",
        "invoice_date":  "invoice_date",
        "unitprice":     "unit_price",
        "unit_price":    "unit_price",
        "customerid":    "customer_id",
        "customer_id":   "customer_id",
        "country":       "country",
    }
    df = df.rename(columns={c: rename_map[c] for c in df.columns if c in rename_map})

    # Parse dates
    df["invoice_date"] = pd.to_datetime(df["invoice_date"], errors="coerce")
    df = df.dropna(subset=["invoice_date"])

    # Remove cancellations (InvoiceNo starting with 'C')
    before = len(df)
    df = df[~df["invoice_no"].astype(str).str.startswith("C")]
    logger.info(f"Removed {before - len(df):,} cancellation rows")

    # Remove non-product stock codes
    df = df[~df["stock_code"].astype(str).str.strip().isin(NON_PRODUCT_CODES)]

    # Numeric cleaning
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce")

    # Remove returns (negative qty) and zero-price items
    df = df[df["quantity"] > 0]
    df = df[df["unit_price"] > 0]

    # Drop nulls in critical columns
    df = df.dropna(subset=["stock_code", "quantity", "unit_price"])

    # Clean stock_code
    df["stock_code"] = df["stock_code"].astype(str).str.strip().str.upper()

    # Clean description
    df["description"] = df["description"].astype(str).str.strip().str.title()

    # Filter UK only for cleaner analysis (majority of data)
    df = df[df["country"] == "United Kingdom"] if "country" in df.columns else df

    # Extract date column
    df["date"] = df["invoice_date"].dt.date.astype(str)

    # Revenue
    df["revenue"] = df["quantity"] * df["unit_price"]

    # Remove duplicates
    before = len(df)
    df = df.drop_duplicates()
    logger.info(f"Removed {before - len(df):,} duplicate rows")

    logger.info(f"Clean dataset: {len(df):,} rows | Date range: {df['date'].min()} → {df['date'].max()}")
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────
# Step 2: Build sales_daily.csv
# ─────────────────────────────────────────────
@timer
def build_sales_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to SKU × date level with promotion flag."""
    logger.info("Building sales_daily table…")

    # Daily aggregation
    daily = (
        df.groupby(["stock_code", "date"])
        .agg(
            quantity=("quantity", "sum"),
            revenue=("revenue", "sum"),
            avg_unit_price=("unit_price", "mean"),
            num_orders=("invoice_no", "nunique"),
            num_customers=("customer_id", "nunique") if "customer_id" in df.columns else ("invoice_no", "nunique"),
        )
        .reset_index()
    )
    daily["date"] = pd.to_datetime(daily["date"])

    # Promotion flag: price < 85% of 30-day rolling average
    daily = daily.sort_values(["stock_code", "date"])
    daily["price_30d_avg"] = (
        daily.groupby("stock_code")["avg_unit_price"]
        .transform(lambda x: x.shift(1).rolling(30, min_periods=5).mean())
    )
    daily["is_promotion"] = (
        (daily["price_30d_avg"].notna()) &
        (daily["avg_unit_price"] < 0.85 * daily["price_30d_avg"])
    ).astype(int)
    daily = daily.drop(columns=["price_30d_avg"])

    # Fill missing SKU-date combos with 0 sales (sparse → dense)
    all_dates = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
    all_skus = daily["stock_code"].unique()
    idx = pd.MultiIndex.from_product([all_skus, all_dates], names=["stock_code", "date"])
    daily = (
        daily.set_index(["stock_code", "date"])
        .reindex(idx, fill_value=0)
        .reset_index()
    )
    # Restore non-zero prices in zero-sales rows
    daily["avg_unit_price"] = daily.groupby("stock_code")["avg_unit_price"].transform(
        lambda x: x.replace(0, np.nan).ffill().bfill()
    )
    daily["revenue"] = daily["quantity"] * daily["avg_unit_price"]

    daily = daily.sort_values(["stock_code", "date"]).reset_index(drop=True)
    logger.info(f"sales_daily: {len(daily):,} rows | {daily['stock_code'].nunique()} SKUs | {daily['date'].nunique()} dates")
    return daily


# ─────────────────────────────────────────────
# Step 3: Build sku_master.csv
# ─────────────────────────────────────────────
@timer
def build_sku_master(df: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """Build product master table with simulated attributes."""
    logger.info("Building sku_master table…")
    rng = np.random.default_rng(CONFIG["RANDOM_SEED"])

    # Base info per SKU
    sku_info = (
        df.groupby("stock_code")
        .agg(
            description=("description", lambda x: x.mode().iloc[0] if len(x) > 0 else "Unknown"),
            avg_unit_price=("unit_price", "median"),
            min_price=("unit_price", "min"),
            max_price=("unit_price", "max"),
            total_revenue=("revenue", "sum"),
            total_quantity=("quantity", "sum"),
            first_sale_date=("invoice_date", "min"),
            last_sale_date=("invoice_date", "max"),
        )
        .reset_index()
    )

    # Category assignment
    sku_info["category"] = sku_info.apply(
        lambda r: _assign_category(r["description"], r["stock_code"]), axis=1
    )

    # Number of unique SKUs
    n = len(sku_info)

    # Simulated lead time: U(3, 14) days per SKU
    sku_info["lead_time_days"] = rng.integers(3, 15, size=n)

    # Simulated reorder quantity (MOQ): multiples of 12 (typical case pack)
    sku_info["moq"] = (rng.integers(1, 10, size=n) * 12)

    # ABC classification based on revenue (80/15/5 rule)
    sku_info = sku_info.sort_values("total_revenue", ascending=False).reset_index(drop=True)
    cumrev = sku_info["total_revenue"].cumsum()
    total_rev = sku_info["total_revenue"].sum()
    sku_info["abc_class"] = "C"
    sku_info.loc[cumrev <= 0.80 * total_rev, "abc_class"] = "A"
    sku_info.loc[
        (cumrev > 0.80 * total_rev) & (cumrev <= 0.95 * total_rev), "abc_class"
    ] = "B"

    # Days since launch
    ref_date = pd.Timestamp("2011-12-09")  # last date in dataset
    sku_info["first_sale_date"] = pd.to_datetime(sku_info["first_sale_date"])
    sku_info["days_since_launch"] = (ref_date - sku_info["first_sale_date"]).dt.days

    # Simulated supplier name (10 suppliers)
    suppliers = [f"Supplier_{chr(65+i)}" for i in range(10)]
    sku_info["supplier"] = rng.choice(suppliers, size=n)

    # Simulated warehouse location
    warehouses = ["Warehouse_East", "Warehouse_West", "Warehouse_North", "Warehouse_South"]
    sku_info["warehouse"] = rng.choice(warehouses, size=n)

    # Simulated shelf life (days): perishable vs durable
    sku_info["shelf_life_days"] = np.where(
        sku_info["category"].isin(["kitchen", "food"]),
        rng.integers(30, 180, size=n),
        rng.integers(365, 1825, size=n),
    )

    logger.info(f"sku_master: {len(sku_info):,} SKUs across {sku_info['category'].nunique()} categories")
    return sku_info.reset_index(drop=True)


# ─────────────────────────────────────────────
# Step 4: Build calendar.csv
# ─────────────────────────────────────────────
@timer
def build_calendar(daily: pd.DataFrame) -> pd.DataFrame:
    """Build date dimension table with rich temporal features."""
    logger.info("Building calendar table…")

    dates = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
    cal = pd.DataFrame({"date": dates})

    cal["year"]           = cal["date"].dt.year
    cal["month"]          = cal["date"].dt.month
    cal["month_name"]     = cal["date"].dt.month_name()
    cal["quarter"]        = cal["date"].dt.quarter
    cal["week_of_year"]   = cal["date"].dt.isocalendar().week.astype(int)
    cal["day_of_week"]    = cal["date"].dt.dayofweek          # 0=Mon
    cal["day_name"]       = cal["date"].dt.day_name()
    cal["day_of_month"]   = cal["date"].dt.day
    cal["day_of_year"]    = cal["date"].dt.dayofyear
    cal["is_weekend"]     = cal["day_of_week"].isin([5, 6]).astype(int)
    cal["is_month_start"] = cal["date"].dt.is_month_start.astype(int)
    cal["is_month_end"]   = cal["date"].dt.is_month_end.astype(int)
    cal["is_quarter_end"] = cal["date"].dt.is_quarter_end.astype(int)

    # UK Public Holidays
    cal["is_holiday"] = cal["date"].astype(str).isin(UK_HOLIDAYS).astype(int)

    # Seasons (Northern Hemisphere)
    cal["season"] = pd.cut(
        cal["month"],
        bins=[0, 2, 5, 8, 11, 12],
        labels=["winter", "spring", "summer", "autumn", "winter"],
        ordered=False,
        right=True,
    )
    # Fix: month 12 → winter
    cal["season"] = cal["season"].astype(str)
    cal.loc[cal["month"] == 12, "season"] = "winter"

    # Christmas season (Nov–Dec)
    cal["is_christmas_season"] = cal["month"].isin([11, 12]).astype(int)

    # Black Friday (4th Friday in November)
    def is_black_friday(dt):
        if dt.month != 11:
            return 0
        fridays = [d for d in pd.date_range(f"{dt.year}-11-01", f"{dt.year}-11-30", freq="W-FRI")]
        return 1 if len(fridays) >= 4 and dt == fridays[3] else 0

    cal["is_black_friday"] = cal["date"].apply(is_black_friday)

    # Cyber Monday (Monday after Black Friday)
    cal["is_cyber_monday"] = cal["date"].apply(
        lambda dt: 1 if dt.month == 11 and dt.weekday() == 0 and is_black_friday(dt - pd.Timedelta(3, "D")) else 0
    )

    # Days to Christmas
    cal["days_to_christmas"] = cal["date"].apply(
        lambda dt: (pd.Timestamp(f"{dt.year}-12-25") - dt).days
    )
    cal["days_to_christmas"] = cal["days_to_christmas"].clip(lower=0)

    # Days to New Year
    cal["days_to_newyear"] = cal["date"].apply(
        lambda dt: (pd.Timestamp(f"{dt.year + 1 if dt.month >= 11 else dt.year}-01-01") - dt).days
    )
    cal["days_to_newyear"] = cal["days_to_newyear"].clip(lower=0)

    cal["date"] = cal["date"].astype(str)
    logger.info(f"calendar: {len(cal):,} rows | {cal['year'].nunique()} years")
    return cal


# ─────────────────────────────────────────────
# Step 5: Build inventory_snapshots.csv
# ─────────────────────────────────────────────
@timer
def build_inventory_snapshots(daily: pd.DataFrame, sku_master: pd.DataFrame) -> pd.DataFrame:
    """
    Simulate daily inventory levels using a deterministic replenishment policy.

    Policy: Continuous Review (s, Q) – reorder when stock ≤ ROP, order EOQ units.
    Lead time: fixed per SKU from sku_master.
    """
    logger.info("Simulating inventory snapshots…")

    # Merge lead time info
    lt_map = sku_master.set_index("stock_code")["lead_time_days"].to_dict()
    price_map = sku_master.set_index("stock_code")["avg_unit_price"].to_dict()

    snapshots = []
    skus = daily["stock_code"].unique()

    for sku in skus:
        sku_df = daily[daily["stock_code"] == sku].sort_values("date").copy()
        sku_df = sku_df.reset_index(drop=True)

        avg_daily_demand = sku_df["quantity"].mean()
        std_daily_demand = sku_df["quantity"].std() or (avg_daily_demand * 0.3)
        lead_time = lt_map.get(sku, 7)
        unit_price = price_map.get(sku, 1.0)

        # ROP = avg_demand × (lead_time + safety_days)
        safety_factor = 1.5
        rop = avg_daily_demand * (lead_time * safety_factor)
        rop = max(rop, avg_daily_demand * 2)  # at least 2 days stock

        # EOQ
        annual_demand = avg_daily_demand * 365
        ordering_cost = 50.0
        holding_cost = max(unit_price * 0.20, 0.01)
        eoq = np.sqrt(2 * annual_demand * ordering_cost / holding_cost)
        eoq = max(eoq, lead_time * avg_daily_demand * 2, 12)  # floor

        # Starting inventory = 12 weeks of demand
        inventory = avg_daily_demand * 84

        pending_orders = []  # list of (arrival_date_idx, qty)

        for i, row in sku_df.iterrows():
            # Receive pending orders arriving today
            arrived = [qty for (arr_idx, qty) in pending_orders if arr_idx <= i]
            pending_orders = [(arr_idx, qty) for (arr_idx, qty) in pending_orders if arr_idx > i]
            inventory += sum(arrived)

            # Consume sales
            sold = min(inventory, row["quantity"])
            inventory -= sold
            inventory = max(inventory, 0)

            # Check reorder
            reorder_triggered = 0
            if inventory <= rop and not any(arr_idx > i for arr_idx, _ in pending_orders):
                order_qty = round(eoq)
                arrival_idx = i + lead_time
                pending_orders.append((arrival_idx, order_qty))
                reorder_triggered = 1

            # Inventory coverage (days of supply)
            coverage_days = (inventory / avg_daily_demand) if avg_daily_demand > 0 else 999

            snapshots.append({
                "stock_code": sku,
                "date": row["date"].strftime("%Y-%m-%d") if hasattr(row["date"], "strftime") else str(row["date"]),
                "opening_inventory": inventory + sold,
                "units_sold": sold,
                "closing_inventory": inventory,
                "reorder_triggered": reorder_triggered,
                "reorder_point": round(rop, 2),
                "eoq": round(eoq, 2),
                "inventory_value": round(inventory * unit_price, 2),
                "coverage_days": round(coverage_days, 2),
                "avg_daily_demand": round(avg_daily_demand, 4),
            })

    inv_df = pd.DataFrame(snapshots)
    logger.info(f"inventory_snapshots: {len(inv_df):,} rows")
    return inv_df


# ─────────────────────────────────────────────
# Main Preprocessing Entry Point
# ─────────────────────────────────────────────
@timer
@safe_run
def run_preprocessing(raw_filepath: Path = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Full preprocessing pipeline.
    Returns (sales_daily, sku_master, calendar, inventory_snapshots).
    """
    # Auto-detect raw file
    if raw_filepath is None:
        candidates = list(DATA_RAW.glob("*.csv")) + list(DATA_RAW.glob("*.xlsx"))
        if not candidates:
            raise FileNotFoundError(
                f"No raw data file found in {DATA_RAW}. "
                "Please place the Online Retail II CSV file there."
            )
        raw_filepath = candidates[0]
        logger.info(f"Auto-detected raw file: {raw_filepath.name}")

    # Step 1: Clean
    df_clean = load_and_clean_raw(raw_filepath)

    # Step 2: Sales daily
    sales_daily = build_sales_daily(df_clean)
    save_csv(sales_daily, DATA_PROCESSED / "sales_daily.csv")

    # Step 3: SKU master
    sku_master = build_sku_master(df_clean, sales_daily)
    save_csv(sku_master, DATA_PROCESSED / "sku_master.csv")

    # Step 4: Calendar
    calendar = build_calendar(sales_daily)
    save_csv(calendar, DATA_PROCESSED / "calendar.csv")

    # Step 5: Inventory snapshots
    inv_snaps = build_inventory_snapshots(sales_daily, sku_master)
    save_csv(inv_snaps, DATA_PROCESSED / "inventory_snapshots.csv")

    logger.info("✅ All four FORESIGHT tables saved to data/processed/")

    # Print assumption summary
    _print_assumptions()

    return sales_daily, sku_master, calendar, inv_snaps


def _print_assumptions():
    logger.info("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DOCUMENTED SIMULATION ASSUMPTIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[A] Categories:  Keyword matching on product descriptions
[B] Inventory:   Starting stock = 12-week avg demand; (s,Q) policy
[C] ROP:         avg_daily_demand × lead_time × 1.5 safety factor
[D] Lead Times:  Uniform(3–14) days, seeded=42 (reproducible)
[E] Promotions:  Flagged when price < 85% of 30-day rolling avg
[F] EOQ:         sqrt(2 × D × S / H); ordering cost=£50, H=20% of price
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")


if __name__ == "__main__":
    run_preprocessing()
