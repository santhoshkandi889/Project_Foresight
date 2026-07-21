"""
Project FORESIGHT – Utility Module
===================================
Shared utilities: logging, config, path management, decorators.
"""

import os
import logging
import time
import functools
from pathlib import Path
from datetime import datetime
from typing import Any, Callable, Optional

from loguru import logger


# ─────────────────────────────────────────────
# Project Paths
# ─────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
LOGS_DIR = PROJECT_ROOT / "logs"

for _dir in [DATA_RAW, DATA_PROCESSED, MODELS_DIR, REPORTS_DIR, LOGS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# Loguru Setup
# ─────────────────────────────────────────────
LOG_FILE = LOGS_DIR / f"foresight_{datetime.now().strftime('%Y%m%d')}.log"

logger.remove()
logger.add(
    LOG_FILE,
    rotation="10 MB",
    retention="30 days",
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{function}:{line} | {message}",
)
logger.add(
    lambda msg: print(msg, end=""),
    level="INFO",
    format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | <cyan>{message}</cyan>",
    colorize=True,
)


# ─────────────────────────────────────────────
# Configuration Constants
# ─────────────────────────────────────────────
CONFIG = {
    # Data
    "TARGET_COLUMN": "quantity",
    "DATE_COLUMN": "date",
    "SKU_COLUMN": "stock_code",
    # Forecasting
    "FORECAST_HORIZON": 28,          # days
    "CV_FOLDS": 1,
    "MIN_HISTORY_DAYS": 60,
    # Risk thresholds
    "STOCKOUT_THRESHOLD_DAYS": 7,
    "OVERSTOCK_THRESHOLD_DAYS": 90,
    "CRITICAL_RISK_THRESHOLD": 0.75,
    "HIGH_RISK_THRESHOLD": 0.50,
    # Business
    "HOLDING_COST_RATE": 0.20,       # 20% annual holding cost
    "STOCKOUT_COST_MULTIPLIER": 2.0, # lost revenue × 2 for brand impact
    # Simulation seeds
    "RANDOM_SEED": 42,
}


# ─────────────────────────────────────────────
# Decorators
# ─────────────────────────────────────────────
def timer(func: Callable) -> Callable:
    """Log execution time of a function."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        logger.info(f"▶  Starting: {func.__qualname__}")
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info(f"✔  Completed: {func.__qualname__} in {elapsed:.2f}s")
        return result
    return wrapper


def safe_run(func: Callable) -> Callable:
    """Catch and log exceptions without crashing the pipeline."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            logger.exception(f"✘  Error in {func.__qualname__}: {exc}")
            raise
    return wrapper


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def memory_usage_mb(df) -> float:
    """Return DataFrame memory usage in MB."""
    return df.memory_usage(deep=True).sum() / 1_048_576


def reduce_mem_usage(df):
    """Downcast numeric columns to reduce memory footprint."""
    import pandas as pd
    import numpy as np

    before = memory_usage_mb(df)
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col]):
            c_min = df[col].min()
            c_max = df[col].max()
            if pd.api.types.is_integer_dtype(df[col]):
                for dtype in [np.int8, np.int16, np.int32, np.int64]:
                    if c_min > np.iinfo(dtype).min and c_max < np.iinfo(dtype).max:
                        df[col] = df[col].astype(dtype)
                        break
            elif pd.api.types.is_float_dtype(df[col]):
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
    after = memory_usage_mb(df)
    logger.debug(f"Memory reduced: {before:.1f} MB → {after:.1f} MB ({100*(before-after)/before:.1f}% saved)")
    return df


def save_csv(df, path: Path, index: bool = False) -> None:
    """Save DataFrame to CSV with logging."""
    df.to_csv(path, index=index)
    logger.info(f"💾 Saved {len(df):,} rows → {path.name}")


def load_csv(path: Path, **kwargs):
    """Load CSV with logging."""
    import pandas as pd
    df = pd.read_csv(path, **kwargs)
    logger.info(f"📂 Loaded {len(df):,} rows from {path.name}")
    return df


def get_processed_path(filename: str) -> Path:
    return DATA_PROCESSED / filename


def get_model_path(filename: str) -> Path:
    return MODELS_DIR / filename


# ─────────────────────────────────────────────
# UK Bank Holidays (2009–2011 for this dataset)
# ─────────────────────────────────────────────
UK_HOLIDAYS = {
    # 2009
    "2009-01-01", "2009-04-10", "2009-04-13", "2009-05-04",
    "2009-05-25", "2009-08-31", "2009-12-25", "2009-12-28",
    # 2010
    "2010-01-01", "2010-04-02", "2010-04-05", "2010-05-03",
    "2010-05-31", "2010-08-30", "2010-12-27", "2010-12-28",
    # 2011
    "2011-01-03", "2011-04-22", "2011-04-25", "2011-04-29",
    "2011-05-02", "2011-05-30", "2011-08-29", "2011-12-26", "2011-12-27",
}

def is_uk_holiday(date_str: str) -> bool:
    return str(date_str)[:10] in UK_HOLIDAYS


if __name__ == "__main__":
    logger.info("Utils module loaded successfully.")
    logger.info(f"Project root: {PROJECT_ROOT}")
