"""
Project FORESIGHT – Visualization Module
==========================================
Publication-quality Plotly charts used by both the Streamlit dashboard
and the reporting pipeline.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, List

# ─────────────────────────────────────────────
# Design System
# ─────────────────────────────────────────────
THEME = {
    "primary":    "#6366f1",   # indigo
    "secondary":  "#06b6d4",   # cyan
    "accent":     "#f59e0b",   # amber
    "success":    "#10b981",   # emerald
    "warning":    "#f97316",   # orange
    "danger":     "#ef4444",   # red
    "bg":         "#0f172a",   # slate-900
    "bg_card":    "#1e293b",   # slate-800
    "text":       "#f1f5f9",   # slate-100
    "muted":      "#94a3b8",   # slate-400
    "border":     "#334155",   # slate-700
}

RISK_COLORS = {
    "Critical Stockout Risk": "#ef4444",
    "High Stockout Risk":     "#f97316",
    "Moderate Stockout Risk": "#f59e0b",
    "Overstock Risk":         "#8b5cf6",
    "Mild Overstock Risk":    "#a78bfa",
    "Volatile Demand":        "#06b6d4",
    "Healthy":                "#10b981",
}

CHART_TEMPLATE = "plotly_dark"


def _base_layout(title: str = "", height: int = 450) -> dict:
    """Standard dark-theme layout."""
    return dict(
        title=dict(text=title, font=dict(size=18, color=THEME["text"]), x=0.02),
        height=height,
        paper_bgcolor=THEME["bg_card"],
        plot_bgcolor=THEME["bg_card"],
        font=dict(family="Inter, sans-serif", color=THEME["muted"], size=12),
        margin=dict(l=50, r=30, t=60, b=40),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=THEME["text"]),
        ),
        xaxis=dict(gridcolor=THEME["border"], zeroline=False),
        yaxis=dict(gridcolor=THEME["border"], zeroline=False),
    )


# ─────────────────────────────────────────────
# Revenue & Sales Trends
# ─────────────────────────────────────────────
def revenue_trend_chart(daily_df: pd.DataFrame) -> go.Figure:
    """Daily revenue trend with 7-day and 28-day moving averages."""
    d = (
        daily_df.groupby("date")["revenue"]
        .sum()
        .reset_index()
        .sort_values("date")
    )
    d["ma7"]  = d["revenue"].rolling(7, min_periods=1).mean()
    d["ma28"] = d["revenue"].rolling(28, min_periods=1).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=d["date"], y=d["revenue"],
        name="Daily Revenue",
        line=dict(color=THEME["primary"], width=1.5),
        opacity=0.5,
        fill="tozeroy",
        fillcolor=f"rgba(99,102,241,0.1)",
    ))
    fig.add_trace(go.Scatter(
        x=d["date"], y=d["ma7"],
        name="7-Day MA",
        line=dict(color=THEME["secondary"], width=2),
    ))
    fig.add_trace(go.Scatter(
        x=d["date"], y=d["ma28"],
        name="28-Day MA",
        line=dict(color=THEME["accent"], width=2, dash="dash"),
    ))
    fig.update_layout(**_base_layout("Revenue Trend", height=400))
    return fig


def sales_trend_chart(daily_df: pd.DataFrame) -> go.Figure:
    """Daily quantity sold trend."""
    d = daily_df.groupby("date")["quantity"].sum().reset_index().sort_values("date")
    d["ma7"] = d["quantity"].rolling(7, min_periods=1).mean()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=d["date"], y=d["quantity"],
        name="Units Sold",
        marker_color=THEME["primary"],
        opacity=0.7,
    ))
    fig.add_trace(go.Scatter(
        x=d["date"], y=d["ma7"],
        name="7-Day MA",
        line=dict(color=THEME["accent"], width=2),
    ))
    fig.update_layout(**_base_layout("Units Sold Trend", height=380))
    return fig


# ─────────────────────────────────────────────
# Category & Product Analysis
# ─────────────────────────────────────────────
def category_revenue_chart(sku_master: pd.DataFrame, daily_df: pd.DataFrame) -> go.Figure:
    """Treemap of revenue by category."""
    rev = daily_df.groupby("stock_code")["revenue"].sum().reset_index()
    merged = rev.merge(sku_master[["stock_code", "category"]], on="stock_code")
    cat_rev = merged.groupby("category")["revenue"].sum().reset_index()

    fig = px.treemap(
        cat_rev, path=["category"], values="revenue",
        color="revenue",
        color_continuous_scale=["#1e293b", "#6366f1", "#06b6d4"],
        title="Revenue by Category",
    )
    fig.update_layout(**{k: v for k, v in _base_layout().items() if k != "xaxis" and k != "yaxis"})
    return fig


def top_skus_chart(daily_df: pd.DataFrame, sku_master: pd.DataFrame, n: int = 20) -> go.Figure:
    """Horizontal bar chart: top N SKUs by revenue."""
    rev = daily_df.groupby("stock_code")["revenue"].sum().reset_index()
    rev = rev.merge(sku_master[["stock_code", "description"]].drop_duplicates(), on="stock_code", how="left")
    top = rev.nlargest(n, "revenue").sort_values("revenue")

    fig = go.Figure(go.Bar(
        x=top["revenue"],
        y=top["description"].str[:35],
        orientation="h",
        marker=dict(
            color=top["revenue"],
            colorscale=[[0, THEME["secondary"]], [1, THEME["primary"]]],
        ),
        text=top["revenue"].apply(lambda v: f"£{v:,.0f}"),
        textposition="outside",
        textfont=dict(color=THEME["text"]),
    ))
    fig.update_layout(**_base_layout(f"Top {n} Products by Revenue", height=max(400, n * 28)))
    return fig


def dead_stock_chart(risk_df: pd.DataFrame) -> go.Figure:
    """Pie chart showing dead stock vs active inventory."""
    dead = (risk_df["coverage_days"] > 365).sum()
    slow = ((risk_df["coverage_days"] > 90) & (risk_df["coverage_days"] <= 365)).sum()
    healthy = (risk_df["coverage_days"] <= 90).sum()

    fig = go.Figure(go.Pie(
        labels=["Dead Stock (>365 days)", "Slow Moving (90–365 days)", "Active (<90 days)"],
        values=[dead, slow, healthy],
        hole=0.55,
        marker=dict(colors=[THEME["danger"], THEME["warning"], THEME["success"]]),
        textinfo="percent+label",
        textfont=dict(color=THEME["text"]),
    ))
    fig.update_layout(**{k: v for k, v in _base_layout("Dead Stock Analysis").items() if k not in ("xaxis", "yaxis")})
    return fig


# ─────────────────────────────────────────────
# Seasonality & Correlation
# ─────────────────────────────────────────────
def seasonality_heatmap(daily_df: pd.DataFrame) -> go.Figure:
    """Heatmap: avg daily sales by month × day-of-week."""
    d = daily_df.copy()
    d["date"]        = pd.to_datetime(d["date"])
    d["month"]       = d["date"].dt.month_name().str[:3]
    d["day_of_week"] = d["date"].dt.day_name().str[:3]

    pivot = (
        d.groupby(["month", "day_of_week"])["quantity"]
        .mean()
        .unstack(fill_value=0)
    )
    # Order months
    month_order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    dow_order   = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    pivot = pivot.reindex([m for m in month_order if m in pivot.index])
    pivot = pivot.reindex(columns=[d for d in dow_order if d in pivot.columns])

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale=[[0, THEME["bg_card"]], [0.5, THEME["primary"]], [1, THEME["secondary"]]],
        text=pivot.values.round(1),
        texttemplate="%{text}",
        showscale=True,
    ))
    fig.update_layout(**_base_layout("Seasonality: Avg Units by Month × Day-of-Week", height=400))
    return fig


def correlation_heatmap(features_df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """Correlation heatmap of top N features vs target."""
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
    if "quantity" in numeric_cols:
        corr_with_target = features_df[numeric_cols].corr()["quantity"].abs().nlargest(top_n + 1)
        selected = corr_with_target.index.tolist()
        corr_matrix = features_df[selected].corr()
    else:
        corr_matrix = features_df[numeric_cols[:top_n]].corr()

    fig = go.Figure(go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns.tolist(),
        y=corr_matrix.index.tolist(),
        colorscale="RdBu",
        zmid=0,
        text=corr_matrix.values.round(2),
        texttemplate="%{text}",
        textfont=dict(size=9),
    ))
    fig.update_layout(**_base_layout(f"Feature Correlation Heatmap (Top {top_n})", height=500))
    return fig


# ─────────────────────────────────────────────
# Forecasting Charts
# ─────────────────────────────────────────────
def forecast_chart(
    history_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    sku: str,
    sku_name: str = "",
) -> go.Figure:
    """Historical + forecast chart for a single SKU."""
    hist = history_df[history_df["stock_code"] == sku].sort_values("date").tail(90)
    fcast = forecast_df[forecast_df["stock_code"] == sku].sort_values("date")

    fig = go.Figure()

    # Historical
    fig.add_trace(go.Scatter(
        x=hist["date"], y=hist["quantity"],
        name="Historical",
        line=dict(color=THEME["secondary"], width=2),
        mode="lines",
    ))

    # Forecast
    fig.add_trace(go.Scatter(
        x=fcast["date"], y=fcast["forecast_quantity"],
        name="Forecast",
        line=dict(color=THEME["accent"], width=2, dash="dash"),
        mode="lines+markers",
        marker=dict(size=5),
    ))

    # Confidence band (±20% simple band)
    fig.add_trace(go.Scatter(
        x=pd.concat([fcast["date"], fcast["date"][::-1]]),
        y=pd.concat([
            fcast["forecast_quantity"] * 1.2,
            fcast["forecast_quantity"][::-1] * 0.8
        ]),
        fill="toself",
        fillcolor=f"rgba(245,158,11,0.15)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Confidence Band",
        showlegend=True,
    ))

    # Vertical line at forecast start
    if len(hist) > 0 and len(fcast) > 0:
        split_date = fcast["date"].min()
        fig.add_vline(
            x=split_date, line_dash="dot",
            line_color=THEME["muted"], opacity=0.7,
            annotation_text="Forecast Start",
            annotation_font_color=THEME["muted"],
        )

    title = f"Demand Forecast: {sku_name or sku}"
    fig.update_layout(**_base_layout(title, height=420))
    return fig


def model_comparison_chart(comparison_df: pd.DataFrame) -> go.Figure:
    """Grouped bar chart comparing model metrics."""
    metrics = ["WAPE", "MAE", "RMSE"]
    fig = make_subplots(rows=1, cols=len(metrics), subplot_titles=metrics)

    colors = [THEME["primary"], THEME["secondary"], THEME["accent"],
              THEME["success"], THEME["warning"], THEME["danger"]]

    for i, metric in enumerate(metrics, 1):
        for j, row in comparison_df.iterrows():
            fig.add_trace(
                go.Bar(
                    x=[row["model"]],
                    y=[row[metric]],
                    name=row["model"],
                    marker_color=colors[j % len(colors)],
                    showlegend=(i == 1),
                    text=[f"{row[metric]:.2f}"],
                    textposition="outside",
                    textfont=dict(color=THEME["text"], size=10),
                ),
                row=1, col=i,
            )

    fig.update_layout(
        **_base_layout("Model Performance Comparison", height=450),
        barmode="group",
    )
    return fig


def feature_importance_chart(fi_df: pd.DataFrame, model_name: str = "", n: int = 20) -> go.Figure:
    """Horizontal bar chart of feature importance."""
    top = fi_df.nlargest(n, "importance").sort_values("importance")

    fig = go.Figure(go.Bar(
        x=top["importance"],
        y=top["feature"],
        orientation="h",
        marker=dict(
            color=top["importance"],
            colorscale=[[0, THEME["secondary"]], [1, THEME["primary"]]],
        ),
        text=top["importance"].round(4),
        textposition="outside",
        textfont=dict(color=THEME["text"], size=10),
    ))
    fig.update_layout(**_base_layout(f"Feature Importance – {model_name}", height=max(400, n * 25)))
    return fig


# ─────────────────────────────────────────────
# Risk & Inventory Charts
# ─────────────────────────────────────────────
def risk_distribution_chart(risk_df: pd.DataFrame) -> go.Figure:
    """Donut chart of risk level distribution."""
    dist = risk_df["risk_level"].value_counts().reset_index()
    dist.columns = ["risk_level", "count"]

    fig = go.Figure(go.Pie(
        labels=dist["risk_level"],
        values=dist["count"],
        hole=0.6,
        marker=dict(colors=[RISK_COLORS.get(r, THEME["muted"]) for r in dist["risk_level"]]),
        textinfo="percent+label",
        textfont=dict(color=THEME["text"]),
    ))
    fig.update_layout(**{k: v for k, v in _base_layout("SKU Risk Distribution").items()
                         if k not in ("xaxis", "yaxis")})
    return fig


def risk_scatter_chart(risk_df: pd.DataFrame) -> go.Figure:
    """Scatter: coverage days vs revenue at risk, colored by risk level."""
    fig = px.scatter(
        risk_df,
        x="coverage_days",
        y="revenue_at_risk",
        color="risk_level",
        size="inventory_value",
        hover_name="description",
        hover_data=["stock_code", "abc_class", "recommendation"],
        color_discrete_map=RISK_COLORS,
        log_y=True,
        title="Risk Landscape: Coverage Days vs Revenue at Risk",
    )
    fig.update_layout(**_base_layout(height=500))
    return fig


def inventory_coverage_chart(risk_df: pd.DataFrame) -> go.Figure:
    """Histogram of inventory coverage days."""
    fig = go.Figure(go.Histogram(
        x=risk_df["coverage_days"].clip(upper=365),
        nbinsx=40,
        marker_color=THEME["primary"],
        opacity=0.8,
    ))
    fig.add_vline(x=7,  line_dash="dash", line_color=THEME["danger"],  annotation_text="Critical (7d)")
    fig.add_vline(x=14, line_dash="dash", line_color=THEME["warning"], annotation_text="High Risk (14d)")
    fig.add_vline(x=90, line_dash="dash", line_color=THEME["accent"],  annotation_text="Overstock (90d)")
    fig.update_layout(**_base_layout("Inventory Coverage Distribution", height=380))
    return fig


def abc_analysis_chart(sku_master: pd.DataFrame, daily_df: pd.DataFrame) -> go.Figure:
    """Pareto/ABC analysis chart."""
    rev = daily_df.groupby("stock_code")["revenue"].sum().reset_index()
    merged = rev.merge(sku_master[["stock_code", "abc_class"]], on="stock_code")
    abc_rev = merged.groupby("abc_class")["revenue"].agg(["sum", "count"]).reset_index()
    abc_rev.columns = ["abc_class", "revenue", "sku_count"]
    abc_rev["revenue_pct"] = 100 * abc_rev["revenue"] / abc_rev["revenue"].sum()

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=abc_rev["abc_class"],
        y=abc_rev["revenue"],
        name="Revenue",
        marker_color=[THEME["primary"], THEME["secondary"], THEME["accent"]],
        text=abc_rev["revenue_pct"].apply(lambda v: f"{v:.1f}%"),
        textposition="outside",
        textfont=dict(color=THEME["text"]),
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=abc_rev["abc_class"],
        y=abc_rev["sku_count"],
        name="SKU Count",
        line=dict(color=THEME["success"], width=2),
        mode="lines+markers",
    ), secondary_y=True)
    fig.update_layout(**_base_layout("ABC Classification Analysis", height=400))
    return fig


def business_impact_waterfall(impact: dict) -> go.Figure:
    """Waterfall chart for business impact."""
    measures = ["absolute", "relative", "relative", "relative", "relative", "total"]
    labels = [
        "Current Revenue at Risk",
        "Stockout Recovery",
        "Overstock Savings",
        "Working Capital",
        "Other",
        "Net Benefit",
    ]
    values = [
        -impact.get("revenue_at_risk", 0),
        impact.get("potential_sales_increase", 0),
        impact.get("overstock_cost_monthly", 0),
        impact.get("working_capital_saved", 0),
        0,
        impact.get("expected_profit_improvement", 0),
    ]

    fig = go.Figure(go.Waterfall(
        name="Business Impact",
        orientation="v",
        measure=measures,
        x=labels,
        y=values,
        text=[f"£{abs(v):,.0f}" for v in values],
        textposition="outside",
        connector=dict(line=dict(color=THEME["border"])),
        decreasing=dict(marker=dict(color=THEME["danger"])),
        increasing=dict(marker=dict(color=THEME["success"])),
        totals=dict(marker=dict(color=THEME["primary"])),
        textfont=dict(color=THEME["text"]),
    ))
    fig.update_layout(**_base_layout("Business Impact Waterfall (£)", height=450))
    return fig


def promotion_impact_chart(daily_df: pd.DataFrame) -> go.Figure:
    """Compare sales on promotion vs non-promotion days."""
    d = daily_df.copy()
    d["promo"] = d["is_promotion"].map({0: "Regular", 1: "Promotion"})
    promo_stats = (
        d.groupby("promo")["quantity"]
        .agg(["mean", "median", "sum"])
        .reset_index()
    )
    fig = go.Figure()
    for col, label, color in [("mean", "Avg Units", THEME["primary"]),
                               ("median", "Median Units", THEME["secondary"])]:
        fig.add_trace(go.Bar(
            x=promo_stats["promo"],
            y=promo_stats[col],
            name=label,
            marker_color=color,
            text=promo_stats[col].round(1),
            textposition="outside",
            textfont=dict(color=THEME["text"]),
        ))
    fig.update_layout(**_base_layout("Promotion Impact on Sales", height=380))
    return fig
