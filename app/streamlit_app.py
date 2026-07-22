"""
Project FORESIGHT – Streamlit Dashboard
=========================================
NorthBay Living | Demand & Inventory Intelligence

12-page enterprise-grade dashboard with:
  🏠 Home
  📂 Dataset Overview
  📈 EDA
  📊 Demand Forecast
  📦 Inventory Analytics
  ⚠  Risk Dashboard
  📉 Model Comparison
  🎯 Decision Matrix
  💡 Recommendations
  💰 Business Impact
  🌐 API Testing
  📄 Project Summary
"""

import json
import time
import warnings
import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from pathlib import Path
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="FORESIGHT | NorthBay Living",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Design System – Inject CSS
# ─────────────────────────────────────────────
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

:root {
  --primary:   #6366f1;
  --secondary: #06b6d4;
  --accent:    #f59e0b;
  --success:   #10b981;
  --warning:   #f97316;
  --danger:    #ef4444;
  --bg:        #0f172a;
  --bg-card:   #1e293b;
  --bg-hover:  #273549;
  --text:      #f1f5f9;
  --muted:     #94a3b8;
  --border:    #334155;
}

html, body, [class*="css"] {
  font-family: 'Inter', sans-serif;
  background-color: var(--bg);
  color: var(--text);
}

/* Sidebar */
.css-1d391kg, [data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%) !important;
  border-right: 1px solid var(--border) !important;
}

/* Hide default streamlit header/footer */
#MainMenu, footer, header { visibility: hidden; }

/* Main area */
.main .block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 100%; }

/* KPI Cards */
.kpi-card {
  background: linear-gradient(135deg, var(--bg-card) 0%, #273549 100%);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 20px 24px;
  text-align: center;
  transition: all 0.25s ease;
  position: relative;
  overflow: hidden;
}
.kpi-card::before {
  content: '';
  position: absolute; top: 0; left: 0; right: 0;
  height: 3px;
  background: linear-gradient(90deg, var(--primary), var(--secondary));
}
.kpi-card:hover {
  transform: translateY(-3px);
  border-color: var(--primary);
  box-shadow: 0 8px 30px rgba(99,102,241,0.3);
}
.kpi-value {
  font-size: 2rem; font-weight: 800;
  background: linear-gradient(135deg, var(--primary), var(--secondary));
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  margin: 8px 0 4px;
}
.kpi-label { font-size: 0.78rem; color: var(--muted); font-weight: 500; letter-spacing: 0.08em; text-transform: uppercase; }
.kpi-delta { font-size: 0.82rem; margin-top: 6px; font-weight: 600; }
.kpi-delta.positive { color: var(--success); }
.kpi-delta.negative { color: var(--danger); }
.kpi-delta.neutral  { color: var(--muted); }

/* Section header */
.section-header {
  font-size: 1.5rem; font-weight: 700; color: var(--text);
  margin: 1.5rem 0 1rem;
  padding-bottom: 0.5rem;
  border-bottom: 2px solid var(--border);
  display: flex; align-items: center; gap: 0.5rem;
}

/* Risk badges */
.badge {
  display: inline-block; padding: 3px 10px; border-radius: 99px;
  font-size: 0.72rem; font-weight: 600; letter-spacing: 0.05em;
}
.badge-danger  { background: rgba(239,68,68,0.2);  color: #ef4444; }
.badge-warning { background: rgba(249,115,22,0.2); color: #f97316; }
.badge-amber   { background: rgba(245,158,11,0.2); color: #f59e0b; }
.badge-success { background: rgba(16,185,129,0.2); color: #10b981; }
.badge-purple  { background: rgba(139,92,246,0.2); color: #8b5cf6; }
.badge-cyan    { background: rgba(6,182,212,0.2);  color: #06b6d4; }

/* Info boxes */
.info-box {
  background: rgba(99,102,241,0.08);
  border: 1px solid rgba(99,102,241,0.3);
  border-radius: 12px; padding: 16px 20px; margin: 12px 0;
}
.warning-box {
  background: rgba(245,158,11,0.08);
  border: 1px solid rgba(245,158,11,0.3);
  border-radius: 12px; padding: 16px 20px; margin: 12px 0;
}
.danger-box {
  background: rgba(239,68,68,0.08);
  border: 1px solid rgba(239,68,68,0.3);
  border-radius: 12px; padding: 16px 20px; margin: 12px 0;
}
.success-box {
  background: rgba(16,185,129,0.08);
  border: 1px solid rgba(16,185,129,0.3);
  border-radius: 12px; padding: 16px 20px; margin: 12px 0;
}

/* Plotly chart backgrounds */
.js-plotly-plot { border-radius: 12px; overflow: hidden; }

/* Tables */
.dataframe { background: var(--bg-card) !important; color: var(--text) !important; }

/* Sidebar nav items */
.sidebar-nav-item {
  padding: 10px 16px; border-radius: 10px; margin: 3px 0;
  cursor: pointer; transition: background 0.2s;
  display: flex; align-items: center; gap: 10px;
}
.sidebar-nav-item:hover { background: var(--bg-hover); }

/* Loading spinner text */
.stSpinner > div { border-top-color: var(--primary) !important; }

/* Metric overrides */
[data-testid="stMetric"] {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 16px !important;
}
[data-testid="stMetricValue"] { color: var(--text) !important; font-weight: 700; }
[data-testid="stMetricDelta"] { font-size: 0.85rem; }

/* Tab styling */
.stTabs [data-baseweb="tab-list"] { background: var(--bg-card); border-radius: 10px; padding: 4px; }
.stTabs [data-baseweb="tab"] { color: var(--muted); border-radius: 8px; }
.stTabs [aria-selected="true"] { background: var(--primary) !important; color: white !important; }

/* Button styling */
.stButton > button {
  background: linear-gradient(135deg, var(--primary), #818cf8);
  color: white; border: none; border-radius: 10px;
  font-weight: 600; transition: all 0.25s;
}
.stButton > button:hover {
  transform: translateY(-2px);
  box-shadow: 0 6px 20px rgba(99,102,241,0.4);
}
/* Splash Screen (Netflix-style) */
.splash-container {
  position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;
  background-color: #0f172a; z-index: 9999999;
  display: flex; justify-content: center; align-items: center; flex-direction: column;
  pointer-events: none;
  animation: splash-fadeout 0.5s ease-in-out 1.5s forwards;
}
.splash-logo {
  font-size: 5rem; font-weight: 900; color: #fff; letter-spacing: 0.12em;
  opacity: 0; transform: scale(0.7);
  animation: splash-zoom 2s cubic-bezier(0.16, 1, 0.3, 1) forwards;
}
.splash-subtitle {
  font-size: 0.9rem; color: #6366f1; font-weight: 600; letter-spacing: 0.25em;
  opacity: 0; margin-top: 12px;
  animation: splash-sub 2s ease forwards 0.3s;
}
@keyframes splash-zoom {
  0%   { transform: scale(0.5); opacity: 0; text-shadow: 0 0 0 #6366f1; }
  30%  { opacity: 1; text-shadow: 0 0 30px #6366f1, 0 0 60px #06b6d4; }
  70%  { transform: scale(1.05); opacity: 1; text-shadow: 0 0 60px #6366f1, 0 0 100px #06b6d4; }
  100% { transform: scale(1.3); opacity: 0; text-shadow: 0 0 120px #6366f1; }
}
@keyframes splash-sub {
  0%   { opacity: 0; transform: translateY(10px); }
  40%  { opacity: 1; transform: translateY(0); }
  100% { opacity: 0; }
}
@keyframes splash-fadeout {
  to { opacity: 0; visibility: hidden; }
}

/* Top navigation bar */
.top-nav-bar {
  background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
  border: 1px solid #334155;
  border-radius: 12px;
  padding: 8px 16px;
  margin-bottom: 12px;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Cinematic Splash Screen
# ─────────────────────────────────────────────
if "splash_shown" not in st.session_state:
    st.session_state.splash_shown = True
    st.markdown("""
    <div class="splash-container">
      <div class="splash-logo">FORESIGHT</div>
      <div class="splash-subtitle">DEMAND & INVENTORY INTELLIGENCE</div>
    </div>
    """, unsafe_allow_html=True)




# ─────────────────────────────────────────────
# Data Loading — uses pre-aggregated tiny files (total <1MB, no raw data needed)
# ─────────────────────────────────────────────
DATA_AGG = ROOT / "data" / "aggregated"

@st.cache_data(ttl=3600, show_spinner=False)
def _read(fpath, **kwargs):
    if not Path(fpath).exists():
        return pd.DataFrame()
    header = pd.read_csv(fpath, nrows=0)
    parse_cols = [c for c in ["date","week","last_sale","first_sale"] if c in header.columns]
    df = pd.read_csv(fpath, parse_dates=parse_cols if parse_cols else False, **kwargs)
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    return df

@st.cache_data(ttl=600, show_spinner=False)
def _load_json(fname):
    fpath = DATA_PROCESSED / fname
    if not fpath.exists():
        return {}
    with open(fpath) as f:
        return json.load(f)

# ── Tiny pre-aggregated loaders (all files <300KB) ─────────────────────────
def load_sales():
    """Daily aggregate totals — 374 rows, 14KB."""
    return _read(DATA_AGG / "sales_daily_agg.csv")

def load_sku_sales():
    """Per-SKU totals — 3804 rows, 252KB."""
    return _read(DATA_AGG / "sku_sales_agg.csv")

def load_sales_weekly():
    """Weekly trend — 53 rows, 1KB."""
    return _read(DATA_AGG / "sales_weekly_agg.csv")

def load_category_agg():
    """Category aggregates — 10 rows, tiny."""
    return _read(DATA_AGG / "category_agg.csv")

def load_inventory():
    """Latest inventory snapshot per SKU — 3804 rows, 184KB."""
    return _read(DATA_AGG / "inventory_latest.csv")

def load_low_stock():
    """SKUs currently below reorder point — 224 rows, 9KB."""
    return _read(DATA_AGG / "low_stock_alerts.csv")

# ── Other small files loaded directly ──────────────────────────────────────
def load_sku_master():  return _read(DATA_PROCESSED / "sku_master.csv")
def load_risk():        return _read(DATA_PROCESSED / "risk_scores.csv")
def load_comparison():  return _read(DATA_PROCESSED / "model_comparison.csv")
def load_forecast():    return _read(DATA_PROCESSED / "forecast_output.csv")
def load_calendar():    return _read(DATA_PROCESSED / "calendar.csv")
def load_impact():      return _load_json("business_impact.json")
def load_features_sample(): return load_sku_sales()  # never load 736MB file

def data_ready() -> bool:
    """Check if pre-aggregated data exists (works on Render without raw files)."""
    return (DATA_AGG / "sales_daily_agg.csv").exists() or \
           (DATA_PROCESSED / "sales_daily.csv").exists()




# ─────────────────────────────────────────────
# Chart theme helper
# ─────────────────────────────────────────────
CHART_LAYOUT = dict(
    paper_bgcolor="#1e293b",
    plot_bgcolor="#1e293b",
    font=dict(family="Inter, sans-serif", color="#94a3b8", size=12),
    margin=dict(l=50, r=30, t=60, b=40),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#f1f5f9")),
    xaxis=dict(gridcolor="#334155", zeroline=False),
    yaxis=dict(gridcolor="#334155", zeroline=False),
)

def apply_theme(fig, title="", height=420):
    fig.update_layout(**CHART_LAYOUT, title=dict(text=title, font=dict(size=17, color="#f1f5f9"), x=0.02), height=height)
    return fig

RISK_COLORS = {
    "Critical Stockout Risk": "#ef4444",
    "High Stockout Risk":     "#f97316",
    "Moderate Stockout Risk": "#f59e0b",
    "Overstock Risk":         "#8b5cf6",
    "Mild Overstock Risk":    "#a78bfa",
    "Volatile Demand":        "#06b6d4",
    "Healthy":                "#10b981",
}


# ─────────────────────────────────────────────
# KPI Card helper
# ─────────────────────────────────────────────
def kpi_card(label, value, delta=None, delta_type="neutral", prefix="", suffix="", tooltip=""):
    delta_html = ""
    if delta is not None:
        arrow = "▲" if delta_type == "positive" else ("▼" if delta_type == "negative" else "●")
        delta_html = f'<div class="kpi-delta {delta_type}">{arrow} {delta}</div>'
    
    tooltip_html = f' title="{tooltip}"' if tooltip else ""
    
    st.markdown(f"""
    <div class="kpi-card"{tooltip_html}>
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{prefix}{value}{suffix}</div>
      {delta_html}
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Sidebar Navigation
# ─────────────────────────────────────────────
PAGES = [
    ("🏠", "Home"),
    ("🔄", "Upload & Retrain"),
    ("📂", "Dataset Overview"),
    ("📈", "Exploratory Data Analysis"),
    ("📊", "Demand Forecast"),
    ("📦", "Inventory Analytics"),
    ("⚠️", "Risk Dashboard"),
    ("📉", "Model Comparison"),
    ("🎯", "Decision Matrix"),
    ("💡", "Recommendations"),
    ("💰", "Business Impact"),
    ("🌐", "API Testing"),
    ("📄", "Project Summary"),
]

# ─────────────────────────────────────────────
# Top Navigation Bar (Elegant Glassmorphism)
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* Hide the default block container padding at top to make header flush */
.block-container {
    padding-top: 1rem;
}
/* Hide the sidebar toggle button completely */
[data-testid="collapsedControl"] {
    display: none;
}
/* Style the pills to look like elegant tabs */
div[data-testid="stPills"] {
    display: flex;
    justify-content: center;
    flex-wrap: wrap;
    gap: 8px;
    padding: 10px 0;
    border-bottom: 1px solid rgba(51, 65, 85, 0.5);
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

page_names = [name for _, name in PAGES]
top_selected = st.pills(
    "Navigate", 
    page_names, 
    selection_mode="single", 
    default=page_names[st.session_state.get("page_index", 0)], 
    key="top_nav",
    label_visibility="collapsed"
)

# Fallback if unselected
if not top_selected:
    top_selected = page_names[st.session_state.get("page_index", 0)]

page = top_selected
st.session_state.page_index = page_names.index(top_selected)

# ─────────────────────────────────────────────
# Global Filters (Horizontal Layout)
# ─────────────────────────────────────────────
selected_cat = "All"
selected_risk = "All"
selected_sku = "All"
date_range = None

if data_ready():
    with st.expander("⚙️ Global Dashboard Filters", expanded=False):
        _sku = load_sku_master()
        _risk = load_risk()
        _sales = load_sales()
        
        c1, c2, c3, c4 = st.columns(4)
        if not _sku.empty:
            cats = ["All"] + sorted(_sku["category"].dropna().unique().tolist())
            selected_cat = c1.selectbox("📁 Category", cats, key="cat_filter")
            
            sku_list = _sku["stock_code"].unique().tolist()
            selected_sku = c3.selectbox("🔑 SKU", ["All"] + sorted(sku_list[:200]), key="sku_filter")
            
        if not _risk.empty:
            risk_levels = ["All"] + sorted(_risk["risk_level"].dropna().unique().tolist())
            selected_risk = c2.selectbox("⚠️ Risk Level", risk_levels, key="risk_filter")
            
        if not _sales.empty and "date" in _sales.columns:
            min_d = _sales["date"].min()
            max_d = _sales["date"].max()
            if pd.notna(min_d) and pd.notna(max_d):
                date_range = c4.date_input("📅 Date Range", value=(min_d.date(), max_d.date()), key="date_filter")

st.markdown('<hr style="border-color:#334155; margin:4px 0 16px;">', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: HOME
# ─────────────────────────────────────────────────────────────────────────────
if page == "Home":
    st.markdown("""
    <div style="text-align:center; padding: 40px 0 20px;">
      <div style="font-size:4rem; margin-bottom:10px;">🔮</div>
      <div style="font-size:2.8rem; font-weight:900; 
                  background: linear-gradient(135deg, #6366f1, #06b6d4, #f59e0b);
                  -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                  letter-spacing:-0.02em; line-height:1.1;">
        Project FORESIGHT
      </div>
      <div style="font-size:1.1rem; color:#94a3b8; margin-top:12px; font-weight:400;">
        Demand & Inventory Intelligence Platform &nbsp;·&nbsp; NorthBay Living
      </div>
    </div>
    """, unsafe_allow_html=True)

    if not data_ready():
        st.markdown("""
        <div class="warning-box">
          <strong>⚡ Pipeline Not Executed Yet</strong><br>
          Run <code>python main.py</code> to process data and train models,
          then refresh this dashboard.
        </div>
        """, unsafe_allow_html=True)
        st.code("python main.py", language="bash")
        st.stop()

    # Data loaded lazily per-file
    sales = load_sales()
    risk  = load_risk()
    impact = load_impact()
    sku_master = load_sku_master()

    # KPI Row
    st.markdown('<div class="section-header">📊 Platform KPIs</div>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        n_skus = sku_master["stock_code"].nunique() if not sku_master.empty else 0
        kpi_card("Total SKUs", f"{n_skus:,}", "Active Products", "neutral")
    with c2:
        rev = sales["revenue"].sum() if "revenue" in sales.columns else 0
        kpi_card("Total Revenue", f"£{rev/1e6:.2f}M", "Historical", "positive", "")
    with c3:
        rar = impact.get("revenue_at_risk", 0)
        kpi_card("Revenue at Risk", f"£{rar:,.0f}", "Next 14 days", "negative")
    with c4:
        cap = impact.get("capital_locked", 0)
        kpi_card("Capital Locked", f"£{cap:,.0f}", "Overstock", "negative")
    with c5:
        crit = impact.get("critical_stockout_count", 0)
        kpi_card("Critical Alerts", f"{crit}", "Immediate Action", "negative")

    st.markdown("<br>", unsafe_allow_html=True)

    # Revenue trend
    col_left, col_right = st.columns([2, 1])
    with col_left:
        if not sales.empty and "date" in sales.columns:
            d = sales.sort_values("date").copy()
            d["ma28"] = d["total_revenue"].rolling(28, min_periods=1).mean()
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=d["date"], y=d["total_revenue"], name="Daily Revenue",
                fill="tozeroy", fillcolor="rgba(99,102,241,0.1)",
                line=dict(color="#6366f1", width=1.5), opacity=0.7))
            fig.add_trace(go.Scatter(x=d["date"], y=d["ma28"], name="28-Day MA",
                line=dict(color="#f59e0b", width=2.5, dash="dash")))
            apply_theme(fig, "📈 Revenue Trend", 360)
            st.plotly_chart(fig, width="stretch")

    with col_right:
        if not risk.empty:
            dist = risk["risk_level"].value_counts().reset_index()
            dist.columns = ["risk_level", "count"]
            fig2 = go.Figure(go.Pie(
                labels=dist["risk_level"], values=dist["count"], hole=0.62,
                marker=dict(colors=[RISK_COLORS.get(r, "#94a3b8") for r in dist["risk_level"]]),
                textinfo="percent", textfont=dict(color="#f1f5f9", size=12),
            ))
            fig2.update_traces(hovertemplate="<b>%{label}</b><br>%{value} SKUs (%{percent})<extra></extra>")
            apply_theme(fig2, "⚠️ Risk Distribution", 360)
            st.plotly_chart(fig2, width="stretch")

    # Feature highlights
    st.markdown('<div class="section-header">🚀 Platform Capabilities</div>', unsafe_allow_html=True)
    fc1, fc2, fc3, fc4 = st.columns(4)
    features_list = [
        ("🤖", "6 ML Models", "RF, GBM, LGBM, XGB, Prophet, Naive — auto-selected by WAPE"),
        ("📦", "Inventory Engine", "EOQ, ROP, (s,Q) replenishment policy with lead-time awareness"),
        ("⚠️", "Risk Scoring", "7-level stockout/overstock classification with business rules"),
        ("💰", "Business Impact", "Revenue at risk, capital locked, profit improvement estimation"),
    ]
    for col, (icon, title, desc) in zip([fc1, fc2, fc3, fc4], features_list):
        with col:
            st.markdown(f"""
            <div class="kpi-card" style="text-align:left; min-height:140px;">
              <div style="font-size:1.8rem; margin-bottom:8px;">{icon}</div>
              <div style="font-size:0.95rem; font-weight:700; color:#f1f5f9; margin-bottom:4px;">{title}</div>
              <div style="font-size:0.78rem; color:#94a3b8; line-height:1.5;">{desc}</div>
            </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: UPLOAD & RETRAIN
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Upload & Retrain":
    st.markdown('<div class="section-header">🔄 Upload New Dataset & Retrain Models</div>', unsafe_allow_html=True)

    # ── Intro cards ──
    intro1, intro2, intro3 = st.columns(3)
    with intro1:
        st.markdown("""
        <div class="kpi-card" style="text-align:left; min-height:130px;">
          <div style="font-size:1.6rem; margin-bottom:6px;">📤</div>
          <div style="font-size:0.92rem; font-weight:700; color:#f1f5f9;">Step 1: Upload</div>
          <div style="font-size:0.76rem; color:#94a3b8; margin-top:4px;">Drag & drop or browse for your new transaction CSV file.</div>
        </div>""", unsafe_allow_html=True)
    with intro2:
        st.markdown("""
        <div class="kpi-card" style="text-align:left; min-height:130px;">
          <div style="font-size:1.6rem; margin-bottom:6px;">🔍</div>
          <div style="font-size:0.92rem; font-weight:700; color:#f1f5f9;">Step 2: Preview & Validate</div>
          <div style="font-size:0.76rem; color:#94a3b8; margin-top:4px;">Inspect the data, check column mappings, and review quality metrics.</div>
        </div>""", unsafe_allow_html=True)
    with intro3:
        st.markdown("""
        <div class="kpi-card" style="text-align:left; min-height:130px;">
          <div style="font-size:1.6rem; margin-bottom:6px;">🚀</div>
          <div style="font-size:0.92rem; font-weight:700; color:#f1f5f9;">Step 3: Retrain</div>
          <div style="font-size:0.76rem; color:#94a3b8; margin-top:4px;">One click to run the full ML pipeline and refresh all dashboards.</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Current dataset info ──
    current_raw = ROOT / "data" / "raw" / "data.csv"
    if current_raw.exists():
        import os
        file_size = os.path.getsize(current_raw) / (1024 * 1024)
        mod_time = datetime.fromtimestamp(os.path.getmtime(current_raw)).strftime("%Y-%m-%d %H:%M")
        st.markdown(f"""
        <div class="info-box">
          📁 <strong>Current Dataset:</strong> data.csv &nbsp;|&nbsp;
          💾 <strong>Size:</strong> {file_size:.1f} MB &nbsp;|&nbsp;
          🕐 <strong>Last Modified:</strong> {mod_time}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="warning-box">
          ⚠️ <strong>No dataset found.</strong> Upload a CSV file below to get started.
        </div>
        """, unsafe_allow_html=True)

    # ── File Upload ──
    st.markdown("### 📤 Upload Your Dataset")
    st.markdown("""Upload a CSV file with e-commerce transaction data.
    The file should contain columns like: `InvoiceNo`/`Invoice`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `UnitPrice`/`Price`, `CustomerID`/`Customer ID`, `Country`.""")

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        help="Supported format: .csv — Maximum recommended size: 50 MB",
        key="retrain_upload"
    )

    if uploaded_file is not None:
        # ── Preview Section ──
        st.markdown("---")
        st.markdown("### 🔍 Data Preview & Validation")

        try:
            preview_df = pd.read_csv(uploaded_file, nrows=200)
            uploaded_file.seek(0)  # reset for later use

            prev_col1, prev_col2 = st.columns([2, 1])
            with prev_col1:
                st.markdown("**First 10 rows of your data:**")
                st.dataframe(preview_df.head(10), width="stretch", height=350)

            with prev_col2:
                st.markdown("**📊 Quick Stats**")
                st.metric("Columns", len(preview_df.columns))
                st.metric("Preview Rows", len(preview_df))
                st.metric("File Size", f"{uploaded_file.size / (1024*1024):.2f} MB")

            # ── Column Validation ──
            st.markdown("**🧪 Column Validation**")
            expected_cols = {
                "InvoiceNo": ["InvoiceNo", "Invoice", "invoice", "invoice_no"],
                "StockCode": ["StockCode", "stock_code", "stockcode", "SKU"],
                "Description": ["Description", "description", "desc", "product_name"],
                "Quantity": ["Quantity", "quantity", "qty", "Qty"],
                "InvoiceDate": ["InvoiceDate", "invoice_date", "date", "Date"],
                "UnitPrice": ["UnitPrice", "Price", "price", "unit_price"],
                "CustomerID": ["CustomerID", "Customer ID", "customer_id", "cust_id"],
                "Country": ["Country", "country"],
            }

            actual_cols = set(preview_df.columns.tolist())
            validation_rows = []
            all_found = True
            for req_name, aliases in expected_cols.items():
                found = any(a in actual_cols for a in aliases)
                matched = next((a for a in aliases if a in actual_cols), "—")
                validation_rows.append({
                    "Required Column": req_name,
                    "Status": "✅ Found" if found else "❌ Missing",
                    "Matched As": matched if found else "—",
                })
                if not found:
                    all_found = False

            val_df = pd.DataFrame(validation_rows)
            st.dataframe(val_df, width="stretch", hide_index=True)

            if all_found:
                st.success("✅ All required columns are present! Your data is ready for training.")
            else:
                st.warning("⚠️ Some columns are missing. The pipeline may still work if the data has compatible column names.")

            # ── Data Quality Metrics ──
            st.markdown("**📈 Data Quality Overview**")
            q1, q2, q3, q4 = st.columns(4)
            with q1:
                null_pct = preview_df.isnull().mean().mean() * 100
                st.metric("Avg Null %", f"{null_pct:.1f}%", help="Average percentage of missing values across all columns")
            with q2:
                dup_pct = preview_df.duplicated().mean() * 100
                st.metric("Duplicate Rows %", f"{dup_pct:.1f}%", help="Percentage of fully duplicated rows")
            with q3:
                num_cols = len(preview_df.select_dtypes(include=[np.number]).columns)
                st.metric("Numeric Columns", num_cols)
            with q4:
                cat_cols = len(preview_df.select_dtypes(include=["object"]).columns)
                st.metric("Text Columns", cat_cols)

            # ── Retrain Button ──
            st.markdown("---")
            st.markdown("### 🚀 Train Models")
            st.markdown("Click below to replace the current dataset and run the full ML pipeline. This will:")
            st.markdown("""
            1. **Save** the uploaded file as the new source dataset
            2. **Build** the 4 FORESIGHT tables (sales, SKU master, calendar, inventory)
            3. **Engineer** 90+ ML features from the raw data
            4. **Train & evaluate** 5 forecasting models (Random Forest, XGBoost, LightGBM, etc.)
            5. **Generate** 28-day demand forecasts for every SKU
            6. **Score** inventory risk levels and business impact
            """)

            col_btn, col_warn = st.columns([1, 2])
            with col_btn:
                retrain_clicked = st.button(
                    "🚀 Upload & Retrain Models",
                    type="primary",
                    use_container_width=True,
                    help="This will take approximately 5-10 minutes"
                )
            with col_warn:
                st.markdown("""
                <div class="warning-box" style="margin:0; padding:12px 16px;">
                  ⏱ <strong>Estimated time:</strong> 5–10 minutes depending on dataset size.<br>
                  💡 Do not close or refresh the browser while training is in progress.
                </div>
                """, unsafe_allow_html=True)

            if retrain_clicked:
                with st.status("🔄 Running Full ML Pipeline...", expanded=True) as status:
                    import subprocess

                    st.write("📥 **Step 1/6** — Saving uploaded dataset...")
                    raw_dir = ROOT / "data" / "raw"
                    raw_dir.mkdir(parents=True, exist_ok=True)
                    file_path = raw_dir / "data.csv"
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    st.write(f"   ✅ Saved {uploaded_file.size / (1024*1024):.1f} MB to data/raw/data.csv")

                    st.write("🏗️ **Step 2/6** — Building FORESIGHT tables (sales, SKU master, calendar, inventory)...")
                    st.write("⚙️ **Step 3/6** — Engineering 90+ ML features...")
                    st.write("🤖 **Step 4/6** — Training & cross-validating 5 ML models...")
                    st.write("📊 **Step 5/6** — Generating 28-day demand forecasts...")
                    st.write("⚠️ **Step 6/6** — Scoring risk levels & calculating business impact...")

                    result = subprocess.run(
                        ["python", "main.py"],
                        cwd=str(ROOT),
                        capture_output=True,
                        text=True
                    )

                    if result.returncode == 0:
                        status.update(label="✅ Pipeline completed successfully!", state="complete", expanded=False)
                        st.toast("🎉 Models retrained! All dashboards updated.", icon="✅")
                        st.balloons()
                        st.cache_data.clear()
                        time.sleep(2)
                        st.rerun()
                    else:
                        status.update(label="❌ Pipeline encountered an error", state="error", expanded=True)
                        st.error("The pipeline failed. See the error log below:")
                        st.code(result.stderr or result.stdout, language="text")

        except Exception as e:
            st.error(f"❌ Could not read the file: {e}")
            st.info("Please ensure the file is a valid CSV with UTF-8 encoding.")

    else:
        # Empty state — show helpful guidance
        st.markdown("---")
        st.markdown("""
        <div style="text-align:center; padding:60px 20px; opacity:0.7;">
          <div style="font-size:4rem; margin-bottom:16px;">📁</div>
          <div style="font-size:1.2rem; font-weight:600; color:#94a3b8; margin-bottom:8px;">No file selected</div>
          <div style="font-size:0.85rem; color:#64748b; max-width:400px; margin:0 auto;">
            Drag and drop a CSV file above, or click <strong>Browse files</strong> to select one from your computer.
          </div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: DATASET OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Dataset Overview":
    st.markdown('<div class="section-header">📂 Dataset Overview</div>', unsafe_allow_html=True)

    if not data_ready():
        st.warning("Run `python main.py` first to generate processed data.")
        st.stop()

    # Data loaded lazily per-file

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Sales Daily", "🏷 SKU Master", "📅 Calendar", "📦 Inventory"])

    with tab1:
        df = load_sales()          # daily aggregates
        df_sku = load_sku_sales()  # per-SKU detail
        if not df.empty:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Daily Rows", f"{len(df):,}")
            c2.metric("SKUs", f"{len(df_sku):,}")
            c3.metric("Date Range", f"{df['date'].min().date() if pd.api.types.is_datetime64_any_dtype(df['date']) else 'N/A'}")
            c4.metric("Null %", f"{df.isnull().mean().mean()*100:.1f}%")
            st.markdown("**Daily Sales Aggregates (first 50 rows)**")
            st.dataframe(df.head(50), width="stretch")
            csv = df.to_csv(index=False).encode()
            st.download_button("⬇ Download sales_daily_agg.csv", csv, "sales_daily_agg.csv", "text/csv")

    with tab2:
        df = load_sku_master()
        if not df.empty:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total SKUs", f"{len(df):,}")
            c2.metric("Categories", f"{df['category'].nunique()}")
            c3.metric("Avg Price", f"£{df['avg_unit_price'].mean():.2f}")
            c4.metric("ABC-A SKUs", f"{(df['abc_class']=='A').sum():,}")
            col_l, col_r = st.columns(2)
            with col_l:
                cat_counts = df["category"].value_counts().reset_index()
                fig = px.bar(cat_counts, x="count", y="category", orientation="h",
                             color="count", color_continuous_scale=["#6366f1","#06b6d4"],
                             title="SKUs per Category")
                apply_theme(fig, height=380)
                st.plotly_chart(fig, width="stretch")
            with col_r:
                abc_counts = df["abc_class"].value_counts().reset_index()
                fig2 = go.Figure(go.Pie(labels=abc_counts["abc_class"], values=abc_counts["count"],
                    hole=0.55, marker=dict(colors=["#6366f1","#06b6d4","#f59e0b"]),
                    textinfo="percent+label", textfont=dict(color="#f1f5f9")))
                apply_theme(fig2, "ABC Classification", 380)
                st.plotly_chart(fig2, width="stretch")
            st.dataframe(df.head(50), width="stretch")
            csv = df.to_csv(index=False).encode()
            st.download_button("⬇ Download sku_master.csv", csv, "sku_master.csv", "text/csv")

    with tab3:
        df = load_calendar()
        if not df.empty:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Date Rows", f"{len(df):,}")
            c2.metric("UK Holidays", f"{df['is_holiday'].sum()}")
            c3.metric("Black Fridays", f"{df['is_black_friday'].sum()}")
            c4.metric("Weekend Days", f"{df['is_weekend'].sum()}")
            st.dataframe(df.head(50), width="stretch")
            csv = df.to_csv(index=False).encode()
            st.download_button("⬇ Download calendar.csv", csv, "calendar.csv", "text/csv")

    with tab4:
        df = load_inventory()
        if not df.empty:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("SKUs Tracked", f"{len(df):,}")
            c2.metric("Avg Coverage (days)", f"{df['coverage_days'].mean():.1f}")
            c3.metric("Total Inv Value", f"£{df['inventory_value'].sum():,.0f}")
            c4.metric("Below Reorder Point", f"{(df['closing_inventory'] <= df['reorder_point']).sum():,}")
            fig = go.Figure(go.Histogram(x=df["coverage_days"].clip(upper=365), nbinsx=40,
                marker_color="#6366f1", opacity=0.8))
            fig.add_vline(x=7, line_dash="dash", line_color="#ef4444", annotation_text="Critical")
            fig.add_vline(x=90, line_dash="dash", line_color="#f59e0b", annotation_text="Overstock")
            apply_theme(fig, "Inventory Coverage Distribution (days)", 360)
            st.plotly_chart(fig, width="stretch")
            st.dataframe(df.head(50), width="stretch")
            csv = df.to_csv(index=False).encode()
            st.download_button("⬇ Download inventory_snapshots.csv", csv, "inventory_snapshots.csv", "text/csv")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: EDA
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Exploratory Data Analysis":
    st.markdown('<div class="section-header">📈 Exploratory Data Analysis</div>', unsafe_allow_html=True)

    if not data_ready():
        st.warning("Run `python main.py` first.")
        st.stop()

    # Use pre-aggregated data (tiny files, no raw CSVs needed)
    sales = load_sales()          # daily agg: date, total_quantity, total_revenue, sku_count, promo_days
    sku_sales = load_sku_sales()  # per-SKU: stock_code, total_quantity, total_revenue, avg_price...
    sku_master = load_sku_master()
    cat_agg = load_category_agg()

    if sales.empty:
        st.info("Aggregated sales data not found. Run the pipeline first.")
        st.stop()

    eda_tabs = st.tabs(["📈 Revenue & Sales", "🏆 Top Products", "🌊 Seasonality", "🔗 Correlation", "🎯 Promotions"])

    with eda_tabs[0]:
        col1, col2 = st.columns(2)
        with col1:
            d = sales.sort_values("date").copy()
            d["ma7"]  = d["total_revenue"].rolling(7, min_periods=1).mean()
            d["ma28"] = d["total_revenue"].rolling(28, min_periods=1).mean()
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=d["date"], y=d["total_revenue"], name="Daily", fill="tozeroy",
                fillcolor="rgba(99,102,241,0.08)", line=dict(color="#6366f1", width=1.2), opacity=0.6))
            fig.add_trace(go.Scatter(x=d["date"], y=d["ma7"], name="7d MA", line=dict(color="#06b6d4", width=2)))
            fig.add_trace(go.Scatter(x=d["date"], y=d["ma28"], name="28d MA", line=dict(color="#f59e0b", width=2, dash="dash")))
            apply_theme(fig, "Revenue Trend", 380)
            st.plotly_chart(fig, width="stretch")
        with col2:
            d2 = sales.sort_values("date").copy()
            d2["ma7"] = d2["total_quantity"].rolling(7, min_periods=1).mean()
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(x=d2["date"], y=d2["total_quantity"], name="Units", marker_color="#6366f1", opacity=0.65))
            fig2.add_trace(go.Scatter(x=d2["date"], y=d2["ma7"], name="7d MA", line=dict(color="#f59e0b", width=2)))
            apply_theme(fig2, "Units Sold Trend", 380)
            st.plotly_chart(fig2, width="stretch")

        # Monthly breakdown from daily agg
        sf = sales.copy()
        sf["date"] = pd.to_datetime(sf["date"])
        sf["month"] = sf["date"].dt.month
        monthly = sf.groupby("month").agg(revenue=("total_revenue","sum"), quantity=("total_quantity","sum")).reset_index()
        month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        monthly["month_name"] = monthly["month"].apply(lambda x: month_names[x-1])
        fig3 = px.bar(monthly, x="month_name", y="revenue", color="revenue",
            color_continuous_scale=["#1e293b","#6366f1","#06b6d4"], title="Monthly Revenue Distribution")
        apply_theme(fig3, "Monthly Revenue", 360)
        st.plotly_chart(fig3, width="stretch")

    with eda_tabs[1]:
        if not sku_sales.empty:
            sku_with_desc = sku_sales.merge(
                sku_master[["stock_code","description","category","abc_class"]].drop_duplicates(),
                on="stock_code", how="left")
            top20 = sku_with_desc.nlargest(20, "total_revenue").sort_values("total_revenue")
            fig = go.Figure(go.Bar(
                x=top20["total_revenue"],
                y=top20["description"].str[:40].fillna(top20["stock_code"]),
                orientation="h",
                marker=dict(color=top20["total_revenue"], colorscale=[[0,"#6366f1"],[1,"#06b6d4"]]),
                text=top20["total_revenue"].apply(lambda v: f"£{v:,.0f}"),
                textposition="outside", textfont=dict(color="#f1f5f9"),
            ))
            apply_theme(fig, "Top 20 Products by Revenue", 600)
            st.plotly_chart(fig, width="stretch")

        # Category treemap from pre-aggregated category file
        if not cat_agg.empty:
            fig2 = px.treemap(cat_agg, path=["category"], values="total_revenue",
                color="total_revenue", color_continuous_scale=["#1e293b","#6366f1","#06b6d4"])
            apply_theme(fig2, "Revenue by Category (Treemap)", 450)
            st.plotly_chart(fig2, width="stretch")

    with eda_tabs[2]:
        sf2 = sales.copy()
        sf2["date"] = pd.to_datetime(sf2["date"])
        sf2["month"] = sf2["date"].dt.month_name().str[:3]
        sf2["dow"]   = sf2["date"].dt.day_name().str[:3]
        month_order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        dow_order   = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
        pivot = sf2.pivot_table(index="month", columns="dow", values="total_quantity", aggfunc="mean").fillna(0)
        pivot = pivot.reindex([m for m in month_order if m in pivot.index])
        pivot = pivot.reindex(columns=[d for d in dow_order if d in pivot.columns])
        fig = go.Figure(go.Heatmap(
            z=pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(),
            colorscale=[[0,"#0f172a"],[0.5,"#6366f1"],[1,"#06b6d4"]],
            text=pivot.values.round(0), texttemplate="%{text}", showscale=True,
        ))
        apply_theme(fig, "Seasonality Heatmap – Total Units (Month)", 420)
        st.plotly_chart(fig, width="stretch")

        dow_avg = sf2.groupby("dow")["total_quantity"].mean().reindex(dow_order).reset_index()
        dow_avg.columns = ["day", "avg_qty"]
        fig2 = px.bar(dow_avg, x="day", y="avg_qty", color="avg_qty",
            color_continuous_scale=["#6366f1","#06b6d4"], title="Avg Daily Sales by Day of Week")
        apply_theme(fig2, height=360)
        st.plotly_chart(fig2, width="stretch")

    with eda_tabs[3]:
        features = load_features_sample()  # returns sku_sales_agg
        if not features.empty:
            num_cols = features.select_dtypes(include=[np.number]).columns.tolist()
            target = "total_revenue" if "total_revenue" in num_cols else (num_cols[0] if num_cols else None)
            if target:
                corr = features[num_cols].corr()[target].abs().nlargest(12).drop(target, errors="ignore")
                top_feats = corr.index.tolist()
                corr_matrix = features[[target] + top_feats].corr()
                fig = go.Figure(go.Heatmap(
                    z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.index,
                    colorscale="RdBu", zmid=0,
                    text=corr_matrix.values.round(2), texttemplate="%{text}",
                    textfont=dict(size=9),
                ))
                apply_theme(fig, "SKU-Level Feature Correlation Heatmap", 500)
                st.plotly_chart(fig, width="stretch")
        else:
            st.info("Aggregated SKU data not available.")

    with eda_tabs[4]:
        # Promo days from daily agg
        if "promo_days" in sales.columns:
            promo_d = sales.copy()
            promo_d["label"] = promo_d["promo_days"].apply(lambda x: "Promotion" if x > 0 else "Regular")
            promo_summary = promo_d.groupby("label").agg(
                avg_qty=("total_quantity","mean"),
                avg_rev=("total_revenue","mean")
            ).reset_index()
            fig = go.Figure()
            for col, label, color in [("avg_qty","Avg Units","#6366f1"),("avg_rev","Avg Revenue","#06b6d4")]:
                fig.add_trace(go.Bar(x=promo_summary["label"], y=promo_summary[col], name=label,
                    marker_color=color, text=promo_summary[col].round(1), textposition="outside",
                    textfont=dict(color="#f1f5f9")))
            apply_theme(fig, "Promotion Impact on Sales Volume", 380)
            st.plotly_chart(fig, width="stretch")
            promo_pct = (sales["promo_days"].gt(0).sum() / len(sales) * 100) if len(sales) > 0 else 0
            st.markdown(f'<div class="info-box">📊 <strong>{promo_pct:.1f}%</strong> of days had promotion activity.</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: DEMAND FORECAST
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Demand Forecast":
    st.markdown('<div class="section-header">📊 Demand Forecast</div>', unsafe_allow_html=True)

    # Data loaded lazily per-file
    sales = load_sales()
    forecast = load_forecast()
    sku_master = load_sku_master()

    if forecast.empty:
        st.warning("Forecast data not yet available. Run `python main.py` first.")
        st.stop()

    # SKU selector
    sku_options = sorted(forecast["stock_code"].unique().tolist())
    col_sel1, col_sel2 = st.columns([1, 3])
    with col_sel1:
        sel_sku = st.selectbox("Select SKU", sku_options[:100], key="fcast_sku")

    sku_name = ""
    if not sku_master.empty:
        desc = sku_master[sku_master["stock_code"] == sel_sku]["description"].values
        sku_name = desc[0] if len(desc) > 0 else ""

    # History from sku_sales_agg (has stock_code + total_quantity)
    sku_sales = load_sku_sales()
    hist_row = sku_sales[sku_sales["stock_code"] == sel_sku] if not sku_sales.empty else pd.DataFrame()
    fcast_sku = forecast[forecast["stock_code"] == sel_sku].sort_values("date")

    fig = go.Figure()
    if not hist_row.empty:
        # Show total historical qty as a single reference point bar
        hist_qty = hist_row["total_quantity"].values[0]
        fig.add_trace(go.Bar(x=["Historical Total"], y=[hist_qty], name="Historical Total",
            marker_color="#06b6d4", width=0.4))
    if not fcast_sku.empty:
        fig.add_trace(go.Scatter(x=fcast_sku["date"], y=fcast_sku["forecast_quantity"], name="Forecast",
            line=dict(color="#f59e0b", width=2, dash="dash"), mode="lines+markers",
            marker=dict(size=6, symbol="circle")))
        # Confidence band
        upper = fcast_sku["forecast_quantity"] * 1.25
        lower = fcast_sku["forecast_quantity"] * 0.75
        fig.add_trace(go.Scatter(
            x=pd.concat([fcast_sku["date"], fcast_sku["date"][::-1]]),
            y=pd.concat([upper, lower[::-1]]),
            fill="toself", fillcolor="rgba(245,158,11,0.12)",
            line=dict(color="rgba(0,0,0,0)"), name="±25% Band",
        ))
        if not hist_row.empty:
            fig.add_vline(x=fcast_sku["date"].min(), line_dash="dot",
                line_color="#94a3b8", opacity=0.6, annotation_text="Forecast →")

    apply_theme(fig, f"Demand Forecast: {sku_name or sel_sku}", 460)
    st.plotly_chart(fig, width="stretch")

    # Forecast table
    col_left, col_right = st.columns([1, 1])
    with col_left:
        st.markdown("**📋 Forecast Details**")
        if not fcast_sku.empty:
            display = fcast_sku[["date","forecast_quantity"]].copy()
            display["date"] = display["date"].astype(str)
            display["forecast_quantity"] = display["forecast_quantity"].round(1)
            st.dataframe(display.rename(columns={"forecast_quantity":"Forecast Units"}),
                width="stretch", height=300)
            csv = display.to_csv(index=False).encode()
            st.download_button("⬇ Download Forecast", csv, f"forecast_{sel_sku}.csv", "text/csv")

    with col_right:
        st.markdown("**📊 Forecast Summary**")
        if not fcast_sku.empty:
            total_fcast = fcast_sku["forecast_quantity"].sum()
            avg_fcast   = fcast_sku["forecast_quantity"].mean()
            peak_fcast  = fcast_sku["forecast_quantity"].max()
            peak_date   = fcast_sku.loc[fcast_sku["forecast_quantity"].idxmax(), "date"]
            st.metric("Total Forecast Qty (28d)", f"{total_fcast:,.1f}")
            st.metric("Avg Daily Forecast", f"{avg_fcast:.1f}")
            st.metric("Peak Day Forecast", f"{peak_fcast:.1f} ({str(peak_date)[:10]})")
            if not sku_master.empty:
                price_row = sku_master[sku_master["stock_code"] == sel_sku]["avg_unit_price"].values
                if len(price_row) > 0:
                    st.metric("Forecast Revenue (28d)", f"£{total_fcast * price_row[0]:,.2f}")

    # All SKUs forecast table
    st.markdown("**🗃 All SKUs – 28-Day Forecast Summary**")
    agg = forecast.groupby("stock_code")["forecast_quantity"].agg(["sum","mean","max"]).reset_index()
    agg.columns = ["SKU","Total Units","Avg Daily","Peak Day"]
    if not sku_master.empty:
        agg = agg.merge(sku_master[["stock_code","description","category","abc_class"]].drop_duplicates(),
            left_on="SKU", right_on="stock_code", how="left").drop(columns=["stock_code"])
    agg = agg.sort_values("Total Units", ascending=False)
    st.dataframe(agg.head(100), width="stretch")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: INVENTORY ANALYTICS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Inventory Analytics":
    st.markdown('<div class="section-header">📦 Inventory Analytics</div>', unsafe_allow_html=True)

    # Data loaded lazily per-file
    inv = load_inventory()
    risk = load_risk()
    sku_master = load_sku_master()

    if inv.empty:
        st.warning("Run `python main.py` first.")
        st.stop()

    # KPIs
    latest = inv.sort_values("date").groupby("stock_code").last().reset_index()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Inventory Value", f"£{latest['inventory_value'].sum():,.0f}")
    c2.metric("Avg Coverage Days", f"{latest['coverage_days'].mean():.1f}")
    c3.metric("Reorder Events (total)", f"{inv['reorder_triggered'].sum():,}")
    c4.metric("Zero Stock SKUs", f"{(latest['closing_inventory'] <= 0).sum()}")

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure(go.Histogram(x=latest["coverage_days"].clip(upper=365), nbinsx=40,
            marker_color="#6366f1", opacity=0.8))
        fig.add_vline(x=7, line_dash="dash", line_color="#ef4444", annotation_text="Critical (7d)")
        fig.add_vline(x=14, line_dash="dash", line_color="#f97316", annotation_text="High Risk (14d)")
        fig.add_vline(x=90, line_dash="dash", line_color="#f59e0b", annotation_text="Overstock (90d)")
        apply_theme(fig, "Inventory Coverage Distribution", 380)
        st.plotly_chart(fig, width="stretch")

    with col2:
        if not risk.empty:
            dead = (latest["coverage_days"] > 365).sum()
            slow = ((latest["coverage_days"] > 90) & (latest["coverage_days"] <= 365)).sum()
            active = (latest["coverage_days"] <= 90).sum()
            fig2 = go.Figure(go.Pie(
                labels=["Dead Stock (>365d)", "Slow Moving (90–365d)", "Active (<90d)"],
                values=[dead, slow, active], hole=0.55,
                marker=dict(colors=["#ef4444","#f59e0b","#10b981"]),
                textinfo="percent+label", textfont=dict(color="#f1f5f9"),
            ))
            apply_theme(fig2, "Dead Stock Analysis", 380)
            st.plotly_chart(fig2, width="stretch")

    # Inventory over time for selected SKU
    st.markdown("**📈 Inventory Level Over Time**")
    sku_sel = st.selectbox("Select SKU", sorted(inv["stock_code"].unique()[:200]), key="inv_sku")
    sku_inv = inv[inv["stock_code"] == sku_sel].sort_values("date")
    if not sku_inv.empty:
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=sku_inv["date"], y=sku_inv["closing_inventory"],
            name="Closing Inventory", fill="tozeroy", fillcolor="rgba(99,102,241,0.1)",
            line=dict(color="#6366f1", width=2)))
        fig3.add_trace(go.Scatter(x=sku_inv["date"],
            y=[sku_inv["reorder_point"].mean()] * len(sku_inv),
            name="Reorder Point", line=dict(color="#ef4444", width=1.5, dash="dash")))
        # Reorder events
        reorders = sku_inv[sku_inv["reorder_triggered"] == 1]
        if not reorders.empty:
            fig3.add_trace(go.Scatter(x=reorders["date"], y=reorders["closing_inventory"],
                mode="markers", name="Reorder Triggered",
                marker=dict(color="#f59e0b", size=8, symbol="triangle-up")))
        apply_theme(fig3, f"Inventory Level: {sku_sel}", 380)
        st.plotly_chart(fig3, width="stretch")

    # Top overstocked SKUs
    st.markdown("**🏗 Top 20 Overstocked SKUs (by coverage days)**")
    over = latest.nlargest(20, "coverage_days")[["stock_code","closing_inventory","coverage_days","inventory_value","reorder_point"]]
    if not sku_master.empty:
        over = over.merge(sku_master[["stock_code","description","category"]].drop_duplicates(), on="stock_code", how="left")
    st.dataframe(over.round(2), width="stretch")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: RISK DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Risk Dashboard":
    st.markdown('<div class="section-header">⚠️ Risk Dashboard</div>', unsafe_allow_html=True)

    # Data loaded lazily per-file
    risk = load_risk()

    if risk.empty:
        st.warning("Run `python main.py` first.")
        st.stop()

    # Apply filters
    risk_f = risk.copy()
    if selected_cat != "All" and "category" in risk_f.columns:
        risk_f = risk_f[risk_f["category"] == selected_cat]
    if selected_risk != "All" and "risk_level" in risk_f.columns:
        risk_f = risk_f[risk_f["risk_level"] == selected_risk]
        
    st.markdown("##### 🎛️ Interactive Filters")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        if "coverage_days" in risk_f.columns and not risk_f.empty:
            max_cov = float(risk["coverage_days"].max())
            max_cov = min(max_cov, 1000.0) # cap slider at 1000 for usability
            cov_filter = st.slider("Inventory Coverage (Days)", 0.0, max_cov, (0.0, max_cov), help="Filter items based on how many days of stock they have left.")
            risk_f = risk_f[(risk_f["coverage_days"] >= cov_filter[0]) & (risk_f["coverage_days"] <= cov_filter[1])]
    with col_s2:
        if "avg_unit_price" in risk_f.columns and not risk_f.empty:
            max_price = float(risk["avg_unit_price"].max())
            price_filter = st.slider("Average Unit Price (£)", 0.0, max_price, (0.0, max_price), help="Filter items based on their average selling price.")
            risk_f = risk_f[(risk_f["avg_unit_price"] >= price_filter[0]) & (risk_f["avg_unit_price"] <= price_filter[1])]
    
    st.markdown("<hr style='border-color:#334155; margin:10px 0;'>", unsafe_allow_html=True)

    # KPI row
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        n = (risk_f["risk_level"] == "Critical Stockout Risk").sum()
        kpi_card("Critical Stockout", str(n), "Reorder Immediately", "negative")
    with c2:
        n = (risk_f["risk_level"] == "High Stockout Risk").sum()
        kpi_card("High Stockout", str(n), "Increase Purchase", "negative")
    with c3:
        n = risk_f["risk_level"].str.contains("Overstock", na=False).sum()
        kpi_card("Overstock Risk", str(n), "Reduce / Markdown", "negative")
    with c4:
        n = (risk_f["risk_level"] == "Volatile Demand").sum()
        kpi_card("Volatile Demand", str(n), "Monitor Closely", "neutral")
    with c5:
        n = (risk_f["risk_level"] == "Healthy").sum()
        kpi_card("Healthy", str(n), "No Action", "positive")

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 2])

    with col1:
        dist = risk_f["risk_level"].value_counts().reset_index()
        dist.columns = ["risk_level", "count"]
        fig = go.Figure(go.Pie(
            labels=dist["risk_level"], values=dist["count"], hole=0.60,
            marker=dict(colors=[RISK_COLORS.get(r, "#94a3b8") for r in dist["risk_level"]]),
            textinfo="percent+label", textfont=dict(color="#f1f5f9", size=11),
        ))
        apply_theme(fig, "Risk Distribution", 380)
        st.plotly_chart(fig, width="stretch")

    with col2:
        # Risk scatter
        fig2 = px.scatter(
            risk_f.head(500),
            x="coverage_days", y="revenue_at_risk",
            color="risk_level",
            size="priority_score",
            hover_name="description" if "description" in risk_f.columns else "stock_code",
            hover_data=["stock_code","abc_class","recommendation"],
            color_discrete_map=RISK_COLORS,
            log_y=True,
            title="Risk Landscape: Coverage Days vs Revenue at Risk",
        )
        apply_theme(fig2, height=380)
        st.plotly_chart(fig2, width="stretch")

    # Risk table
    st.markdown("**📋 Risk Scores – All SKUs**")
    display_cols = [c for c in [
        "stock_code","description","category","abc_class","risk_level","risk_score",
        "coverage_days","revenue_at_risk","capital_locked","priority_score","recommendation"
    ] if c in risk_f.columns]

    def color_risk(val):
        colors = {
            "Critical Stockout Risk": "color: #ef4444; font-weight: 700",
            "High Stockout Risk":     "color: #f97316; font-weight: 600",
            "Overstock Risk":         "color: #8b5cf6; font-weight: 600",
            "Healthy":                "color: #10b981",
        }
        return colors.get(val, "")

    st.dataframe(
        risk_f[display_cols].head(200).style.map(color_risk, subset=["risk_level"] if "risk_level" in display_cols else []),
        width="stretch", height=400,
    )
    csv = risk_f[display_cols].to_csv(index=False).encode()
    st.download_button("⬇ Download Risk Report", csv, "risk_scores.csv", "text/csv")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: MODEL COMPARISON
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Model Comparison":
    st.markdown('<div class="section-header">📉 Model Comparison</div>', unsafe_allow_html=True)

    # Data loaded lazily per-file
    comp = load_comparison()

    if comp.empty:
        st.warning("Model comparison data not available. Run `python main.py` first.")
        st.stop()

    best_name = ""
    best_file = MODELS_DIR / "best_model_name.txt"
    if best_file.exists():
        best_name = best_file.read_text().strip()

    st.markdown(f"""
    <div class="success-box">
      🏆 <strong>Best Model Selected: {best_name}</strong> &nbsp;&mdash;&nbsp;
      Chosen automatically based on lowest WAPE across {len(comp)} evaluated models.
    </div>
    """, unsafe_allow_html=True)

    # Metrics chart
    metrics = ["WAPE", "MAPE", "MAE", "RMSE"]
    available_metrics = [m for m in metrics if m in comp.columns]

    if available_metrics:
        fig = go.Figure()
        colors = ["#6366f1","#06b6d4","#f59e0b","#10b981","#f97316","#ef4444"]
        for i, (_, row) in enumerate(comp.iterrows()):
            highlight = row.get("model", "") == best_name
            fig.add_trace(go.Bar(
                name=row.get("model",""),
                x=available_metrics,
                y=[row.get(m, 0) for m in available_metrics],
                marker_color=colors[i % len(colors)],
                opacity=1.0 if highlight else 0.65,
                text=[f"{row.get(m,0):.2f}" for m in available_metrics],
                textposition="outside",
                textfont=dict(color="#f1f5f9", size=10),
            ))
        fig.update_layout(barmode="group")
        apply_theme(fig, "Model Performance Comparison (Rolling-Origin CV)", 460)
        st.plotly_chart(fig, width="stretch")

    # Metrics table
    st.markdown("**📊 Full Metrics Table**")
    st.dataframe(comp.round(4), width="stretch")

    # Feature importance
    fi_files = list(MODELS_DIR.glob("feature_importance_*.csv"))
    if fi_files:
        st.markdown("**🔍 Feature Importance**")
        fi_tabs = st.tabs([f.stem.replace("feature_importance_", "") for f in fi_files])
        for tab, fpath in zip(fi_tabs, fi_files):
            with tab:
                fi_df = pd.read_csv(fpath)
                top = fi_df.nlargest(20, "importance").sort_values("importance")
                fig2 = go.Figure(go.Bar(
                    x=top["importance"], y=top["feature"], orientation="h",
                    marker=dict(color=top["importance"], colorscale=[[0,"#6366f1"],[1,"#06b6d4"]]),
                    text=top["importance"].round(4), textposition="outside",
                    textfont=dict(color="#f1f5f9", size=10),
                ))
                apply_theme(fig2, f"Feature Importance – {fpath.stem.replace('feature_importance_','')}", max(400, len(top)*25))
                st.plotly_chart(fig2, width="stretch")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: DECISION MATRIX
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Decision Matrix":
    st.markdown('<div class="section-header">🎯 Decision Matrix</div>', unsafe_allow_html=True)

    # Data loaded lazily per-file
    risk = load_risk()
    forecast = load_forecast()
    sku_master = load_sku_master()

    if risk.empty:
        st.warning("Run `python main.py` first.")
        st.stop()

    # Priority matrix: 2×2 → ABC class vs risk
    st.markdown("**2×2 Decision Matrix: ABC Class × Risk Level**")
    matrix_data = risk.groupby(["abc_class", "risk_level"]).size().unstack(fill_value=0)
    if not matrix_data.empty:
        fig = px.imshow(
            matrix_data.values,
            x=matrix_data.columns.tolist(),
            y=matrix_data.index.tolist(),
            color_continuous_scale=["#0f172a","#6366f1","#ef4444"],
            text_auto=True, aspect="auto",
        )
        apply_theme(fig, "Decision Matrix: ABC Class × Risk Level", 400)
        st.plotly_chart(fig, width="stretch")

    # Action priority table
    st.markdown("**📋 Priority Action List (Top 50 SKUs)**")
    action_cols = [c for c in [
        "priority_score","stock_code","description","category","abc_class",
        "risk_level","coverage_days","revenue_at_risk","recommendation"
    ] if c in risk.columns]
    top50 = risk.nlargest(50, "priority_score")[action_cols]
    st.dataframe(top50, width="stretch", height=450)

    csv = top50.to_csv(index=False).encode()
    st.download_button("⬇ Download Priority Action List", csv, "priority_actions.csv", "text/csv")

    # Bubble chart: priority_score vs coverage_days vs revenue_at_risk
    if "priority_score" in risk.columns:
        fig2 = px.scatter(
            risk.head(300),
            x="coverage_days", y="priority_score",
            size="revenue_at_risk" if "revenue_at_risk" in risk.columns else None,
            color="abc_class", hover_name="description" if "description" in risk.columns else "stock_code",
            color_discrete_map={"A":"#6366f1","B":"#06b6d4","C":"#f59e0b"},
            title="Priority Score vs Coverage Days (bubble = revenue at risk)",
        )
        apply_theme(fig2, height=420)
        st.plotly_chart(fig2, width="stretch")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: RECOMMENDATIONS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Recommendations":
    st.markdown('<div class="section-header">💡 Recommendations</div>', unsafe_allow_html=True)

    # Data loaded lazily per-file
    risk = load_risk()

    if risk.empty:
        st.warning("Run `python main.py` first.")
        st.stop()

    # Filter
    risk_f = risk.copy()
    if selected_cat != "All" and "category" in risk_f.columns:
        risk_f = risk_f[risk_f["category"] == selected_cat]

    REC_ICONS = {
        "🚨 Reorder Immediately":  ("danger-box",  "🚨"),
        "📦 Increase Purchase":    ("warning-box", "📦"),
        "📉 Reduce Purchase":      ("warning-box", "📉"),
        "📉 Markdown Promotion":   ("warning-box", "🏷"),
        "🔗 Bundle Offer":         ("info-box",    "🔗"),
        "🏷 Clearance Sale":       ("warning-box", "🏷"),
        "📊 Monitor Closely":      ("info-box",    "📊"),
        "✅ Healthy – No Action":  ("success-box", "✅"),
    }

    rec_groups = risk_f.groupby("recommendation")
    for rec, group in sorted(rec_groups, key=lambda x: -x[1]["priority_score"].mean()):
        box_class, icon = REC_ICONS.get(rec, ("info-box", "●"))
        with st.expander(f"{rec}  ·  {len(group)} SKUs", expanded=(box_class in ("danger-box","warning-box"))):
            st.markdown(f'<div class="{box_class}"><strong>{icon} {rec}</strong> – {len(group)} SKUs need this action</div>', unsafe_allow_html=True)
            display_cols = [c for c in [
                "stock_code","description","category","abc_class",
                "risk_score","coverage_days","revenue_at_risk","capital_locked","priority_score"
            ] if c in group.columns]
            st.dataframe(group[display_cols].sort_values("priority_score", ascending=False).head(50).round(2),
                width="stretch")

    st.markdown("<br>")
    csv = risk_f.to_csv(index=False).encode()
    st.download_button("⬇ Download All Recommendations", csv, "recommendations.csv", "text/csv")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: BUSINESS IMPACT
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Business Impact":
    st.markdown('<div class="section-header">💰 Business Impact</div>', unsafe_allow_html=True)

    # Data loaded lazily per-file
    impact = load_impact()
    risk = load_risk()

    if not impact:
        st.warning("Run `python main.py` first.")
        st.stop()

    # KPI cards
    c1, c2, c3 = st.columns(3)
    with c1:
        kpi_card("Revenue at Risk", f"£{impact.get('revenue_at_risk',0):,.0f}", "Next 14 days at risk", "negative")
    with c2:
        kpi_card("Capital Locked", f"£{impact.get('capital_locked',0):,.0f}", "Tied up in overstock", "negative")
    with c3:
        kpi_card("Forecast Revenue (28d)", f"£{impact.get('forecast_revenue_28d',0):,.0f}", "Expected demand", "positive")

    st.markdown("<br>")
    c4, c5, c6 = st.columns(3)
    with c4:
        kpi_card("Working Capital Saved", f"£{impact.get('working_capital_saved',0):,.0f}", "If overstock resolved", "positive")
    with c5:
        kpi_card("Potential Sales Increase", f"£{impact.get('potential_sales_increase',0):,.0f}", "Stockout recovery", "positive")
    with c6:
        kpi_card("Expected Profit Improvement", f"£{impact.get('expected_profit_improvement',0):,.0f}", "Net benefit", "positive")

    st.markdown("<br>")

    # Waterfall chart
    categories = ["Revenue at Risk", "Stockout Recovery", "Overstock Savings",
                  "Working Capital", "Net Benefit"]
    values_wf = [
        -impact.get("revenue_at_risk", 0),
        impact.get("potential_sales_increase", 0),
        impact.get("overstock_cost_monthly", 0),
        impact.get("working_capital_saved", 0),
        impact.get("expected_profit_improvement", 0),
    ]
    measures = ["absolute", "relative", "relative", "relative", "total"]
    fig = go.Figure(go.Waterfall(
        name="Impact", orientation="v", measure=measures,
        x=categories, y=values_wf,
        text=[f"£{abs(v):,.0f}" for v in values_wf],
        textposition="outside", textfont=dict(color="#f1f5f9"),
        connector=dict(line=dict(color="#334155")),
        decreasing=dict(marker=dict(color="#ef4444")),
        increasing=dict(marker=dict(color="#10b981")),
        totals=dict(marker=dict(color="#6366f1")),
    ))
    apply_theme(fig, "Business Impact Waterfall (£)", 480)
    st.plotly_chart(fig, width="stretch")

    # Summary table
    st.markdown("**📊 Impact Summary**")
    impact_rows = [
        ("Metric", "Value", "Description"),
        ("Revenue at Risk", f"£{impact.get('revenue_at_risk',0):,.2f}", "Estimated lost revenue from stockouts"),
        ("Forecast Revenue (28d)", f"£{impact.get('forecast_revenue_28d',0):,.2f}", "ML-predicted 28-day revenue"),
        ("Inventory Value", f"£{impact.get('inventory_value',0):,.2f}", "Current total inventory at cost"),
        ("Capital Locked", f"£{impact.get('capital_locked',0):,.2f}", "Capital in excess inventory"),
        ("Working Capital Saved", f"£{impact.get('working_capital_saved',0):,.2f}", "If overstocks reduced"),
        ("Potential Sales Increase", f"£{impact.get('potential_sales_increase',0):,.2f}", "Recovered from stockout elimination"),
        ("Monthly Overstock Cost", f"£{impact.get('overstock_cost_monthly',0):,.2f}", "Holding cost of excess stock"),
        ("Stockout Cost", f"£{impact.get('stockout_cost',0):,.2f}", "Revenue loss + brand penalty"),
        ("Expected Profit Improvement", f"£{impact.get('expected_profit_improvement',0):,.2f}", "Net benefit of FORESIGHT actions"),
    ]
    impact_df = pd.DataFrame(impact_rows[1:], columns=impact_rows[0])
    st.dataframe(impact_df, width="stretch", hide_index=True)

    # SKU breakdown
    st.markdown("**🏷 Risk Count by Level**")
    risk_counts = [
        ("Critical Stockout", impact.get("critical_stockout_count", 0), "#ef4444"),
        ("High Stockout", impact.get("high_stockout_count", 0), "#f97316"),
        ("Overstock", impact.get("overstock_count", 0), "#8b5cf6"),
        ("Healthy", impact.get("healthy_count", 0), "#10b981"),
    ]
    fig2 = go.Figure(go.Bar(
        x=[r[0] for r in risk_counts],
        y=[r[1] for r in risk_counts],
        marker_color=[r[2] for r in risk_counts],
        text=[r[1] for r in risk_counts],
        textposition="outside",
        textfont=dict(color="#f1f5f9"),
    ))
    apply_theme(fig2, "SKU Count by Risk Level", 360)
    st.plotly_chart(fig2, width="stretch")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: API TESTING
# ─────────────────────────────────────────────────────────────────────────────
elif page == "API Testing":
    st.markdown('<div class="section-header">🌐 API Testing Console</div>', unsafe_allow_html=True)

    API_BASE = st.text_input("API Base URL", "http://localhost:8000", key="api_url")

    def api_call(method, path, payload=None):
        url = f"{API_BASE}{path}"
        try:
            if method == "GET":
                r = requests.get(url, timeout=30)
            else:
                r = requests.post(url, json=payload, timeout=120)
            return r.status_code, r.json()
        except requests.exceptions.ConnectionError:
            return None, {"error": "Cannot connect to API. Make sure `uvicorn service.api:app --reload` is running."}
        except Exception as e:
            return None, {"error": str(e)}

    api_tabs = st.tabs(["❤️ Health", "📊 Forecast", "⚠️ Risk", "💡 Recommendations", "📉 Metrics", "🔮 Predict", "🚀 Train"])

    with api_tabs[0]:
        if st.button("GET /health", key="health_btn"):
            with st.spinner("Calling API…"):
                status, resp = api_call("GET", "/health")
            if status == 200:
                st.success(f"✅ Status {status}")
            else:
                st.error(f"❌ Status {status}")
            st.json(resp)

    with api_tabs[1]:
        col1, col2 = st.columns(2)
        with col1:
            f_sku = st.text_input("stock_code filter (blank = all)", "", key="fcast_api_sku")
        if st.button("GET /forecast", key="fcast_btn"):
            path = f"/forecast?stock_code={f_sku}" if f_sku else "/forecast"
            with st.spinner("Fetching…"):
                status, resp = api_call("GET", path)
            st.write(f"Status: {status}")
            if isinstance(resp, dict) and "data" in resp:
                st.dataframe(pd.DataFrame(resp["data"]).head(50), width="stretch")
            else:
                st.json(resp)

    with api_tabs[2]:
        rl = st.selectbox("risk_level", ["", "Critical Stockout Risk", "High Stockout Risk", "Overstock Risk", "Healthy"], key="risk_api")
        if st.button("GET /risk", key="risk_btn"):
            path = f"/risk?risk_level={rl}" if rl else "/risk"
            with st.spinner("Fetching…"):
                status, resp = api_call("GET", path)
            st.write(f"Status: {status}")
            if isinstance(resp, dict) and "data" in resp:
                st.dataframe(pd.DataFrame(resp["data"]).head(50), width="stretch")
            else:
                st.json(resp)

    with api_tabs[3]:
        top_n = st.slider("Top N", 5, 100, 20, key="rec_n")
        if st.button("GET /recommendation", key="rec_btn"):
            with st.spinner("Fetching…"):
                status, resp = api_call("GET", f"/recommendation?top_n={top_n}")
            st.write(f"Status: {status}")
            st.json(resp)

    with api_tabs[4]:
        if st.button("GET /model_metrics", key="metrics_btn"):
            with st.spinner("Fetching…"):
                status, resp = api_call("GET", "/model_metrics")
            st.write(f"Status: {status}")
            st.json(resp)

    with api_tabs[5]:
        p_sku = st.text_input("stock_code", "85123A", key="pred_sku")
        p_start = st.date_input("start_date", datetime.now().date(), key="pred_start")
        p_end   = st.date_input("end_date", (datetime.now() + timedelta(days=7)).date(), key="pred_end")
        if st.button("POST /predict", key="pred_btn"):
            payload = {"stock_code": p_sku, "start_date": str(p_start), "end_date": str(p_end)}
            with st.spinner("Predicting…"):
                status, resp = api_call("POST", "/predict", payload)
            st.write(f"Status: {status}")
            st.json(resp)

    with api_tabs[6]:
        st.warning("⚠️ Training takes several minutes. This will re-run the entire ML pipeline.")
        skip = st.checkbox("Skip model training (only re-process data)", key="skip_train")
        if st.button("POST /train", key="train_btn", type="primary"):
            with st.spinner("Training in progress… (this may take a few minutes)"):
                payload = {"skip_training": skip}
                status, resp = api_call("POST", "/train", payload)
            st.write(f"Status: {status}")
            st.json(resp)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: PROJECT SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Project Summary":
    st.markdown('<div class="section-header">📄 Project Summary</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="display:grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom:24px;">

      <div class="kpi-card" style="text-align:left;">
        <div style="font-size:1.4rem; font-weight:800; color:#f1f5f9; margin-bottom:8px;">🔮 Project FORESIGHT</div>
        <div style="color:#94a3b8; font-size:0.9rem; line-height:1.8;">
          <strong>Client:</strong> NorthBay Living (D2C Retail)<br>
          <strong>Domain:</strong> Demand Forecasting & Inventory Intelligence<br>
          <strong>Type:</strong> End-to-End ML Production System<br>
          <strong>Dataset:</strong> Online Retail II UCI (541,909 transactions)
        </div>
      </div>

      <div class="kpi-card" style="text-align:left;">
        <div style="font-size:1.1rem; font-weight:700; color:#f1f5f9; margin-bottom:8px;">🏗 Architecture</div>
        <div style="color:#94a3b8; font-size:0.85rem; line-height:1.9;">
          Raw CSV → Preprocessing → 4 FORESIGHT Tables<br>
          → Feature Engineering (25+ features)<br>
          → 6 ML Models (Rolling-Origin CV)<br>
          → Risk Engine → Recommendations<br>
          → Streamlit Dashboard + FastAPI REST
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**📊 Data Pipeline: 4 FORESIGHT Tables**")
        tables = [
            ("sales_daily.csv", "Daily SKU-level sales aggregations, promotion flags"),
            ("sku_master.csv", "Product catalog with category, ABC class, lead time, EOQ"),
            ("calendar.csv", "Date dimension: holidays, seasons, Black Friday, Fourier"),
            ("inventory_snapshots.csv", "Simulated daily inventory with (s,Q) replenishment policy"),
        ]
        for name, desc in tables:
            st.markdown(f"""
            <div class="info-box" style="margin:6px 0; padding:10px 14px;">
              <code style="color:#06b6d4;">{name}</code><br>
              <span style="font-size:0.82rem; color:#94a3b8;">{desc}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("**🤖 Models Implemented**")
        models = ["Seasonal Naive (Baseline)", "Random Forest", "Gradient Boosting",
                  "LightGBM", "XGBoost", "Prophet"]
        for m in models:
            st.markdown(f"  • {m}")

    with col2:
        st.markdown("**⚠️ Risk Levels**")
        for level, color in RISK_COLORS.items():
            st.markdown(f'<span class="badge" style="background:rgba(99,102,241,0.1); color:{color}; margin:2px 0;">■</span> {level}<br>', unsafe_allow_html=True)

        st.markdown("**📌 Documented Simulation Assumptions**")
        assumptions = [
            ("A", "Categories", "Keyword matching on product descriptions"),
            ("B", "Inventory", "Starting stock = 12-week demand; (s,Q) replenishment"),
            ("C", "ROP", "avg_daily × lead_time × 1.5 safety factor"),
            ("D", "Lead Times", "Uniform(3–14) days, seed=42 (reproducible)"),
            ("E", "Promotions", "Flagged when price < 85% of 30-day rolling avg"),
            ("F", "EOQ", "√(2·D·S/H); ordering_cost=£50, H=20% of price"),
        ]
        for code, name, desc in assumptions:
            st.markdown(f"""
            <div style="padding:6px 10px; margin:3px 0; background:#1e293b;
                        border-left:3px solid #6366f1; border-radius:4px; font-size:0.82rem;">
              <strong style="color:#6366f1;">[{code}] {name}:</strong>
              <span style="color:#94a3b8;"> {desc}</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("**🚀 Quick Start Commands**")
    st.code("""# 1. Install dependencies
pip install -r requirements.txt

# 2. Place dataset in data/raw/ (Online Retail II CSV from Kaggle)

# 3. Run full pipeline
python main.py

# 4. Launch dashboard
streamlit run app/streamlit_app.py

# 5. Start API server
uvicorn service.api:app --reload --port 8000""", language="bash")


# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<hr style="border-color:#334155; margin:40px 0 16px;">
<div style="text-align:center; color:#475569; font-size:0.75rem; padding-bottom:20px;">
  🔮 <strong style="color:#6366f1;">Project FORESIGHT</strong> &nbsp;·&nbsp;
  NorthBay Living &nbsp;·&nbsp;
  Built with Streamlit, FastAPI, LightGBM, XGBoost &nbsp;·&nbsp;
  v1.0.0
</div>
""", unsafe_allow_html=True)
