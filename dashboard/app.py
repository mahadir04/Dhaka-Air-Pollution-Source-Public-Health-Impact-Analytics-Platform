"""
Dhaka Air Pollution — Streamlit Dashboard
Main entry point.

Launch:  streamlit run dashboard/app.py
"""

import streamlit as st
from pathlib import Path
import json, os, sys

# ── Project path ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = PROJECT_ROOT / "figures"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AirImpact Dhaka — Air Pollution Analytics",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS for premium dark aesthetic ─────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #1a1a3e 50%, #24243e 100%);
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #141432 0%, #1e1e4a 100%);
        border-right: 1px solid rgba(255,255,255,0.05);
    }

    /* KPI metric cards */
    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 16px;
        backdrop-filter: blur(10px);
    }
    div[data-testid="stMetric"] label {
        color: #a0a0cc !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #e0e0ff !important;
        font-weight: 700 !important;
    }

    /* Headers */
    h1, h2, h3 {
        color: #e8e8ff !important;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(255,255,255,0.03);
        border-radius: 8px;
    }

    /* Tables */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
    }

    /* Status badge */
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85em;
        font-weight: 600;
    }
    .status-ok { background: rgba(46,204,113,0.2); color: #2ecc71; }
    .status-warn { background: rgba(241,196,15,0.2); color: #f1c40f; }
    .status-err { background: rgba(231,76,60,0.2); color: #e74c3c; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# 🌍 AirImpact Dhaka")
    st.markdown("**Air Pollution & Public Health Analytics**")
    st.markdown("---")

    # Data status
    meta_path = OUTPUTS_DIR / "model_metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        st.markdown(
            f'<span class="status-badge status-ok">✓ Model loaded</span>',
            unsafe_allow_html=True
        )
        st.caption(f"**Model**: {meta.get('model_name', 'N/A')}")
        metrics = meta.get("metrics", {})
        if metrics.get("rmse"):
            st.caption(f"**RMSE**: {metrics['rmse']:.3f} µg/m³")
            st.caption(f"**R²**: {metrics.get('r2', 'N/A')}")
    else:
        st.markdown(
            '<span class="status-badge status-warn">⚠ No model found</span>',
            unsafe_allow_html=True
        )
        st.caption("Run the notebook first to train and export a model.")

    st.markdown("---")

    # Check for required output files
    required_files = [
        "enriched_data.parquet",
        "model_metadata.json",
        "health_burden_table.csv",
        "health_ranking_summary.csv",
    ]
    missing = [f for f in required_files if not (OUTPUTS_DIR / f).exists()]
    if missing:
        st.warning(f"Missing outputs: {', '.join(missing)}")
        st.info("Run the notebook cells (Sections 1-9) to generate all required data.")

    st.markdown("---")
    st.caption("Built with PySpark + Streamlit")
    st.caption("Data: OpenAQ v3 API")
    st.caption("© 2026 AirImpact Dhaka")


# ── Home page content ─────────────────────────────────────────────────────────
st.markdown("# 🌍 Dhaka Air Pollution & Public Health Analytics")
st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### 📊 Data Pipeline
    - **12 monitoring stations** across Dhaka metro
    - **OpenAQ v3 API** real-time data
    - Weather enrichment via Open-Meteo
    - Population-weighted exposure (LandScan)
    """)

with col2:
    st.markdown("""
    ### 🔬 Analytics Engine
    - Source-signature classification (4 types)
    - WHO CRF-based health burden
    - Population-weighted health impact ranking
    - Sensitivity analysis across CRF coefficients
    """)

with col3:
    st.markdown("""
    ### 📈 Forecasting
    - Trained via notebook (GBT / LightGBM / XGBoost)
    - Feature engineering with Spark Window functions
    - Next-hour PM2.5 prediction
    - Interactive prediction interface
    """)

st.markdown("---")

st.info(
    "👈 **Navigate using the sidebar** to explore Source Analysis, Health Impact, "
    "Forecasting, and Methodology pages."
)

# Show pipeline status
if (OUTPUTS_DIR / "enriched_data.parquet").exists():
    import pandas as pd
    try:
        df = pd.read_parquet(OUTPUTS_DIR / "enriched_data.parquet")
        c1, c2, c3, c4 = st.columns(4)
        n_stations = df["name"].nunique() if "name" in df.columns else "N/A"
        pm25_mean = df["pm25"].mean() if "pm25" in df.columns else None
        pm25_max = df["pm25"].max() if "pm25" in df.columns else None
        who_exceed = (
            (df["pm25"] > 15).mean() * 100 if "pm25" in df.columns else None
        )
        c1.metric("Stations", n_stations)
        c2.metric("Mean PM2.5", f"{pm25_mean:.1f} µg/m³" if pm25_mean else "N/A")
        c3.metric("Max PM2.5", f"{pm25_max:.1f} µg/m³" if pm25_max else "N/A")
        c4.metric("WHO Exceedance", f"{who_exceed:.1f}%" if who_exceed else "N/A")
    except Exception as e:
        st.error(f"Error loading data: {e}")
else:
    st.warning(
        "**No data available yet.** Run the notebook "
        "(`notebooks/notebook9d70e46c6b.ipynb`) from top to bottom to generate "
        "all required output files."
    )
