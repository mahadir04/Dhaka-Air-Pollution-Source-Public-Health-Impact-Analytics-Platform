"""
Page 5 — Methodology & Documentation
Comprehensive documentation of epidemiological concentration-response functions,
source-signature diagnostic rules, ML forecasting architecture, and known limitations.
"""

import streamlit as st
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

st.set_page_config(page_title="Methodology — AirImpact Dhaka", page_icon="📋", layout="wide")
st.markdown("# 📋 Methodology & Scientific Framework")
st.markdown("Mathematical formulations, epidemiological literature grounding, source diagnostic logic, and engineering constraints.")
st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs([
    "❤️ Epidemiological CRF Framework",
    "🏭 Source-Signature Rules",
    "🤖 ML Forecasting Architecture",
    "⚠️ Limitations & Data Governance"
])

with tab1:
    st.subheader("Concentration-Response Functions (CRFs) & Health Burden Estimation")
    st.markdown("""
    Unlike raw AQI index reporting, this platform translates ambient pollutant exposure into **quantified public health burden** 
    grounded in peer-reviewed epidemiological cohort studies and WHO guideline syntheses.
    """)

    st.markdown("#### 1. Mathematical Formulation")
    st.latex(r"RR = \exp\left(\beta \cdot \max(0, C - C_0)\right)")
    st.latex(r"AR = 1 - \frac{1}{RR} = 1 - \exp\left(-\beta \cdot \max(0, C - C_0)\right)")
    st.latex(r"\text{Attributable Excess Deaths} = AR \times B \times P_{\text{catchment}}")

    st.markdown("""
    Where:
    - **$C$**: Ambient annual/seasonal mean pollutant concentration ($\mu\text{g/m}^3$)
    - **$C_0$**: Counterfactual baseline threshold ($5.0\,\mu\text{g/m}^3$ annual WHO 2021 guideline for $\text{PM}_{2.5}$)
    - **$\beta$**: Epidemiological concentration-response coefficient per unit exposure ($\mu\text{g/m}^3$)
    - **$AR$**: Attributable Risk fraction (proportion of total mortality attributable to pollution above baseline)
    - **$B$**: Baseline all-cause mortality rate for Dhaka metro ($\approx 5.48 \text{ per } 1,000 \text{ population/year}$, BBS 2022)
    - **$P_{\text{catchment}}$**: Gridded population within the station's spatial sphere of influence
    """)

    st.markdown("#### 2. Published Epidemiological Coefficients Applied")
    crf_table = pd.DataFrame([
        {
            "Study / Source": "Pope et al. (ACS Cohort Study)",
            "Pollutant": "PM2.5",
            "Exposure Horizon": "Long-Term (Annual)",
            "Effect Estimate (per 10 µg/m³)": "+8.0% (β = 0.0080)",
            "Endpoint": "All-cause cardiopulmonary mortality",
            "Citation": "Pope et al., JAMA 2002; Circulation 2004"
        },
        {
            "Study / Source": "India Diff-in-Diff Cohort (2024)",
            "Pollutant": "PM2.5",
            "Exposure Horizon": "Long-Term (Annual)",
            "Effect Estimate (per 10 µg/m³)": "+8.6% (β = 0.0086)",
            "Endpoint": "All-cause mortality in South Asian urban context",
            "Citation": "Lancet Planetary Health / South Asia Environmental Health Group (2024)"
        },
        {
            "Study / Source": "Orellano et al. (WHO AQG 2021 Systematic Review)",
            "Pollutant": "PM10 / PM2.5",
            "Exposure Horizon": "Short-Term (Daily)",
            "Effect Estimate (per 10 µg/m³)": "+0.4% (β = 0.0004)",
            "Endpoint": "All-cause daily excess mortality",
            "Citation": "Orellano et al., Environment International 2020 (WHO AQG 2021)"
        },
    ])
    st.table(crf_table)

    st.info("💡 **Sensitivity Analysis**: Burden is reported as a sensitivity range bounded by the Pope et al. lower estimate and India regional upper estimate, reflecting structural uncertainty.")


with tab2:
    st.subheader("Diagnostic Source-Signature Rules (Spark SQL)")
    st.markdown("""
    Source identification uses deterministic, rule-based diagnostic pattern matching over grouped spatio-temporal aggregations, 
    calibrated against Dhaka's documented emission inventory calendars (Begum, Biswas & Hopke 2011; DoE Bangladesh).
    """)

    st.markdown("""
    | Source Category | Temporal Window | Meteorological & Pollutant Signature | Spark SQL Diagnostic Rule |
    | :--- | :--- | :--- | :--- |
    | **🚗 Vehicular Traffic** | Morning Rush (07:00–09:00)<br>Evening Rush (17:00–20:00) | Weekday elevation, vehicular $\\text{NO}_2$ spikes, localized fine fraction elevation | `(hour IN (7,8,9,17,18,19,20) AND dow BETWEEN 1 AND 5)` |
    | **🧱 Brick Kilns (FCBK/ZZK)** | Dry Season (Nov–Mar)<br>Morning inversion (06:00–12:00) | Heavy $\\text{SO}_2$ plume trapping under low planetary boundary layer, downwind transport | `(month IN (11,12,1,2,3) AND hour BETWEEN 6 AND 12)` |
    | **🔥 Biomass & Open Burning** | Post-Monsoon & Winter<br>Evening/Night (18:00–04:00) | $\\text{PM}_{2.5}$ elevation without proportional $\\text{NO}_2$ traffic rise; trash/crop burning | `(month IN (10,11,12,1,2) AND (hour >= 18 OR hour <= 4))` |
    | **🏗 Construction & Road Dust** | Dry Season / Pre-Monsoon<br>Work hours (08:00–18:00) | Coarse particulate dominant ($\\text{PM}_{10} \\gg \\text{PM}_{2.5}$), low gaseous combustion | `(season IN ('Winter', 'Pre-Monsoon') AND hour BETWEEN 8 AND 18)` |
    """)

    st.markdown("#### Diagnostic Scoring Scale")
    st.markdown("""
    Each observation receives a normalized score $[0.0, 1.0]$ for each source candidate. Scores are aggregated by station and season to diagnose dominant emission regimes:
    """)
    st.code("""
-- Example Spark SQL Expression for Brick Kiln Signature
CASE 
  WHEN month IN (11, 12, 1, 2, 3) AND hour BETWEEN 6 AND 12 THEN 1.0
  WHEN month IN (11, 12, 1, 2, 3) THEN 0.6
  ELSE 0.1 
END AS score_brick_kiln
    """, language="sql")


with tab3:
    st.subheader("Machine Learning Forecasting Architecture")
    st.markdown("""
    The forecasting module provides lightweight, short-term (1-hour ahead) predictive capabilities designed to run directly on distributed Spark clusters or edge sklearn runtimes.
    """)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### Feature Engineering Vector")
        st.markdown("""
        1. **Autoregressive Lags**:
           - `pm25_lag1`: 1-hour prior concentration
           - `pm25_lag24`: 24-hour diurnal cycle lag
           - `pm25_roll24`: 24-hour rolling window moving average
        2. **Calendar & Diurnal Encodings**:
           - `hour`: Diurnal hour of the day (0–23)
           - `dow`: Day of week (1=Sunday through 7=Saturday)
           - `month`: Calendar month (1–12)
           - `season_enc`: Ordinal seasonal cycle (Winter, Pre-monsoon, Monsoon, Post-monsoon)
        3. **Meteorological Covariates**:
           - `temp_c`: Ambient dry-bulb temperature (°C)
           - `rh_pct`: Relative humidity (%)
           - `wind_speed_kmh`: Surface wind velocity (km/h)
           - `blh_m`: Boundary layer height (planetary dispersion depth)
        """)

    with col_b:
        st.markdown("#### Candidate Model Benchmark Suite")
        st.markdown("""
        All models are evaluated on a strict chronological **80% train / 20% test holdout split** (no future data leakage):
        - **Naive Baseline**: Pure persistence ($\hat{y}_{t+1} = y_t$)
        - **Linear Regression**: L2-regularized Ridge baseline
        - **Random Forest Regressor**: 100 trees, maximum depth 8 (Spark MLlib)
        - **Gradient Boosted Trees (GBT)**: 80 iterations, max depth 5, step size 0.1 (Spark MLlib)
        - **LightGBM**: 300 estimators, leaf-wise histogram gradient boosting
        - **XGBoost**: 300 estimators, depth 5, exact/hist tree method
        
        **Model Export Protocol**:
        The training pipeline exports:
        - Trained weights to `outputs/model/` (Spark) or `outputs/model_sklearn/` (joblib)
        - Standardized metadata in `outputs/model_metadata.json`
        - Evaluated test metrics in `outputs/forecast_metrics.csv`
        """)


with tab4:
    st.subheader("Operational Constraints & Ethical Governance")
    st.markdown("""
    #### 1. Low-Cost Sensor Network Characteristics
    - The OpenAQ Dhaka station network comprises primarily low-cost optical particle counters (OPCs).
    - OPCs measure particle light scattering, which is susceptible to hygroscopic particle swelling under Dhaka's extreme monsoon relative humidity (>80%).
    - Ingestion pipelines apply IQR clipping and QC boundaries to suppress unphysical humidity artifacts.

    #### 2. Pollutant Dimensionality Scope
    - While the platform architecture is fully multi-pollutant ($\text{PM}_{2.5}, \text{PM}_{10}, \text{NO}_2, \text{SO}_2, \text{CO}, \text{O}_3$), current real-time reporting across Dhaka metro stations is predominantly $\text{PM}_{2.5}$.
    - Gaseous diagnostic indicators utilize seasonal calendars and meteorology as documented proxies until municipal continuous ambient air quality monitoring (CAMS) stations resume full public gas telemetry.

    #### 3. Spatial Allocation via Gridded Rasters
    - Population exposure uses ORNL LandScan 1km gridded population rasters intersected with a 3km radial buffer around each monitoring station.
    - This provides a substantially more realistic measure of human exposure than assigning equal weight to industrial periphery stations and dense residential wards (e.g., Lalbagh, Mirpur).
    """)
