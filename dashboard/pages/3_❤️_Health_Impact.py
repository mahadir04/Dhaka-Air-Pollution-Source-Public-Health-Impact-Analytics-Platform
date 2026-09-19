"""
Page 3 — Health Impact
CRF-based health burden, population-weighted exposure, ranking.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

st.set_page_config(page_title="Health Impact — AirImpact Dhaka", page_icon="❤️", layout="wide")
st.markdown("# ❤️ Health Impact Analysis")
st.markdown("---")

# ── Load data ─────────────────────────────────────────────────────────────────
burden_path = OUTPUTS_DIR / "health_burden_table.csv"
ranking_path = OUTPUTS_DIR / "health_ranking_summary.csv"

if not burden_path.exists():
    st.error("**health_burden_table.csv not found.** Run notebook Section 6 first.")
    st.stop()

burden_df = pd.read_csv(burden_path)
ranking_df = pd.read_csv(ranking_path) if ranking_path.exists() else pd.DataFrame()

# ── KPI cards ─────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)

if not burden_df.empty:
    total_deaths = burden_df["attrib_deaths"].sum()
    max_ar = burden_df["ar_pct"].max()
    mean_conc = burden_df["mean_conc"].mean()
    n_stations = burden_df["station"].nunique()

    c1.metric("📊 Stations Analyzed", n_stations)
    c2.metric("💀 Total Attributable Deaths", f"{total_deaths:,.0f}")
    c3.metric("📈 Max Attributable Risk", f"{max_ar:.2f}%")
    c4.metric("🔴 Mean PM2.5", f"{mean_conc:.1f} µg/m³")

st.markdown("---")

# ── Health burden by station and CRF ──────────────────────────────────────────
st.markdown("### 📊 Attributable Deaths by Station & CRF Coefficient")

if "crf_label" in burden_df.columns:
    crf_labels = burden_df["crf_label"].unique()

    fig, ax = plt.subplots(figsize=(14, max(4, burden_df["station"].nunique() * 0.5)))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    crf_colors = {
        "PM2.5 Long-Term (Pope et al.)": "#e74c3c",
        "PM2.5 Annual (India D-in-D 2024)": "#e67e22",
    }

    width = 0.35
    stations = sorted(burden_df["station"].unique())
    y = np.arange(len(stations))

    for i, crf in enumerate(crf_labels):
        subset = burden_df[burden_df["crf_label"] == crf]
        deaths = [subset[subset["station"] == s]["attrib_deaths"].sum() for s in stations]
        color = crf_colors.get(crf, f"#{hash(crf) % 0xFFFFFF:06x}")
        ax.barh(y + i * width, deaths, width, label=crf, color=color, alpha=0.85,
                edgecolor="white", linewidth=0.3)

    ax.set_yticks(y + width / 2)
    ax.set_yticklabels(stations, fontsize=9)
    ax.set_xlabel("Attributable Deaths (annual)", color="white", fontsize=11)
    ax.set_title("Health Burden by Station — CRF Sensitivity", color="white", fontsize=14)
    ax.legend(facecolor="#2a2a4e", edgecolor="white", labelcolor="white", fontsize=9, loc="lower right")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("white")
        spine.set_alpha(0.3)
    ax.grid(alpha=0.15, color="white", axis="x")

    st.pyplot(fig)
    plt.close(fig)

# ── Population × Concentration bubble chart ──────────────────────────────────
st.markdown("### 🫧 Population vs PM2.5 Concentration")

if "catchment_pop" in burden_df.columns and "mean_conc" in burden_df.columns:
    # Use one CRF for the bubble chart to avoid duplicates
    bubble_df = burden_df.drop_duplicates(subset=["station"]).copy()

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    scatter = ax.scatter(
        bubble_df["mean_conc"],
        bubble_df["catchment_pop"],
        s=bubble_df["attrib_deaths"] * 3,
        c=bubble_df["ar_pct"],
        cmap="YlOrRd",
        alpha=0.8,
        edgecolors="white",
        linewidth=0.8,
    )

    for _, row in bubble_df.iterrows():
        label = row["station"]
        if len(str(label)) > 18:
            label = str(label)[:16] + "…"
        ax.annotate(
            label,
            (row["mean_conc"], row["catchment_pop"]),
            textcoords="offset points", xytext=(8, 5),
            fontsize=8, color="white", alpha=0.8,
        )

    cbar = fig.colorbar(scatter, ax=ax, shrink=0.7)
    cbar.set_label("Attributable Risk (%)", color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")

    ax.axvline(5.0, color="#2ecc71", linestyle="--", alpha=0.7, label="WHO annual (5 µg/m³)")
    ax.set_xlabel("Mean PM2.5 (µg/m³)", color="white", fontsize=11)
    ax.set_ylabel("Catchment Population", color="white", fontsize=11)
    ax.set_title("Population Exposure — Bubble Size = Attributable Deaths", color="white", fontsize=14)
    ax.legend(facecolor="#2a2a4e", edgecolor="white", labelcolor="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("white")
        spine.set_alpha(0.3)
    ax.grid(alpha=0.15, color="white")

    st.pyplot(fig)
    plt.close(fig)

# ── Health-Impact Ranking ────────────────────────────────────────────────────
if not ranking_df.empty:
    st.markdown("### 🏆 Health-Impact Ranking (Population-Weighted)")

    col1, col2 = st.columns(2)

    with col1:
        # Ranking by health impact score
        station_rank = (
            ranking_df.groupby("name")
            .agg({"health_impact_score": "sum", "mean_pm25": "mean", "catchment_pop": "first"})
            .sort_values("health_impact_score", ascending=True)
            .reset_index()
        )

        fig, ax = plt.subplots(figsize=(8, max(4, len(station_rank) * 0.5)))
        fig.patch.set_facecolor("#1a1a2e")
        ax.set_facecolor("#1a1a2e")

        colors = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(station_rank)))
        ax.barh(station_rank["name"], station_rank["health_impact_score"],
                color=colors, edgecolor="white", linewidth=0.3)
        ax.set_xlabel("Health Impact Score", color="white", fontsize=11)
        ax.set_title("Stations Ranked by Health Impact", color="white", fontsize=13)
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("white")
            spine.set_alpha(0.3)
        ax.grid(alpha=0.15, color="white", axis="x")

        st.pyplot(fig)
        plt.close(fig)

    with col2:
        # Raw pollution ranking for comparison
        raw_rank = (
            ranking_df.groupby("name")
            .agg({"mean_pm25": "mean"})
            .sort_values("mean_pm25", ascending=True)
            .reset_index()
        )

        fig, ax = plt.subplots(figsize=(8, max(4, len(raw_rank) * 0.5)))
        fig.patch.set_facecolor("#1a1a2e")
        ax.set_facecolor("#1a1a2e")

        colors_raw = plt.cm.Reds(np.linspace(0.3, 0.9, len(raw_rank)))
        ax.barh(raw_rank["name"], raw_rank["mean_pm25"],
                color=colors_raw, edgecolor="white", linewidth=0.3)
        ax.axvline(5.0, color="#2ecc71", linestyle="--", alpha=0.7, label="WHO guideline")
        ax.set_xlabel("Mean PM2.5 (µg/m³)", color="white", fontsize=11)
        ax.set_title("Stations Ranked by Raw PM2.5", color="white", fontsize=13)
        ax.legend(facecolor="#2a2a4e", edgecolor="white", labelcolor="white")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("white")
            spine.set_alpha(0.3)
        ax.grid(alpha=0.15, color="white", axis="x")

        st.pyplot(fig)
        plt.close(fig)

    st.markdown("""
    > **Why do rankings differ?** The health-impact ranking weights pollution by
    > catchment population. A station with moderate PM2.5 but a very large
    > surrounding population may rank higher than a heavily polluted but
    > sparsely populated area.
    """)

# ── Data tables ──────────────────────────────────────────────────────────────
st.markdown("### 📋 Health Burden Data")

tab1, tab2 = st.tabs(["Long-Term Burden", "Health Impact Ranking"])

with tab1:
    st.dataframe(burden_df, hide_index=True)

with tab2:
    if not ranking_df.empty:
        display_cols = [c for c in ["rank", "name", "season", "mean_pm25", "ar_pct",
                                     "catchment_pop", "health_impact_score"] if c in ranking_df.columns]
        st.dataframe(ranking_df[display_cols].head(20), hide_index=True)
    else:
        st.info("Ranking data not available.")

st.markdown("---")
st.caption(
    "⚠️ Short-term and long-term burdens answer DIFFERENT clinical questions "
    "and must NOT be summed. See Methodology page for details."
)
