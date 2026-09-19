"""
Page 2 — Source Analysis
Source-signature distribution, seasonal patterns, weather correlations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = PROJECT_ROOT / "figures"

st.set_page_config(page_title="Source Analysis — AirImpact Dhaka", page_icon="🔍", layout="wide")
st.markdown("# 🔍 Source-Signature Analysis")
st.markdown("---")

# ── Load source scores ────────────────────────────────────────────────────────
source_csv = OUTPUTS_DIR / "5_source_scores_by_station_season.csv"
enriched_path = OUTPUTS_DIR / "enriched_data.parquet"

if not source_csv.exists():
    st.error("**Source scores not found.** Run notebook Section 5 first.")
    st.stop()

source_df = pd.read_csv(source_csv)

SOURCES = ["score_traffic", "score_brick_kiln", "score_biomass", "score_construction"]
SOURCE_LABELS = {
    "score_traffic":      "Traffic",
    "score_brick_kiln":   "Brick Kilns",
    "score_biomass":      "Biomass Burning",
    "score_construction": "Construction Dust",
}
SOURCE_COLORS = {
    "Traffic":           "#e74c3c",
    "Brick Kilns":       "#e67e22",
    "Biomass Burning":   "#2ecc71",
    "Construction Dust": "#3498db",
}

available_sources = [s for s in SOURCES if s in source_df.columns]

st.markdown("""
> **Source-signature rules** classify each PM2.5 reading based on temporal patterns
> (hour, day-of-week, season), pollutant ratios, and meteorological conditions.
> Scores range from 0 (no match) to 1 (strong match).
""")

# ── Overall source distribution (pie chart) ──────────────────────────────────
st.markdown("### 📊 Overall Source-Signature Distribution")

col1, col2 = st.columns(2)

with col1:
    mean_scores = {SOURCE_LABELS[s]: source_df[s].mean() for s in available_sources}
    total = sum(mean_scores.values())
    if total > 0:
        labels = list(mean_scores.keys())
        sizes = list(mean_scores.values())
        colors = [SOURCE_COLORS.get(l, "#888") for l in labels]

        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor("#1a1a2e")
        ax.set_facecolor("#1a1a2e")

        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, autopct="%1.1f%%",
            colors=colors, startangle=140,
            textprops={"color": "white", "fontsize": 11},
            pctdistance=0.75,
            wedgeprops={"edgecolor": "#1a1a2e", "linewidth": 2},
        )
        for t in autotexts:
            t.set_fontweight("bold")

        ax.set_title("Mean Source-Signature Scores", color="white", fontsize=14, pad=20)
        st.pyplot(fig)
        plt.close(fig)

with col2:
    # Bar chart of mean scores
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    labels_bar = [SOURCE_LABELS[s] for s in available_sources]
    values_bar = [source_df[s].mean() for s in available_sources]
    colors_bar = [SOURCE_COLORS.get(l, "#888") for l in labels_bar]

    bars = ax.barh(labels_bar, values_bar, color=colors_bar, edgecolor="white", linewidth=0.5, alpha=0.85)
    ax.set_xlabel("Mean Score", color="white", fontsize=11)
    ax.set_title("Average Source Scores Across All Stations", color="white", fontsize=14)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("white")
        spine.set_alpha(0.3)
    ax.grid(alpha=0.15, color="white", axis="x")

    for bar, val in zip(bars, values_bar):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", color="white", fontsize=10)

    st.pyplot(fig)
    plt.close(fig)

# ── Seasonal source distribution ─────────────────────────────────────────────
st.markdown("### 🌦️ Source Signatures by Season")

SEASON_ORDER = ["Winter", "Pre-Monsoon", "Monsoon", "Post-Monsoon"]
season_col = "season" if "season" in source_df.columns else None

if season_col:
    seasonal = source_df.groupby(season_col)[available_sources].mean().reindex(
        [s for s in SEASON_ORDER if s in source_df[season_col].values]
    )
    seasonal.columns = [SOURCE_LABELS[s] for s in available_sources]

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    x = np.arange(len(seasonal))
    width = 0.2
    for i, col in enumerate(seasonal.columns):
        color = SOURCE_COLORS.get(col, "#888")
        ax.bar(x + i * width, seasonal[col], width, label=col, color=color, alpha=0.85, edgecolor="white", linewidth=0.5)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(seasonal.index, fontsize=11)
    ax.set_ylabel("Mean Score", color="white", fontsize=11)
    ax.set_title("Source-Signature Scores by Season", color="white", fontsize=14)
    ax.legend(facecolor="#2a2a4e", edgecolor="white", labelcolor="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("white")
        spine.set_alpha(0.3)
    ax.grid(alpha=0.15, color="white", axis="y")

    st.pyplot(fig)
    plt.close(fig)

# ── Station-level source comparison ──────────────────────────────────────────
st.markdown("### 🏭 Source Scores by Station")

name_col = "name" if "name" in source_df.columns else None
if name_col:
    station_scores = source_df.groupby(name_col)[available_sources].mean()
    station_scores.columns = [SOURCE_LABELS[s] for s in available_sources]
    station_scores = station_scores.sort_values(station_scores.columns[0], ascending=True)

    fig, ax = plt.subplots(figsize=(12, max(4, len(station_scores) * 0.6)))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    station_scores.plot(kind="barh", ax=ax, color=[SOURCE_COLORS.get(c, "#888") for c in station_scores.columns],
                        alpha=0.85, edgecolor="white", linewidth=0.3)
    ax.set_xlabel("Mean Score", color="white", fontsize=11)
    ax.set_title("Source Signatures by Station", color="white", fontsize=14)
    ax.legend(facecolor="#2a2a4e", edgecolor="white", labelcolor="white", fontsize=9)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("white")
        spine.set_alpha(0.3)
    ax.grid(alpha=0.15, color="white", axis="x")

    st.pyplot(fig)
    plt.close(fig)

# ── Weather correlations ─────────────────────────────────────────────────────
if enriched_path.exists():
    st.markdown("### 🌡️ PM2.5 vs Weather Variables")

    edf = pd.read_parquet(enriched_path)
    weather_cols = {
        "temp_c": "Temperature (°C)",
        "rh_pct": "Humidity (%)",
        "wind_speed_kmh": "Wind Speed (km/h)",
        "temperature": "Temperature (°C)",
        "humidity": "Humidity (%)",
        "wind_speed": "Wind Speed",
    }
    available_weather = {k: v for k, v in weather_cols.items() if k in edf.columns and edf[k].notna().any()}

    if available_weather and "pm25" in edf.columns:
        n_plots = min(len(available_weather), 4)
        fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))
        fig.patch.set_facecolor("#1a1a2e")
        if n_plots == 1:
            axes = [axes]

        for ax, (col, label) in zip(axes, list(available_weather.items())[:n_plots]):
            ax.set_facecolor("#1a1a2e")
            sample = edf[[col, "pm25"]].dropna().sample(min(3000, len(edf)), random_state=42)
            ax.scatter(sample[col], sample["pm25"], alpha=0.15, s=8, color="#e74c3c")
            ax.set_xlabel(label, color="white", fontsize=10)
            ax.set_ylabel("PM2.5 (µg/m³)", color="white", fontsize=10)
            ax.tick_params(colors="white")
            for spine in ax.spines.values():
                spine.set_color("white")
                spine.set_alpha(0.3)
            ax.grid(alpha=0.15, color="white")

            # Correlation coefficient
            corr = sample[col].corr(sample["pm25"])
            ax.set_title(f"r = {corr:.3f}", color="white", fontsize=11)

        fig.suptitle("PM2.5 vs Weather Variables", color="white", fontsize=14, y=1.02)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

# ── Source scores data table ──────────────────────────────────────────────────
st.markdown("### 📋 Source Scores Data")
with st.expander("Show raw data"):
    display_df = source_df.copy()
    for s in available_sources:
        display_df[SOURCE_LABELS[s]] = display_df[s].round(4)
    cols_to_show = [c for c in [name_col, season_col, "n_readings"] + list(SOURCE_LABELS.values()) if c in display_df.columns]
    st.dataframe(display_df[cols_to_show], use_container_width=True, hide_index=True)
