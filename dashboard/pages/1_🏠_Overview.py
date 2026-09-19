"""
Page 1 — Overview
KPIs, station map, PM2.5 trends, data completeness.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = PROJECT_ROOT / "figures"

st.set_page_config(page_title="Overview — AirImpact Dhaka", page_icon="🏠", layout="wide")
st.markdown("# 🏠 Overview")
st.markdown("---")

# ── Load data ─────────────────────────────────────────────────────────────────
parquet_path = OUTPUTS_DIR / "enriched_data.parquet"
if not parquet_path.exists():
    st.error(
        "**enriched_data.parquet not found.** "
        "Run the notebook (Sections 1–9) first to generate the data exports."
    )
    st.stop()

df = pd.read_parquet(parquet_path)

# Parse timestamps
for col in ["ts", "timestamp", "date"]:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")

ts_col = "ts" if "ts" in df.columns else "timestamp" if "timestamp" in df.columns else None
name_col = "name" if "name" in df.columns else "station_name" if "station_name" in df.columns else None

# ── KPI Cards ─────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)

n_stations = df[name_col].nunique() if name_col else "N/A"
pm25_mean = df["pm25"].mean() if "pm25" in df.columns else None
pm25_max = df["pm25"].max() if "pm25" in df.columns else None
who_24h = 15.0  # WHO 2021 24-hour guideline
who_annual = 5.0
who_exceed_pct = (df["pm25"] > who_24h).mean() * 100 if "pm25" in df.columns else None

c1.metric("🏭 Stations", n_stations)
c2.metric("📊 Mean PM2.5", f"{pm25_mean:.1f} µg/m³" if pm25_mean else "N/A")
c3.metric("🔴 Max PM2.5", f"{pm25_max:.0f} µg/m³" if pm25_max else "N/A")
c4.metric("⚠️ WHO Exceedance", f"{who_exceed_pct:.1f}%" if who_exceed_pct else "N/A")

st.markdown("---")

# ── Station Location Map (static scatter) ────────────────────────────────────
lat_col = "lat" if "lat" in df.columns else "latitude"
lon_col = "lon" if "lon" in df.columns else "longitude"

if lat_col in df.columns and lon_col in df.columns and name_col:
    st.markdown("### 📍 Station Locations & Mean PM2.5")

    station_summary = (
        df.groupby(name_col)
        .agg({
            lat_col: "first",
            lon_col: "first",
            "pm25": "mean",
        })
        .dropna(subset=["pm25"])
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    scatter = ax.scatter(
        station_summary[lon_col],
        station_summary[lat_col],
        c=station_summary["pm25"],
        s=station_summary["pm25"] * 4,
        cmap="RdYlGn_r",
        edgecolors="white",
        linewidth=0.8,
        alpha=0.9,
        vmin=who_annual,
        vmax=station_summary["pm25"].max(),
    )

    for _, row in station_summary.iterrows():
        label = row[name_col]
        if len(label) > 20:
            label = label[:18] + "…"
        ax.annotate(
            label,
            (row[lon_col], row[lat_col]),
            textcoords="offset points",
            xytext=(8, 5),
            fontsize=7,
            color="white",
            alpha=0.8,
        )

    cbar = fig.colorbar(scatter, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label("Mean PM2.5 (µg/m³)", color="white", fontsize=10)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")

    ax.set_xlabel("Longitude", color="white", fontsize=11)
    ax.set_ylabel("Latitude", color="white", fontsize=11)
    ax.set_title("Dhaka Monitoring Stations — PM2.5 Concentration", color="white", fontsize=14)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("white")
        spine.set_alpha(0.3)
    ax.grid(alpha=0.15, color="white")

    st.pyplot(fig)
    plt.close(fig)

# ── PM2.5 Daily Trend ────────────────────────────────────────────────────────
if "pm25" in df.columns and ts_col:
    st.markdown("### 📈 PM2.5 Daily Trend with 7-Day Rolling Average")

    daily = df.groupby(df[ts_col].dt.date)["pm25"].mean().reset_index()
    daily.columns = ["date", "pm25"]
    daily["date"] = pd.to_datetime(daily["date"])
    daily["rolling_7d"] = daily["pm25"].rolling(7, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    ax.fill_between(daily["date"], daily["pm25"], alpha=0.15, color="#e74c3c")
    ax.plot(daily["date"], daily["pm25"], color="#e74c3c", alpha=0.5, linewidth=0.8, label="Daily mean")
    ax.plot(daily["date"], daily["rolling_7d"], color="#f39c12", linewidth=2.5, label="7-day rolling avg")
    ax.axhline(who_24h, color="#2ecc71", linestyle="--", linewidth=1.2, label=f"WHO 24h = {who_24h} µg/m³")
    ax.axhline(who_annual, color="#3498db", linestyle=":", linewidth=1.2, label=f"WHO annual = {who_annual} µg/m³")

    ax.set_xlabel("Date", color="white", fontsize=11)
    ax.set_ylabel("PM2.5 (µg/m³)", color="white", fontsize=11)
    ax.set_title("Daily Average PM2.5 — All Stations", color="white", fontsize=14)
    ax.legend(facecolor="#2a2a4e", edgecolor="white", labelcolor="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("white")
        spine.set_alpha(0.3)
    ax.grid(alpha=0.15, color="white")

    st.pyplot(fig)
    plt.close(fig)

# ── Diurnal Cycle ─────────────────────────────────────────────────────────────
hour_col = "hour" if "hour" in df.columns else None
if hour_col is None and ts_col:
    df["hour"] = df[ts_col].dt.hour
    hour_col = "hour"

if "pm25" in df.columns and hour_col:
    st.markdown("### 🕐 Diurnal PM2.5 Cycle")

    col_left, col_right = st.columns(2)

    with col_left:
        hourly = df.groupby(hour_col)["pm25"].agg(["mean", "std"]).reset_index()

        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor("#1a1a2e")
        ax.set_facecolor("#1a1a2e")

        ax.plot(hourly[hour_col], hourly["mean"], color="#e74c3c", linewidth=2.5, marker="o", markersize=5)
        ax.fill_between(
            hourly[hour_col],
            hourly["mean"] - hourly["std"],
            hourly["mean"] + hourly["std"],
            alpha=0.15, color="#e74c3c"
        )
        ax.axhline(who_24h, color="#2ecc71", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_xlabel("Hour of Day (UTC)", color="white")
        ax.set_ylabel("PM2.5 (µg/m³)", color="white")
        ax.set_title("Hourly PM2.5 Pattern", color="white", fontsize=13)
        ax.set_xticks(range(0, 24, 3))
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("white")
            spine.set_alpha(0.3)
        ax.grid(alpha=0.15, color="white")

        st.pyplot(fig)
        plt.close(fig)

    with col_right:
        # Seasonal boxplot
        season_col = "season" if "season" in df.columns else None
        if season_col and "pm25" in df.columns:
            season_order = ["Winter", "Pre-Monsoon", "Monsoon", "Post-Monsoon"]
            plot_data = df[df[season_col].isin(season_order)].copy()

            fig, ax = plt.subplots(figsize=(8, 5))
            fig.patch.set_facecolor("#1a1a2e")
            ax.set_facecolor("#1a1a2e")

            colors = ["#3498db", "#e67e22", "#2ecc71", "#9b59b6"]
            bp = ax.boxplot(
                [plot_data[plot_data[season_col] == s]["pm25"].dropna() for s in season_order if s in plot_data[season_col].values],
                labels=[s for s in season_order if s in plot_data[season_col].values],
                patch_artist=True,
                showfliers=False,
            )
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
            for element in ["whiskers", "caps", "medians"]:
                plt.setp(bp[element], color="white", alpha=0.7)

            ax.axhline(who_24h, color="#2ecc71", linestyle="--", linewidth=1, alpha=0.7)
            ax.set_ylabel("PM2.5 (µg/m³)", color="white")
            ax.set_title("Seasonal PM2.5 Distribution", color="white", fontsize=13)
            ax.tick_params(colors="white")
            for spine in ax.spines.values():
                spine.set_color("white")
                spine.set_alpha(0.3)
            ax.grid(alpha=0.15, color="white", axis="y")

            st.pyplot(fig)
            plt.close(fig)

# ── Data Completeness Table ──────────────────────────────────────────────────
st.markdown("### 📋 Data Completeness by Station")

if name_col and ts_col:
    completeness = []
    for station, grp in df.groupby(name_col):
        row = {
            "Station": station,
            "Records": len(grp),
            "Date From": str(grp[ts_col].min().date()) if pd.notna(grp[ts_col].min()) else "N/A",
            "Date To": str(grp[ts_col].max().date()) if pd.notna(grp[ts_col].max()) else "N/A",
            "PM2.5 Coverage": f"{grp['pm25'].notna().mean()*100:.1f}%" if "pm25" in grp.columns else "N/A",
            "Mean PM2.5": f"{grp['pm25'].mean():.1f}" if "pm25" in grp.columns else "N/A",
        }
        if "catchment_pop" in grp.columns:
            pop = grp["catchment_pop"].iloc[0]
            row["Population"] = f"{int(pop):,}" if pd.notna(pop) else "N/A"
        completeness.append(row)

    comp_df = pd.DataFrame(completeness)
    st.dataframe(comp_df, hide_index=True)
