"""
Comparative Health-Impact Ranking.

Exact same logic as notebook Cell 7.1:
  health_impact_score = mean_pm25 × catchment_pop × beta / 1e6

Usage:
    python ranking/rank_health_burden.py
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from pyspark.sql import functions as F
from utils.config import CLEANED_PARQUET_DIR, OUTPUTS_DIR
from utils.spark_session import get_spark

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# CRF for ranking (India D-in-D — same as notebook)
BETA_PM25 = 0.0086
WHO_PM25  = 5.0


def rank_health_burden():
    spark = get_spark()

    print("=" * 60)
    print("  Comparative Health-Impact Ranking")
    print("=" * 60 + "\n")

    sdf = spark.read.parquet(str(CLEANED_PARQUET_DIR))
    sdf = sdf.withColumn("station_id", F.col("station_id").cast("string"))

    # Add season
    if "season" not in sdf.columns:
        sdf = sdf.withColumn("month", F.month("timestamp"))
        sdf = sdf.withColumn("season",
            F.when(F.col("month").isin(12, 1, 2), "Winter")
             .when(F.col("month").isin(3, 4, 5), "Pre-Monsoon")
             .when(F.col("month").isin(6, 7, 8, 9), "Monsoon")
             .otherwise("Post-Monsoon")
        )

    name_col = "station_name" if "station_name" in sdf.columns else "station_id"

    # Station × season aggregation (identical to notebook Cell 7.1)
    ranking_sdf = (
        sdf
        .filter(F.col("pm25").isNotNull() & F.col("population_catchment").isNotNull())
        .groupBy(name_col, "season", "latitude", "longitude")
        .agg(
            F.mean("pm25").alias("mean_pm25"),
            F.mean("pm10").alias("mean_pm10"),
            F.first("population_catchment").alias("catchment_pop"),
            F.count("*").alias("n"),
        )
    )

    ranking_pd = ranking_sdf.toPandas()
    ranking_pd.rename(columns={name_col: "name"}, inplace=True)

    ranking_pd["delta_c"] = (ranking_pd["mean_pm25"] - WHO_PM25).clip(lower=0)
    ranking_pd["ar_pct"] = ranking_pd["delta_c"].apply(
        lambda dc: round((1 - math.exp(-BETA_PM25 * dc)) * 100, 3)
    )
    ranking_pd["health_impact_score"] = (
        ranking_pd["mean_pm25"] * ranking_pd["catchment_pop"] * BETA_PM25 / 1e6
    ).round(4)
    ranking_pd["rank"] = ranking_pd["health_impact_score"].rank(ascending=False, method="min").astype(int)
    ranking_pd = ranking_pd.sort_values("health_impact_score", ascending=False).reset_index(drop=True)

    ranking_pd.to_csv(OUTPUTS_DIR / "health_ranking_summary.csv", index=False)
    print("[ranking] Saved → outputs/health_ranking_summary.csv")
    print(ranking_pd[["rank", "name", "season", "mean_pm25", "ar_pct",
                       "catchment_pop", "health_impact_score"]].head(15).to_string(index=False))

    # ── Ranking visualization ─────────────────────────────────────────────────
    station_rank = (
        ranking_pd.groupby("name")
        .agg({"health_impact_score": "sum", "mean_pm25": "mean"})
        .sort_values("health_impact_score", ascending=True)
        .reset_index()
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, max(4, len(station_rank) * 0.5)))

    # Left: health-impact ranking
    ax = axes[0]
    colors = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(station_rank)))
    ax.barh(station_rank["name"], station_rank["health_impact_score"], color=colors)
    ax.set_xlabel("Health Impact Score")
    ax.set_title("Population-Weighted Health Impact Ranking")
    ax.grid(alpha=0.3, axis="x")

    # Right: raw PM2.5 ranking
    ax = axes[1]
    raw_rank = station_rank.sort_values("mean_pm25", ascending=True)
    colors_raw = plt.cm.Reds(np.linspace(0.3, 0.9, len(raw_rank)))
    ax.barh(raw_rank["name"], raw_rank["mean_pm25"], color=colors_raw)
    ax.axvline(WHO_PM25, color="green", linestyle="--", alpha=0.7, label="WHO guideline")
    ax.set_xlabel("Mean PM2.5 (µg/m³)")
    ax.set_title("Raw PM2.5 Ranking (for comparison)")
    ax.legend()
    ax.grid(alpha=0.3, axis="x")

    fig.suptitle("Health-Impact vs Raw-Pollution Ranking — Dhaka Stations", fontsize=14, y=1.02)
    fig.tight_layout()
    from utils.config import FIGURES_DIR
    fig.savefig(FIGURES_DIR / "health_impact_ranking.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[ranking] Plot saved → figures/health_impact_ranking.png")

    print(f"\n{'=' * 60}")
    print(f"  Ranking complete.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    rank_health_burden()
