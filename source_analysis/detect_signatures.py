"""
Source-Signature Detection — Spark SQL rule-based scoring.

Exact same logic as notebook Cell 5.1:
  - Traffic:      rush-hour NO₂/PM10, weekday-heavy
  - Brick Kilns:  dry season SO₂/PM10, morning 06-12
  - Biomass:      post-monsoon CO/PM2.5, evening
  - Construction: high PM10/PM2.5 ratio, daytime

Usage:
    python source_analysis/detect_signatures.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from pyspark.sql import functions as F
from utils.config import CLEANED_PARQUET_DIR, OUTPUTS_DIR, FIGURES_DIR
from utils.spark_session import get_spark

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


# ── Season / time feature helpers (same as notebook) ─────────────────────────

def add_time_features(sdf):
    """Add hour, dow, month, season, date columns — mirrors notebook enrichment."""
    if "hour" not in sdf.columns:
        sdf = sdf.withColumn("hour", F.hour("timestamp"))
    if "dow" not in sdf.columns:
        sdf = sdf.withColumn("dow", F.dayofweek("timestamp"))
    if "month" not in sdf.columns:
        sdf = sdf.withColumn("month", F.month("timestamp"))
    if "date" not in sdf.columns:
        sdf = sdf.withColumn("date", F.to_date("timestamp"))
    if "season" not in sdf.columns:
        sdf = sdf.withColumn("season",
            F.when(F.col("month").isin(12, 1, 2), "Winter")
             .when(F.col("month").isin(3, 4, 5), "Pre-Monsoon")
             .when(F.col("month").isin(6, 7, 8, 9), "Monsoon")
             .otherwise("Post-Monsoon")
        )
    return sdf


# ── Source-signature Spark SQL (identical to notebook Cell 5.1) ───────────────

SOURCE_SQL = """
SELECT *,

  -- TRAFFIC SIGNATURE
  CASE
    WHEN (hour BETWEEN 8 AND 10 OR hour BETWEEN 17 AND 20)
         AND dow BETWEEN 2 AND 6
         AND no2  IS NOT NULL AND no2  > 30
         AND pm10 IS NOT NULL AND pm10 > 60
    THEN LEAST(1.0,
           0.30 * LEAST((no2  / 80.0), 1.0) +
           0.25 * LEAST((pm10 / 120.0), 1.0) +
           0.25 * (CASE WHEN hour BETWEEN 8 AND 10 OR hour BETWEEN 17 AND 20
                        THEN 1.0 ELSE 0.0 END) +
           0.20 * (CASE WHEN dow BETWEEN 2 AND 6 THEN 1.0 ELSE 0.5 END))
    ELSE
      CASE WHEN (hour BETWEEN 8 AND 10 OR hour BETWEEN 17 AND 20) AND dow BETWEEN 2 AND 6
           THEN 0.3
           ELSE 0.0 END
  END AS score_traffic,

  -- BRICK KILN SIGNATURE
  CASE
    WHEN month IN (11, 12, 1, 2, 3, 4)
         AND hour BETWEEN 6 AND 12
         AND so2  IS NOT NULL AND so2  > 15
         AND pm10 IS NOT NULL AND pm10 > 80
    THEN LEAST(1.0,
           0.35 * LEAST((so2  / 50.0),  1.0) +
           0.30 * LEAST((pm10 / 150.0), 1.0) +
           0.20 * (CASE WHEN month IN (12,1,2) THEN 1.0
                        WHEN month IN (11,3)   THEN 0.7
                        ELSE 0.4 END) +
           0.15 * (CASE WHEN hour BETWEEN 7 AND 11 THEN 1.0 ELSE 0.5 END))
    ELSE
      CASE WHEN month IN (11,12,1,2,3,4) AND hour BETWEEN 6 AND 12
           THEN 0.2
           ELSE 0.0 END
  END AS score_brick_kiln,

  -- BIOMASS BURNING SIGNATURE
  CASE
    WHEN month IN (10, 11)
         AND hour BETWEEN 17 AND 22
         AND co   IS NOT NULL AND co   > 1.0
         AND pm25 IS NOT NULL AND pm25 > 50
    THEN LEAST(1.0,
           0.40 * LEAST((co   / 5.0),   1.0) +
           0.35 * LEAST((pm25 / 120.0), 1.0) +
           0.25 * (CASE WHEN month = 11 THEN 1.0 ELSE 0.7 END))
    ELSE
      CASE WHEN month IN (10,11) AND hour BETWEEN 17 AND 22
           THEN 0.15
           ELSE 0.0 END
  END AS score_biomass,

  -- CONSTRUCTION DUST SIGNATURE
  CASE
    WHEN pm25 IS NOT NULL AND pm25 > 5
         AND pm10 IS NOT NULL
         AND (pm10 / NULLIF(pm25, 0)) > 2.5
         AND hour BETWEEN 7 AND 18
    THEN LEAST(1.0,
           0.50 * LEAST(((pm10 / NULLIF(pm25, 0)) - 2.5) / 4.5, 1.0) +
           0.30 * LEAST((pm10 / 200.0), 1.0) +
           0.20 * (CASE WHEN hour BETWEEN 9 AND 16 THEN 1.0 ELSE 0.5 END))
    ELSE
      CASE WHEN pm10 IS NOT NULL AND pm25 IS NOT NULL AND pm25 > 0
                AND (pm10 / pm25) > 2.5
           THEN 0.10
           ELSE 0.0 END
  END AS score_construction

FROM aq
"""


SOURCES = ["score_traffic", "score_brick_kiln", "score_biomass", "score_construction"]
SOURCE_LABELS = {
    "score_traffic":      "Traffic",
    "score_brick_kiln":   "Brick Kilns",
    "score_biomass":      "Biomass Burning",
    "score_construction": "Construction Dust",
}
SEASON_ORDER = ["Winter", "Pre-Monsoon", "Monsoon", "Post-Monsoon"]


def run_source_analysis():
    spark = get_spark()

    print("=" * 60)
    print("  Source-Signature Analysis")
    print("=" * 60 + "\n")

    # Load cleaned data
    sdf = spark.read.parquet(str(CLEANED_PARQUET_DIR))
    sdf = sdf.withColumn("station_id", F.col("station_id").cast("string"))
    sdf = add_time_features(sdf)

    # Use station_name as 'name' if present
    if "station_name" in sdf.columns and "name" not in sdf.columns:
        sdf = sdf.withColumnRenamed("station_name", "name")

    print(f"[source] Loaded: {sdf.count():,} rows")

    # Register as temp view and run SQL
    sdf.createOrReplaceTempView("aq")
    scored = spark.sql(SOURCE_SQL)
    scored.cache()
    print(f"[source] Source-signature scores computed: {scored.count():,} rows")

    # Aggregate by station × season (Cell 5.2)
    name_col = "name" if "name" in scored.columns else "station_id"
    agg_sdf = (
        scored
        .groupBy(name_col, "season")
        .agg(
            F.mean("score_traffic").alias("score_traffic"),
            F.mean("score_brick_kiln").alias("score_brick_kiln"),
            F.mean("score_biomass").alias("score_biomass"),
            F.mean("score_construction").alias("score_construction"),
            F.count("*").alias("n_readings"),
        )
        .orderBy(name_col, "season")
    )

    agg_pd = agg_sdf.toPandas()
    agg_pd.to_csv(OUTPUTS_DIR / "5_source_scores_by_station_season.csv", index=False)
    print(f"[source] Saved → outputs/5_source_scores_by_station_season.csv")
    print(agg_pd.to_string(index=False))

    # ── Heatmap (Cell 5.3) ───────────────────────────────────────────────────
    season_source = (
        agg_pd
        .groupby("season")[SOURCES]
        .mean()
        .reindex([s for s in SEASON_ORDER if s in agg_pd["season"].values])
        .rename(columns=SOURCE_LABELS)
    )

    fig, ax = plt.subplots(figsize=(9, 4))
    sns.heatmap(
        season_source.T,
        annot=True, fmt=".3f",
        cmap="YlOrRd", vmin=0, vmax=0.6,
        linewidths=0.5, ax=ax,
        cbar_kws={"label": "Mean Source Score (0–1)"},
    )
    ax.set_title("Emission Source-Signature Scores by Season — Dhaka",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Season")
    ax.set_ylabel("Emission Source")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "source_signature_heatmap.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[source] Heatmap saved → figures/source_signature_heatmap.png")

    # Save enriched data with scores as Parquet for downstream
    scored.write.mode("overwrite").parquet(str(OUTPUTS_DIR / "source_scored.parquet"))
    print(f"[source] Scored Parquet saved → outputs/source_scored.parquet")

    return scored


if __name__ == "__main__":
    run_source_analysis()
