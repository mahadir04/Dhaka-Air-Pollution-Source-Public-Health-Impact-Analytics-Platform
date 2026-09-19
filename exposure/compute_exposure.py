"""
Population-Weighted Exposure Calculation.

Matches notebook logic:
  exposure_score = pm25_concentration × population_catchment

Usage:
    python exposure/compute_exposure.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from pyspark.sql import functions as F
from utils.config import CLEANED_PARQUET_DIR, OUTPUTS_DIR
from utils.spark_session import get_spark


def compute_exposure():
    spark = get_spark()

    print("=" * 60)
    print("  Population-Weighted Exposure Calculation")
    print("=" * 60 + "\n")

    sdf = spark.read.parquet(str(CLEANED_PARQUET_DIR))
    sdf = sdf.withColumn("station_id", F.col("station_id").cast("string"))
    print(f"[exposure] Loaded: {sdf.count():,} rows")

    # Ensure required columns
    if "pm25" not in sdf.columns:
        print("[exposure] ERROR: pm25 column not found.")
        return
    if "population_catchment" not in sdf.columns:
        print("[exposure] WARNING: population_catchment not found — using default 300,000")
        sdf = sdf.withColumn("population_catchment", F.lit(300_000))

    # Compute exposure score
    sdf = sdf.withColumn(
        "exposure_score",
        F.col("pm25") * F.col("population_catchment")
    )

    # Add time features
    if "month" not in sdf.columns:
        sdf = sdf.withColumn("month", F.month("timestamp"))
    if "hour" not in sdf.columns:
        sdf = sdf.withColumn("hour", F.hour("timestamp"))

    name_col = "station_name" if "station_name" in sdf.columns else "station_id"

    # Aggregate by station
    station_exp = (
        sdf.filter(F.col("pm25").isNotNull())
        .groupBy("station_id", name_col, "latitude", "longitude")
        .agg(
            F.mean("pm25").alias("mean_pm25"),
            F.mean("exposure_score").alias("mean_exposure_score"),
            F.first("population_catchment").alias("population_catchment"),
            F.count("*").alias("n_readings"),
        )
        .orderBy(F.desc("mean_exposure_score"))
    )

    exp_pd = station_exp.toPandas()
    exp_pd.to_csv(OUTPUTS_DIR / "exposure_by_station.csv", index=False)
    print(f"[exposure] Saved → outputs/exposure_by_station.csv")
    print(exp_pd.to_string(index=False))

    # Aggregate by station × month
    monthly_exp = (
        sdf.filter(F.col("pm25").isNotNull())
        .groupBy("station_id", name_col, "month")
        .agg(
            F.mean("pm25").alias("mean_pm25"),
            F.mean("exposure_score").alias("mean_exposure_score"),
            F.first("population_catchment").alias("population_catchment"),
        )
        .orderBy("station_id", "month")
    )

    monthly_pd = monthly_exp.toPandas()
    monthly_pd.to_csv(OUTPUTS_DIR / "exposure_by_station_month.csv", index=False)
    print(f"[exposure] Monthly saved → outputs/exposure_by_station_month.csv")

    # Validate against known high-density areas
    print("\n[exposure] Top 5 stations by exposure score:")
    for _, row in exp_pd.head(5).iterrows():
        print(f"  {row.get(name_col, row['station_id'])}: "
              f"PM2.5={row['mean_pm25']:.1f} × pop={int(row['population_catchment']):,} "
              f"= exposure {row['mean_exposure_score']:,.0f}")

    print(f"\n{'=' * 60}")
    print(f"  Exposure calculation complete.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    compute_exposure()
