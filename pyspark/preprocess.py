"""
Stage 1.5 — Data Cleaning & Quality Handling

Applies the full data-quality pipeline to the enriched dataset:
  1. Deduplication on (station_id, timestamp)
  2. Forward-fill within-station gaps ≤ 3 hours, else station-level median
  3. IQR-based outlier clipping per pollutant, per station

Outputs the final cleaned, unified-schema Parquet to data/processed/cleaned/.

Usage
-----
    python pyspark/preprocess.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pyspark.sql import functions as F
from pyspark.sql import Window
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql.types import FloatType
from utils.config import (
    POPULATION_PARQUET_DIR, CLEANED_PARQUET_DIR,
    POLLUTANTS, MAX_FFILL_GAP_HOURS, IQR_MULTIPLIER,
    WEATHER_RENAME,
)
from utils.spark_session import get_spark


# ─────────────────────────── 1. Deduplication ────────────────────────────────

def deduplicate(sdf: SparkDataFrame) -> SparkDataFrame:
    """Remove duplicate rows on (station_id, timestamp).

    When duplicates exist, keep the row with the most non-null pollutant values.
    """
    before = sdf.count()

    # Count non-null pollutant columns per row to pick the "best" duplicate
    pollutant_cols = [c for c in POLLUTANTS if c in sdf.columns]
    non_null_expr = sum(F.when(F.col(c).isNotNull(), 1).otherwise(0)
                        for c in pollutant_cols)

    sdf = sdf.withColumn("_nn_count", non_null_expr)

    w = (Window
         .partitionBy("station_id", "timestamp")
         .orderBy(F.desc("_nn_count")))

    sdf = (
        sdf
        .withColumn("_rn", F.row_number().over(w))
        .filter(F.col("_rn") == 1)
        .drop("_rn", "_nn_count")
    )

    after = sdf.count()
    removed = before - after
    print(f"[preprocess] Deduplication: {before:,} → {after:,} "
          f"({removed:,} duplicates removed)")
    return sdf


# ─────────────────────────── 2. Gap-filling ──────────────────────────────────

def fill_gaps(sdf: SparkDataFrame) -> SparkDataFrame:
    """Forward-fill within-station gaps ≤ MAX_FFILL_GAP_HOURS.

    For longer gaps, fall back to the station-level hourly median.
    """
    pollutant_cols = [c for c in POLLUTANTS if c in sdf.columns]
    weather_cols = [c for c in WEATHER_RENAME.values() if c in sdf.columns]
    fill_cols = pollutant_cols + weather_cols

    # Window ordered by timestamp within each station
    w_ordered = Window.partitionBy("station_id").orderBy("timestamp")

    # Step A: Detect gap sizes (hours since last non-null reading per column)
    for col in fill_cols:
        # Mark timestamps where this column has a real (non-null) value
        sdf = sdf.withColumn(
            f"_last_valid_{col}",
            F.last(
                F.when(F.col(col).isNotNull(), F.col("timestamp")),
                ignorenulls=True
            ).over(w_ordered)
        )
        # Gap in hours
        sdf = sdf.withColumn(
            f"_gap_h_{col}",
            (F.unix_timestamp("timestamp") -
             F.unix_timestamp(f"_last_valid_{col}")) / 3600.0
        )

    # Step B: Forward-fill only where gap ≤ threshold
    for col in fill_cols:
        last_val = F.last(F.col(col), ignorenulls=True).over(w_ordered)
        sdf = sdf.withColumn(
            col,
            F.when(
                F.col(col).isNotNull(), F.col(col)
            ).otherwise(
                F.when(
                    F.col(f"_gap_h_{col}") <= MAX_FFILL_GAP_HOURS,
                    last_val
                )
            )
        )

    # Materialize here — Steps A+B chained 2×len(fill_cols) window-function
    # columns onto sdf without ever triggering an action. Without cutting the
    # lineage now, each of Step C's per-column joins below (and the final
    # per-column .count() calls) would re-execute that entire chain from
    # scratch, compounding into a near-exponential slowdown.
    sdf = sdf.localCheckpoint(eager=True)

    # Step C: For remaining nulls, fill with station-level hourly median
    # (hour-of-day median per station)
    sdf = sdf.withColumn("_hour", F.hour("timestamp"))

    for col in fill_cols:
        station_hour_median = (
            sdf.filter(F.col(col).isNotNull())
            .groupBy("station_id", "_hour")
            .agg(F.expr(f"percentile_approx({col}, 0.5)").alias(f"_median_{col}"))
        )
        sdf = sdf.join(
            F.broadcast(station_hour_median),
            on=["station_id", "_hour"],
            how="left"
        )
        sdf = sdf.withColumn(
            col,
            F.coalesce(F.col(col), F.col(f"_median_{col}"))
        ).drop(f"_median_{col}")
        # Each join self-references the growing sdf lineage — checkpoint
        # after every column so the chain doesn't compound across the loop.
        sdf = sdf.localCheckpoint(eager=True)

    # Clean up temp columns
    temp_cols = [c for c in sdf.columns if c.startswith("_")]
    sdf = sdf.drop(*temp_cols)

    # Report remaining nulls
    for col in pollutant_cols:
        null_count = sdf.filter(F.col(col).isNull()).count()
        if null_count > 0:
            print(f"  [fill] {col}: {null_count:,} remaining nulls after fill")
        else:
            print(f"  [fill] {col}: ✓ fully filled")

    return sdf


# ─────────────────────────── 3. Outlier clipping ─────────────────────────────

def clip_outliers(sdf: SparkDataFrame) -> SparkDataFrame:
    """IQR-based outlier clipping per pollutant, per station.

    Values beyond Q1 - k*IQR or Q3 + k*IQR are capped to those bounds.
    k = IQR_MULTIPLIER (default 1.5).
    """
    pollutant_cols = [c for c in POLLUTANTS if c in sdf.columns]

    for col in pollutant_cols:
        # Compute Q1, Q3 per station
        bounds = (
            sdf.filter(F.col(col).isNotNull())
            .groupBy("station_id")
            .agg(
                F.expr(f"percentile_approx({col}, 0.25)").alias("_q1"),
                F.expr(f"percentile_approx({col}, 0.75)").alias("_q3"),
            )
            .withColumn("_iqr", F.col("_q3") - F.col("_q1"))
            .withColumn("_lower", F.col("_q1") - IQR_MULTIPLIER * F.col("_iqr"))
            .withColumn("_upper", F.col("_q3") + IQR_MULTIPLIER * F.col("_iqr"))
        )

        sdf = sdf.join(F.broadcast(bounds.select("station_id", "_lower", "_upper")),
                        on="station_id", how="left")

        # Clip
        clipped_count_before = sdf.filter(
            (F.col(col) < F.col("_lower")) | (F.col(col) > F.col("_upper"))
        ).count()

        sdf = sdf.withColumn(
            col,
            F.when(F.col(col) < F.col("_lower"), F.col("_lower"))
            .when(F.col(col) > F.col("_upper"), F.col("_upper"))
            .otherwise(F.col(col))
        ).drop("_lower", "_upper")

        if clipped_count_before > 0:
            print(f"  [clip] {col}: {clipped_count_before:,} values clipped")
        else:
            print(f"  [clip] {col}: no outliers detected")

        # Checkpoint after each column — each iteration joins onto sdf, and
        # without cutting the lineage here the chain (and its per-column
        # .count()) recompiles from scratch every time it grows.
        sdf = sdf.localCheckpoint(eager=True)

    return sdf


# ─────────────────────────── 4. Schema enforcement ───────────────────────────

def enforce_schema(sdf: SparkDataFrame) -> SparkDataFrame:
    """Ensure the output matches the documented unified schema."""
    # Core columns in order
    schema_cols = [
        "station_id", "station_name", "latitude", "longitude", "timestamp",
    ]
    # Pollutants
    for p in POLLUTANTS:
        if p in sdf.columns:
            schema_cols.append(p)

    # Weather
    for w in WEATHER_RENAME.values():
        if w in sdf.columns:
            schema_cols.append(w)

    # Population
    if "population_catchment" in sdf.columns:
        schema_cols.append("population_catchment")

    # Date partitioning columns
    if "date" in sdf.columns:
        schema_cols.append("date")
    if "year" in sdf.columns:
        schema_cols.append("year")
    if "month" in sdf.columns:
        schema_cols.append("month")

    # Select only schema columns (drop any temp/extra cols)
    available = [c for c in schema_cols if c in sdf.columns]
    sdf = sdf.select(*available)

    # Cast pollutants to float
    for p in POLLUTANTS:
        if p in sdf.columns:
            sdf = sdf.withColumn(p, F.col(p).cast(FloatType()))

    return sdf


# ─────────────────────────── Main pipeline ───────────────────────────────────

def main():
    print(f"{'=' * 60}")
    print(f"  Data Cleaning & Quality Handling")
    print(f"{'=' * 60}\n")

    spark = get_spark()

    # Load the enriched dataset
    sdf = spark.read.parquet(str(POPULATION_PARQUET_DIR))
    sdf = sdf.withColumn("station_id", F.col("station_id").cast("string"))
    print(f"[preprocess] Loaded: {sdf.count():,} rows, "
          f"{len(sdf.columns)} columns\n")

    # 1. Deduplicate
    print("─── Step 1: Deduplication ───")
    sdf = deduplicate(sdf)

    # 2. Fill gaps
    print("\n─── Step 2: Gap-filling ───")
    sdf = fill_gaps(sdf)

    # 3. Clip outliers
    print("\n─── Step 3: Outlier clipping ───")
    sdf = clip_outliers(sdf)

    # 4. Enforce schema
    print("\n─── Step 4: Schema enforcement ───")
    sdf = enforce_schema(sdf)

    # Add partition columns if missing
    if "date" not in sdf.columns:
        sdf = sdf.withColumn("date", F.to_date("timestamp"))
    if "year" not in sdf.columns:
        sdf = sdf.withColumn("year", F.year("date"))
    if "month" not in sdf.columns:
        sdf = sdf.withColumn("month", F.month("date"))

    # Force station_id to string — Spark's partition-column type inference
    # can otherwise treat an all-numeric station_id as a double on re-read.
    sdf = sdf.withColumn("station_id", F.col("station_id").cast("string"))

    # Save cleaned data
    out_path = str(CLEANED_PARQUET_DIR)
    sdf.write.mode("overwrite").partitionBy("station_id", "year", "month").parquet(out_path)
    print(f"\n[preprocess] ✓ Cleaned data saved → {out_path}")
    print(f"[preprocess]   Final rows: {sdf.count():,}")
    print(f"[preprocess]   Columns: {sdf.columns}")

    print(f"\n{'=' * 60}")
    print(f"  Preprocessing complete.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
