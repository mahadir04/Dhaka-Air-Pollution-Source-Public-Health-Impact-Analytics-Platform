"""
Health Burden Estimation — CRF coefficient application.

Exact same logic as notebook Cell 6.1:
  LONG-TERM (annual means): Pope et al. β=0.008, India D-in-D β=0.0086
  SHORT-TERM (daily PM10):  Orellano 2020 β=0.004

Usage:
    python health_burden/apply_crf.py
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from pyspark.sql import functions as F
from utils.config import CLEANED_PARQUET_DIR, OUTPUTS_DIR
from utils.spark_session import get_spark


# ── CRF coefficients (same as notebook Cell 6.1) ─────────────────────────────
CRF_ANNUAL = {
    "pm25_long_term": {"pollutant": "pm25", "beta": 0.008,  "label": "PM2.5 Long-Term (Pope et al.)"},
    "pm25_annual":    {"pollutant": "pm25", "beta": 0.0086, "label": "PM2.5 Annual (India D-in-D 2024)"},
}
WHO_ANNUAL = {"pm25": 5.0, "pm10": 15.0}
DHAKA_POPULATION = 10_356_500      # BBS 2022 estimate
MORTALITY_RATE   = 6.1 / 1000      # Bangladesh (World Bank 2022)

CRF_DAILY = {"pollutant": "pm10", "beta": 0.004, "label": "PM10 Short-Term (Orellano 2020)"}
WHO_DAILY_PM10 = 45.0


def estimate_health_burden():
    spark = get_spark()

    print("=" * 60)
    print("  Health Burden Estimation")
    print("=" * 60 + "\n")

    sdf = spark.read.parquet(str(CLEANED_PARQUET_DIR))
    sdf = sdf.withColumn("station_id", F.col("station_id").cast("string"))

    name_col = "station_name" if "station_name" in sdf.columns else "station_id"

    # ── Annual means per station ──────────────────────────────────────────────
    station_annual = (
        sdf.groupBy("station_id", name_col, "latitude", "longitude")
        .agg(
            F.mean("pm25").alias("mean_pm25"),
            F.mean("pm10").alias("mean_pm10"),
            F.first("population_catchment").alias("catchment_pop"),
        )
    ).toPandas()

    # ── Long-term CRF (identical to notebook) ────────────────────────────────
    long_term_results = []
    for _, row in station_annual.iterrows():
        for crf_key, crf in CRF_ANNUAL.items():
            pol = crf["pollutant"]
            beta = crf["beta"]
            conc = row.get(f"mean_{pol}")
            if conc is None or (isinstance(conc, float) and math.isnan(conc)):
                continue
            baseline = WHO_ANNUAL.get(pol, 0.0)
            delta_c = max(conc - baseline, 0.0)
            ar_pct = (1 - math.exp(-beta * delta_c)) * 100
            catchment = float(row["catchment_pop"]) if pd.notna(row["catchment_pop"]) else DHAKA_POPULATION / 6
            attrib_deaths = catchment * MORTALITY_RATE * ar_pct / 100
            long_term_results.append({
                "station":         row[name_col],
                "lat":             row["latitude"],
                "lon":             row["longitude"],
                "crf":             crf_key,
                "crf_label":       crf["label"],
                "exposure_window": "Annual",
                "pollutant":       pol.upper(),
                "mean_conc":       round(conc, 2),
                "who_baseline":    baseline,
                "delta_c":         round(delta_c, 2),
                "ar_pct":          round(ar_pct, 3),
                "catchment_pop":   int(catchment),
                "attrib_deaths":   round(attrib_deaths, 1),
            })

    health_df = pd.DataFrame(long_term_results)
    health_df.to_csv(OUTPUTS_DIR / "health_burden_table.csv", index=False)
    print("[health] Long-term burden table saved → outputs/health_burden_table.csv")
    print("\n── LONG-TERM (annual means) ──────────────────────────────────────")
    if not health_df.empty:
        print(health_df.groupby("crf_label")[["ar_pct", "attrib_deaths"]].mean().round(3).to_string())

    # ── Short-term CRF (PM10 daily, identical to notebook) ────────────────────
    if "pm10" in sdf.columns:
        sdf_with_date = sdf.withColumn("date", F.to_date("timestamp"))
        daily_station_sdf = (
            sdf_with_date
            .filter(F.col("pm10").isNotNull() & F.col("population_catchment").isNotNull())
            .groupBy("station_id", name_col, "date")
            .agg(
                F.mean("pm10").alias("mean_pm10_daily"),
                F.first("population_catchment").alias("catchment_pop"),
            )
        )
        daily_pd = daily_station_sdf.toPandas()
        beta_short = CRF_DAILY["beta"]
        daily_pd["delta_c"] = (daily_pd["mean_pm10_daily"] - WHO_DAILY_PM10).clip(lower=0)
        daily_pd["ar_pct"] = (1 - np.exp(-beta_short * daily_pd["delta_c"])) * 100
        daily_pd["excess_deaths_day"] = (
            daily_pd["catchment_pop"] * MORTALITY_RATE / 365 * daily_pd["ar_pct"] / 100
        )
        short_term = (
            daily_pd.groupby(name_col)["excess_deaths_day"]
            .sum()
            .rename("annual_excess_deaths_short_term_pm10")
            .reset_index()
            .sort_values("annual_excess_deaths_short_term_pm10", ascending=False)
        )
        short_term.to_csv(OUTPUTS_DIR / "health_burden_short_term_pm10.csv", index=False)
        print("\n── SHORT-TERM (daily PM10, Orellano 2020) ──────────────────────")
        print(short_term.to_string(index=False))
    else:
        print("\n[health] PM10 not available — short-term CRF skipped.")

    print("\n⚠️  Short-term and long-term burdens answer DIFFERENT questions and must NOT be summed.")

    print(f"\n{'=' * 60}")
    print(f"  Health burden estimation complete.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    estimate_health_burden()
