"""
Stage 1.3 — Weather Enrichment

Pulls historical hourly weather data from the Open-Meteo API for each
monitoring station's coordinates, then joins it to the OpenAQ pollution
data by timestamp.

Usage
-----
    python pyspark/weather_join.py
"""

import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pyspark.sql import functions as F
from pyspark.sql import DataFrame as SparkDataFrame
from utils.config import (
    OPENAQ_PARQUET_DIR, WEATHER_PARQUET_DIR,
    WEATHER_FEATURES, WEATHER_RENAME,
)
from utils.spark_session import get_spark


# ─────────────────────────── Open-Meteo fetch ────────────────────────────────

OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"


def fetch_weather_for_station(lat: float, lon: float,
                               date_from: str, date_to: str) -> pd.DataFrame:
    """Pull hourly weather from Open-Meteo Historical Weather API.

    Parameters
    ----------
    lat, lon : float       Station coordinates
    date_from, date_to : str   YYYY-MM-DD

    Returns
    -------
    pd.DataFrame with columns: timestamp, temperature, humidity,
                                wind_speed, wind_direction, pressure
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": date_from,
        "end_date": date_to,
        "hourly": ",".join(WEATHER_FEATURES),
        "timezone": "UTC",
    }

    data = None
    backoffs = [5, 15, 30]
    for attempt, delay in enumerate([0] + backoffs):
        if delay:
            print(f"  [weather] Rate-limited ({lat:.4f}, {lon:.4f}) — "
                  f"retrying in {delay}s …")
            time.sleep(delay)
        try:
            resp = requests.get(OPEN_METEO_URL, params=params, timeout=60)
            if resp.status_code == 429:
                continue
            resp.raise_for_status()
            data = resp.json()
            break
        except requests.RequestException as exc:
            print(f"  [weather] Open-Meteo request failed ({lat}, {lon}): {exc}")
            return pd.DataFrame()

    if data is None:
        print(f"  [weather] Gave up after repeated 429s ({lat:.4f}, {lon:.4f})")
        return pd.DataFrame()

    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    if not times:
        print(f"  [weather] No hourly data returned for ({lat:.4f}, {lon:.4f})")
        return pd.DataFrame()

    df = pd.DataFrame({"timestamp": pd.to_datetime(times, utc=True)})
    for meteo_col, our_col in WEATHER_RENAME.items():
        df[our_col] = hourly.get(meteo_col, [None] * len(times))

    return df


# ─────────────────────────── Core pipeline ───────────────────────────────────

def load_openaq_parquet() -> SparkDataFrame:
    """Load the ingested OpenAQ wide-format Parquet."""
    spark = get_spark()
    path = str(OPENAQ_PARQUET_DIR)
    sdf = spark.read.parquet(path)
    # Partitioned-Parquet partition-column type inference can turn a purely
    # numeric station_id into DoubleType on read (e.g. "6240023" -> 6240023.0);
    # round-trip through long to strip any spurious ".0" before it's used as
    # a join key or string identifier anywhere downstream.
    sdf = sdf.withColumn("station_id", F.col("station_id").cast("long").cast("string"))
    print(f"[weather_join] Loaded OpenAQ data: {sdf.count():,} rows, "
          f"{sdf.select('station_id').distinct().count()} stations")
    return sdf


def enrich_with_weather(sdf: SparkDataFrame) -> SparkDataFrame:
    """Fetch weather for each unique station (over that station's own
    observed date range) and join by timestamp.

    Uses a broadcast join since weather data per station is relatively small.
    """
    spark = get_spark()

    # Per-station coordinates + observed date range (stations can have very
    # different reporting histories — fetching each one's own range keeps
    # requests small instead of pulling every station over the widest span).
    stations = (
        sdf.groupBy("station_id")
        .agg(
            F.first("latitude").alias("latitude"),
            F.first("longitude").alias("longitude"),
            F.min("timestamp").alias("min_ts"),
            F.max("timestamp").alias("max_ts"),
        )
        .toPandas()
    )

    all_weather_dfs = []
    for _, row in stations.iterrows():
        sid = row["station_id"]
        lat = row["latitude"]
        lon = row["longitude"]
        date_from = str(row["min_ts"].date())
        date_to = str(row["max_ts"].date())
        print(f"[weather_join] Fetching weather for station {sid} "
              f"({lat:.4f}, {lon:.4f}) — {date_from} → {date_to} …")

        wdf = fetch_weather_for_station(lat, lon, date_from, date_to)
        if wdf.empty:
            print(f"  → No weather data, station will have NULLs.")
            continue

        wdf["station_id"] = str(sid)
        all_weather_dfs.append(wdf)
        time.sleep(1.0)  # politeness — avoid Open-Meteo rate limiting

    if not all_weather_dfs:
        print("[weather_join] WARNING: No weather data for any station!")
        # Add NULL weather columns to preserve schema
        for col in WEATHER_RENAME.values():
            sdf = sdf.withColumn(col, F.lit(None).cast("float"))
        return sdf

    # Combine all station weather into one pandas DF, then Spark
    weather_pdf = pd.concat(all_weather_dfs, ignore_index=True)
    weather_sdf = spark.createDataFrame(weather_pdf)

    # Truncate both timestamps to the hour for a clean join
    sdf = sdf.withColumn("join_ts", F.date_trunc("hour", "timestamp"))
    weather_sdf = weather_sdf.withColumn("join_ts", F.date_trunc("hour", "timestamp")).drop("timestamp")

    # Broadcast join (weather table is small)
    enriched = sdf.join(
        F.broadcast(weather_sdf),
        on=["station_id", "join_ts"],
        how="left",
    ).drop("join_ts")

    # Join coverage validation
    total = enriched.count()
    with_weather = enriched.filter(F.col("temperature").isNotNull()).count()
    coverage_pct = (with_weather / total * 100) if total > 0 else 0
    print(f"[weather_join] Join coverage: {with_weather:,}/{total:,} "
          f"({coverage_pct:.1f}%)")

    # Check for silent station drops
    orig_stations = sdf.select("station_id").distinct().count()
    enriched_stations = enriched.select("station_id").distinct().count()
    if enriched_stations < orig_stations:
        print(f"[weather_join] WARNING: Station drop detected! "
              f"{orig_stations} → {enriched_stations}")
    else:
        print(f"[weather_join] ✓ All {enriched_stations} stations preserved.")

    return enriched


def save_weather_enriched(sdf: SparkDataFrame) -> None:
    """Save weather-enriched data as Parquet."""
    out_path = str(WEATHER_PARQUET_DIR)
    sdf.write.mode("overwrite").parquet(out_path)
    print(f"[weather_join] Saved enriched data → {out_path}")


# ─────────────────────────── CLI entry point ─────────────────────────────────

def main():
    print(f"{'=' * 60}")
    print(f"  Weather Enrichment")
    print(f"{'=' * 60}\n")

    # Load pollution data
    sdf = load_openaq_parquet()

    # Enrich with weather
    enriched = enrich_with_weather(sdf)

    # Save
    save_weather_enriched(enriched)

    print(f"\n{'=' * 60}")
    print(f"  Weather enrichment complete.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
