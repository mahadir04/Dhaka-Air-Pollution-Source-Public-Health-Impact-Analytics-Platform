"""
Stage 1.2 — OpenAQ Multi-Station Ingestion

Pulls multi-station, multi-pollutant air quality data for Dhaka from the
OpenAQ v3 API, unions everything into a wide-format PySpark DataFrame,
and persists as partitioned Parquet.

Usage
-----
    python pyspark/ingest.py --days 365
    python pyspark/ingest.py --start 2023-01-01 --end 2023-12-31
"""

import argparse
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests
import pandas as pd

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, FloatType, TimestampType, IntegerType
)
from utils.config import (
    OPENAQ_API_KEY, OPENAQ_BASE_URL, OPENAQ_PAGE_LIMIT,
    DHAKA_BBOX, POLLUTANTS, OPENAQ_PARAMETER_MAP,
    RAW_DIR, OPENAQ_PARQUET_DIR,
)
from utils.spark_session import get_spark


# ─────────────────────────── API helpers ─────────────────────────────────────

def _headers() -> dict:
    """Build request headers with API key."""
    h = {"Accept": "application/json"}
    if OPENAQ_API_KEY and OPENAQ_API_KEY != "YOUR_OPENAQ_API_KEY_HERE":
        h["X-API-Key"] = OPENAQ_API_KEY
    return h


def discover_dhaka_locations() -> list[dict]:
    """Identify all Dhaka-area monitoring locations from OpenAQ v3.

    Returns a list of dicts with keys: location_id, name, latitude, longitude,
    sensors (list of {sensor_id, parameter} restricted to our POLLUTANTS).

    Note: OpenAQ v3 has no location-level measurements endpoint — each
    location exposes one sensor per parameter it reports, and measurements
    must be pulled per sensor_id (see fetch_sensor_measurements). The
    /locations response already includes each location's sensor list, so
    we capture it here rather than re-deriving it from a fixed parameter map.
    """
    url = f"{OPENAQ_BASE_URL}/locations"
    params = {
        "coordinates": f"{(DHAKA_BBOX['lat_min'] + DHAKA_BBOX['lat_max']) / 2},"
                       f"{(DHAKA_BBOX['lon_min'] + DHAKA_BBOX['lon_max']) / 2}",
        "radius": 25000,  # 25 km radius from Dhaka center
        "limit": 200,
    }

    print("[ingest] Discovering Dhaka-area monitoring locations …")
    try:
        resp = requests.get(url, headers=_headers(), params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as exc:
        print(f"[ingest] WARNING: OpenAQ locations endpoint failed: {exc}")
        print("[ingest] Real ingestion unavailable — use pyspark/mock_ingest.py "
              "for offline development.")
        return []

    results = data.get("results", [])
    if not results:
        print("[ingest] No locations returned by API.")
        return []

    locations = []
    for loc in results:
        lat = loc.get("coordinates", {}).get("latitude")
        lon = loc.get("coordinates", {}).get("longitude")
        if lat is None or lon is None:
            continue
        # Filter to Dhaka bounding box
        if not (DHAKA_BBOX["lat_min"] <= lat <= DHAKA_BBOX["lat_max"] and
                DHAKA_BBOX["lon_min"] <= lon <= DHAKA_BBOX["lon_max"]):
            continue

        sensors = [
            {"sensor_id": s.get("id"), "parameter": s.get("parameter", {}).get("name")}
            for s in loc.get("sensors", [])
            if s.get("parameter", {}).get("name") in POLLUTANTS and s.get("id") is not None
        ]
        if not sensors:
            continue  # no sensor for any pollutant we track

        locations.append({
            "location_id": loc.get("id"),
            "name": loc.get("name", f"station_{loc.get('id')}"),
            "latitude": lat,
            "longitude": lon,
            "sensors": sensors,
        })

    if not locations:
        print("[ingest] No stations inside bounding box report a tracked pollutant.")
        return []

    print(f"[ingest] Found {len(locations)} station(s) in Dhaka metro area.")
    for s in locations:
        params_str = ", ".join(sorted({sn["parameter"] for sn in s["sensors"]}))
        print(f"         • {s['name']} (id={s['location_id']}, "
              f"{s['latitude']:.4f}, {s['longitude']:.4f}) — {params_str}")
    return locations


def fetch_sensor_measurements(sensor_id: int, date_from: str, date_to: str) -> list[dict]:
    """Fetch paginated hourly measurements for one OpenAQ v3 sensor.

    Parameters
    ----------
    sensor_id : int             OpenAQ sensor ID (specific to one station + parameter)
    date_from, date_to : str    ISO-8601 date strings (YYYY-MM-DD)

    Returns
    -------
    list[dict]  Raw measurement records.
    """
    url = f"{OPENAQ_BASE_URL}/sensors/{sensor_id}/measurements/hourly"
    all_records = []
    page = 1

    while True:
        params = {
            "date_from": date_from,
            "date_to": date_to,
            "limit": OPENAQ_PAGE_LIMIT,
            "page": page,
        }
        try:
            resp = requests.get(url, headers=_headers(), params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as exc:
            print(f"  [fetch] Page {page} failed for sensor {sensor_id}: {exc}")
            break

        results = data.get("results", [])
        if not results:
            break

        all_records.extend(results)
        page += 1

        # Rate-limit politeness
        time.sleep(0.25)

        # Safety cap — don't loop forever
        if page > 500:
            print(f"  [fetch] Reached page cap (500) for sensor {sensor_id}")
            break

    return all_records


# ─────────────────────────── Core pipeline ───────────────────────────────────

def ingest_all_stations(locations: list[dict],
                        date_from: str, date_to: str) -> pd.DataFrame:
    """Pull all tracked pollutants for all stations and return a flat
    pandas DataFrame.

    Each row: station_id, station_name, latitude, longitude, timestamp,
              parameter, value, unit

    Iterates per-sensor (each OpenAQ v3 sensor is one station + one
    parameter) rather than assuming every station has every pollutant.
    """
    rows = []
    total = sum(len(loc["sensors"]) for loc in locations)
    counter = 0

    for loc in locations:
        for sensor in loc["sensors"]:
            counter += 1
            poll_key = sensor["parameter"]
            print(f"[ingest] ({counter}/{total}) Station '{loc['name']}' — "
                  f"parameter {poll_key} (sensor {sensor['sensor_id']}) …")
            records = fetch_sensor_measurements(
                sensor["sensor_id"], date_from, date_to
            )
            for rec in records:
                period = rec.get("period", {})
                dt_local = period.get("datetimeFrom", {}).get("local")
                val = rec.get("value")
                if dt_local is None or val is None:
                    continue
                unit = rec.get("parameter", {}).get("units") or \
                    OPENAQ_PARAMETER_MAP.get(poll_key, {}).get("unit", "")
                rows.append({
                    "station_id": str(loc["location_id"]),
                    "station_name": loc["name"],
                    "latitude": loc["latitude"],
                    "longitude": loc["longitude"],
                    "timestamp": dt_local,
                    "parameter": poll_key,
                    "value": float(val),
                    "unit": unit,
                })
            print(f"         → {len(records)} records")

    df = pd.DataFrame(rows)
    if df.empty:
        print("[ingest] WARNING: No data returned from any station!")
        return df

    # The /sensors/{id}/measurements/hourly endpoint does not reliably
    # honor date_from/date_to server-side — some sensors return their full
    # multi-year history regardless. Enforce the requested window here so
    # downstream stages (weather join, etc.) see the range they asked for.
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    window_start = pd.Timestamp(date_from, tz="UTC")
    window_end = pd.Timestamp(date_to, tz="UTC") + pd.Timedelta(days=1)
    before = len(df)
    df = df[(df["timestamp"] >= window_start) & (df["timestamp"] < window_end)].copy()
    if len(df) != before:
        print(f"[ingest] Clamped to requested window {date_from} → {date_to}: "
              f"{before:,} → {len(df):,} records")
    df["timestamp"] = df["timestamp"].astype(str)

    print(f"[ingest] Total raw records: {len(df):,}")
    return df


def pivot_to_wide(pdf: pd.DataFrame) -> pd.DataFrame:
    """Pivot long-format (one row per parameter) to wide format
    (one row per station × timestamp, columns per pollutant).
    """
    if pdf.empty:
        return pdf

    # Parse timestamps
    pdf["timestamp"] = pd.to_datetime(pdf["timestamp"], utc=True)

    # Pivot: index = (station_id, station_name, lat, lon, timestamp),
    #         columns = parameter, values = value
    wide = pdf.pivot_table(
        index=["station_id", "station_name", "latitude", "longitude", "timestamp"],
        columns="parameter",
        values="value",
        aggfunc="mean",  # handle duplicates within same timestamp
    ).reset_index()

    # Flatten column names
    wide.columns.name = None

    # Ensure all pollutant columns exist
    for p in POLLUTANTS:
        if p not in wide.columns:
            wide[p] = None

    print(f"[ingest] Wide-format shape: {wide.shape}")
    return wide


def save_raw_json(pdf: pd.DataFrame, date_from: str, date_to: str) -> None:
    """Save the raw long-format DataFrame as JSON for provenance."""
    if pdf.empty:
        return
    out = RAW_DIR / f"openaq_dhaka_{date_from}_to_{date_to}.json"
    # Convert timestamps to string for JSON serialization
    save_df = pdf.copy()
    if "timestamp" in save_df.columns:
        save_df["timestamp"] = save_df["timestamp"].astype(str)
    save_df.to_json(out, orient="records", indent=2)
    print(f"[ingest] Raw JSON saved → {out}")


def save_as_parquet(wide_pdf: pd.DataFrame) -> None:
    """Convert wide pandas DataFrame to PySpark and save as partitioned Parquet."""
    if wide_pdf.empty:
        print("[ingest] No data to save.")
        return

    spark = get_spark()

    # Convert to Spark DataFrame
    sdf = spark.createDataFrame(wide_pdf)

    # Derive partition columns. station_id is force-cast to string — Spark's
    # partition-column type inference on read can otherwise treat an
    # all-numeric station_id as a double (e.g. "6240023" -> 6240023.0).
    sdf = (
        sdf
        .withColumn("station_id", F.col("station_id").cast("string"))
        .withColumn("date", F.to_date("timestamp"))
        .withColumn("year", F.year("date"))
        .withColumn("month", F.month("date"))
    )

    # Write partitioned by station and year-month
    out_path = str(OPENAQ_PARQUET_DIR)
    sdf.write.mode("overwrite").partitionBy("station_id", "year", "month").parquet(out_path)
    print(f"[ingest] Parquet saved → {out_path}")
    print(f"         Rows: {sdf.count():,}  |  Stations: "
          f"{sdf.select('station_id').distinct().count()}")


# ─────────────────────────── CLI entry point ─────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Ingest multi-station OpenAQ data for Dhaka."
    )
    parser.add_argument("--source", default="openaq", help="Data source (default: openaq)")
    parser.add_argument("--city", default="dhaka", help="City (default: dhaka)")
    parser.add_argument("--days", type=int, default=90,
                        help="Number of past days to pull (default: 90)")
    parser.add_argument("--start", type=str, default=None,
                        help="Start date (YYYY-MM-DD), overrides --days")
    parser.add_argument("--end", type=str, default=None,
                        help="End date (YYYY-MM-DD), defaults to today")
    return parser.parse_args()


def main():
    args = parse_args()

    # Determine date range
    if args.start:
        date_from = args.start
        date_to = args.end or datetime.utcnow().strftime("%Y-%m-%d")
    else:
        end_dt = datetime.utcnow()
        start_dt = end_dt - timedelta(days=args.days)
        date_from = start_dt.strftime("%Y-%m-%d")
        date_to = end_dt.strftime("%Y-%m-%d")

    print(f"{'=' * 60}")
    print(f"  OpenAQ Dhaka Ingestion")
    print(f"  Date range: {date_from} → {date_to}")
    print(f"{'=' * 60}\n")

    # Step 1: Discover stations
    locations = discover_dhaka_locations()
    if not locations:
        print("[ingest] Aborting — no live Dhaka stations found. "
              "Run pyspark/mock_ingest.py for offline development instead.")
        return

    # Step 2: Fetch measurements for all stations & pollutants
    raw_df = ingest_all_stations(locations, date_from, date_to)

    # Step 3: Save raw JSON
    save_raw_json(raw_df, date_from, date_to)

    # Step 4: Pivot to wide format
    wide_df = pivot_to_wide(raw_df)

    # Step 5: Save as partitioned Parquet
    save_as_parquet(wide_df)

    print(f"\n{'=' * 60}")
    print(f"  Ingestion complete.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
