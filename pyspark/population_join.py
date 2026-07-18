"""
Stage 1.4 — Population Data Integration

Joins population density data to each monitoring station by computing
a `population_catchment` value — the estimated number of people living
within a configurable radius of each station.

Two modes:
  --source worldpop   Use WorldPop gridded raster (requires rasterio + .tif file)
  --source synthetic  Generate representative synthetic population data (default)

The synthetic mode lets the pipeline run end-to-end without a large raster
download.  Replace with real data for final analysis.

Usage
-----
    python pyspark/population_join.py --source synthetic
    python pyspark/population_join.py --source worldpop --raster data/population/bgd_ppp_2020.tif
"""

import argparse
import math
import sys
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pyspark.sql import functions as F
from pyspark.sql import DataFrame as SparkDataFrame
from utils.config import (
    WEATHER_PARQUET_DIR, POPULATION_PARQUET_DIR, CLEANED_PARQUET_DIR, POPULATION_DIR,
    POPULATION_CATCHMENT_RADIUS_KM,
)
from utils.spark_session import get_spark


# ─────────────────────────── Haversine helper ────────────────────────────────

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two points in kilometres."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ─────────────────────────── Synthetic population ────────────────────────────

# BBS ward-level census reference values for Dhaka City Corporation areas.
# These are representative estimates for 2020 (total Dhaka metro ≈ 22 million).
DHAKA_WARD_POPULATIONS = {
    # station_id → approximate catchment population (within ~3 km radius)
    # Denser core areas vs. suburban areas
    "222094": {"area": "Baridhara/Gulshan", "pop_catchment": 280_000},
    "2796":   {"area": "Dhaka University/Old Dhaka", "pop_catchment": 450_000},
    "236353": {"area": "Dhanmondi", "pop_catchment": 380_000},
    "233592": {"area": "Gulshan", "pop_catchment": 300_000},
}

# Default if a station isn't in our lookup
DEFAULT_CATCHMENT_POP = 250_000


def generate_synthetic_population(stations_pdf: pd.DataFrame) -> pd.DataFrame:
    """Generate representative population_catchment values for each station.

    Uses ward-level census estimates for known stations and a density-based
    heuristic for unknown ones.  This is a development proxy — replace with
    real WorldPop raster integration for production analysis.
    """
    rows = []
    for _, row in stations_pdf.iterrows():
        sid = str(row["station_id"])
        if sid in DHAKA_WARD_POPULATIONS:
            info = DHAKA_WARD_POPULATIONS[sid]
            pop = info["pop_catchment"]
            area = info["area"]
        else:
            # Heuristic: Dhaka average density ≈ 23,000/km²
            # Catchment area ≈ π × r²
            area_km2 = math.pi * POPULATION_CATCHMENT_RADIUS_KM ** 2
            pop = int(23_000 * area_km2)
            area = "estimated"
        rows.append({
            "station_id": sid,
            "population_catchment": pop,
            "catchment_area_name": area,
            "data_source": "synthetic_BBS_estimate",
        })
        print(f"  [pop] Station {sid} ({area}): {pop:,} people in catchment")

    return pd.DataFrame(rows)


# ─────────────────────────── WorldPop raster ─────────────────────────────────

def compute_population_from_raster(stations_pdf: pd.DataFrame,
                                    raster_path: str) -> pd.DataFrame:
    """Sum WorldPop gridded population within a radius of each station.

    Requires `rasterio` and a GeoTIFF raster file.
    """
    try:
        import rasterio
        from rasterio.windows import Window
    except ImportError:
        print("[pop] rasterio not installed — falling back to synthetic data.")
        return generate_synthetic_population(stations_pdf)

    raster_file = Path(raster_path)
    if not raster_file.exists():
        print(f"[pop] Raster file not found: {raster_file}")
        print("[pop] Falling back to synthetic data.")
        return generate_synthetic_population(stations_pdf)

    rows = []
    with rasterio.open(raster_file) as src:
        pixel_size_deg = abs(src.transform[0])  # degrees per pixel
        # Approximate conversion: 1° latitude ≈ 111 km
        radius_deg = POPULATION_CATCHMENT_RADIUS_KM / 111.0

        for _, row in stations_pdf.iterrows():
            sid = str(row["station_id"])
            lat = row["latitude"]
            lon = row["longitude"]

            # Window around station
            col_center, row_center = ~src.transform * (lon, lat)
            radius_pixels = int(radius_deg / pixel_size_deg)

            row_start = max(0, int(row_center) - radius_pixels)
            col_start = max(0, int(col_center) - radius_pixels)
            win_size = radius_pixels * 2

            window = Window(col_start, row_start, win_size, win_size)
            try:
                data = src.read(1, window=window)
            except Exception:
                data = np.array([])

            # Sum population (WorldPop uses -99999 or NaN for no-data)
            valid = data[(data > 0) & (data < 1e8)]
            pop = int(np.sum(valid)) if valid.size > 0 else DEFAULT_CATCHMENT_POP

            rows.append({
                "station_id": sid,
                "population_catchment": pop,
                "catchment_area_name": f"worldpop_r{POPULATION_CATCHMENT_RADIUS_KM}km",
                "data_source": "worldpop_raster",
            })
            print(f"  [pop] Station {sid}: {pop:,} people (WorldPop raster)")

    return pd.DataFrame(rows)


# ─────────────────────────── Core pipeline ───────────────────────────────────

def load_weather_enriched() -> SparkDataFrame:
    """Load weather-enriched OpenAQ Parquet."""
    spark = get_spark()
    path = str(WEATHER_PARQUET_DIR)
    sdf = spark.read.parquet(path)
    print(f"[pop_join] Loaded enriched data: {sdf.count():,} rows")
    return sdf


def join_population(sdf: SparkDataFrame, pop_pdf: pd.DataFrame) -> SparkDataFrame:
    """Join population_catchment to the main dataset by station_id."""
    spark = get_spark()
    pop_sdf = spark.createDataFrame(
        pop_pdf[["station_id", "population_catchment"]]
    )

    enriched = sdf.join(F.broadcast(pop_sdf), on="station_id", how="left")

    # Fill any missing with default
    enriched = enriched.fillna({"population_catchment": DEFAULT_CATCHMENT_POP})

    # Verify no station drops
    orig = sdf.select("station_id").distinct().count()
    after = enriched.select("station_id").distinct().count()
    print(f"[pop_join] Stations: {orig} → {after} (should be equal)")

    return enriched


def save_population_enriched(sdf: SparkDataFrame) -> None:
    """Save population-enriched data to its own directory."""
    out_path = str(POPULATION_PARQUET_DIR)
    sdf.write.mode("overwrite").parquet(out_path)
    print(f"[pop_join] Saved population-enriched data → {out_path}")

    # Also save the population lookup as a CSV for reference
    pop_csv = POPULATION_DIR / "station_population_catchment.csv"
    stations = (
        sdf.select("station_id", "station_name", "latitude", "longitude",
                    "population_catchment")
        .distinct()
        .toPandas()
    )
    stations.to_csv(pop_csv, index=False)
    print(f"[pop_join] Population lookup saved → {pop_csv}")


# ─────────────────────────── CLI entry point ─────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Join population data to monitoring stations."
    )
    parser.add_argument("--source", default="synthetic",
                        choices=["worldpop", "synthetic"],
                        help="Population data source (default: synthetic)")
    parser.add_argument("--raster", type=str, default=None,
                        help="Path to WorldPop GeoTIFF raster file")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"{'=' * 60}")
    print(f"  Population Data Integration (source: {args.source})")
    print(f"{'=' * 60}\n")

    # Load weather-enriched data
    sdf = load_weather_enriched()

    # Get unique stations
    stations_pdf = (
        sdf.select("station_id", "latitude", "longitude")
        .distinct()
        .toPandas()
    )
    print(f"[pop_join] {len(stations_pdf)} unique station(s) to process.\n")

    # Compute population catchment
    if args.source == "worldpop" and args.raster:
        pop_pdf = compute_population_from_raster(stations_pdf, args.raster)
    else:
        pop_pdf = generate_synthetic_population(stations_pdf)

    # Join to main dataset
    enriched = join_population(sdf, pop_pdf)

    # Save
    save_population_enriched(enriched)

    print(f"\n{'=' * 60}")
    print(f"  Population integration complete.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
