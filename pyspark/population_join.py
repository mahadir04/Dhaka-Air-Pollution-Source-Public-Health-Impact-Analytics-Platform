"""
Stage 1.4 — Population Data Integration

Joins population density data to each monitoring station by computing
a `population_catchment` value — the estimated number of people living
within a configurable radius of each station.

Two modes:
  --source landscan   Use LandScan Global gridded raster (requires rasterio + .tif file)
  --source synthetic  Generate representative synthetic population data (default)

LandScan Global (https://landscan.ornl.gov/) is Oak Ridge National
Laboratory's ambient-population raster, ~1km (30 arc-second) resolution
worldwide. ORNL requires free account registration before you can download
the GeoTIFF — there's no anonymous/programmatic download endpoint — so:
  1. Register at https://landscan.ornl.gov/ and download the current
     "LandScan Global" GeoTIFF release.
  2. Save it to data/population/landscan_global.tif (or any path, and pass
     --raster <path>).
  3. Run: python pyspark/population_join.py --source landscan

The synthetic mode lets the pipeline run end-to-end without that download —
it's a documented BBS-census-based placeholder, not a substitute for the
raster in the final analysis.

Usage
-----
    python pyspark/population_join.py --source synthetic
    python pyspark/population_join.py --source landscan --raster data/population/landscan_global.tif
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
    POPULATION_CATCHMENT_RADIUS_KM, LANDSCAN_RASTER_PATH, LANDSCAN_CITATION,
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
# Keyed by station_id — covers the fixed 4-station set used by mock_ingest.py
# for offline development.
DHAKA_WARD_POPULATIONS = {
    # station_id → approximate catchment population (within ~3 km radius)
    # Denser core areas vs. suburban areas
    "222094": {"area": "Baridhara/Gulshan", "pop_catchment": 280_000},
    "2796":   {"area": "Dhaka University/Old Dhaka", "pop_catchment": 450_000},
    "236353": {"area": "Dhanmondi", "pop_catchment": 380_000},
    "233592": {"area": "Gulshan", "pop_catchment": 300_000},
}

# Real OpenAQ station names churn (station IDs are reissued), so real-API
# stations are matched by neighborhood keyword in their name instead —
# approximate ward population density (people/km²) per area, from BBS-style
# density tiers (dense core wards vs. planned/diplomatic suburbs).
DHAKA_AREA_DENSITY = {
    "hazaribagh":       ("Hazaribagh/Jigatola", 42_000),
    "jigatola":         ("Hazaribagh/Jigatola", 42_000),
    "mirpur":           ("Mirpur", 38_000),
    "dhanmondi":        ("Dhanmondi", 35_000),
    "moghbazar":        ("Moghbazar", 33_000),
    "badda":            ("Badda", 30_000),
    "dhaka university": ("Dhaka University/Old Dhaka", 40_000),
    "uttara":           ("Uttara", 20_000),
    "baridhara":        ("Baridhara", 16_000),
    "gulshan":          ("Gulshan", 18_000),
    "diplomatic":       ("Baridhara/Gulshan (Diplomatic Zone)", 16_000),
}

# Dhaka metro average density (people/km²) — fallback when neither a
# station_id nor a name keyword match is found.
DEFAULT_DENSITY = 23_000
DEFAULT_CATCHMENT_POP = int(DEFAULT_DENSITY * math.pi * POPULATION_CATCHMENT_RADIUS_KM ** 2)


def generate_synthetic_population(stations_pdf: pd.DataFrame) -> pd.DataFrame:
    """Generate representative population_catchment values for each station.

    Uses ward-level census estimates for known station_ids (the fixed
    mock_ingest.py set), then neighborhood-keyword matching against the
    station name (for real OpenAQ stations, whose IDs churn), then a
    Dhaka-average density heuristic as a last resort. This is a development
    proxy — replace with real LandScan raster integration for production
    analysis (see module docstring).
    """
    area_km2 = math.pi * POPULATION_CATCHMENT_RADIUS_KM ** 2
    rows = []
    for _, row in stations_pdf.iterrows():
        sid = str(row["station_id"])
        name_lower = str(row.get("station_name", "")).lower()

        if sid in DHAKA_WARD_POPULATIONS:
            info = DHAKA_WARD_POPULATIONS[sid]
            pop = info["pop_catchment"]
            area = info["area"]
        else:
            matched = next(
                (v for kw, v in DHAKA_AREA_DENSITY.items() if kw in name_lower),
                None,
            )
            if matched:
                area, density = matched
                pop = int(density * area_km2)
            else:
                # Heuristic: Dhaka average density ≈ 23,000/km²
                pop = int(DEFAULT_DENSITY * area_km2)
                area = "estimated (Dhaka metro average)"
        rows.append({
            "station_id": sid,
            "population_catchment": pop,
            "catchment_area_name": area,
            "data_source": "synthetic_BBS_estimate_pending_landscan",
        })
        print(f"  [pop] Station {sid} ({area}): {pop:,} people in catchment")

    return pd.DataFrame(rows)


# ─────────────────────────── LandScan raster ──────────────────────────────────

def compute_population_from_raster(stations_pdf: pd.DataFrame,
                                    raster_path: str) -> pd.DataFrame:
    """Sum LandScan Global gridded population within a radius of each station.

    LandScan Global (https://landscan.ornl.gov/) distributes an ambient-
    population count raster (~1km / 30 arc-second cells) as a GeoTIFF, in
    the same "population count per cell" convention used here — so summing
    valid cells within a radius window gives the catchment population.

    Requires `rasterio` and a LandScan GeoTIFF downloaded per the
    registration instructions in this module's docstring.
    """
    try:
        import rasterio
        from rasterio.windows import Window
    except ImportError:
        print("[pop] rasterio not installed — falling back to synthetic data.")
        return generate_synthetic_population(stations_pdf)

    raster_file = Path(raster_path)
    if not raster_file.exists():
        print(f"[pop] LandScan raster not found: {raster_file}")
        print(f"[pop] Register and download it from https://landscan.ornl.gov/ "
              f"then place it at that path (see module docstring).")
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
            radius_pixels = max(1, int(radius_deg / pixel_size_deg))

            row_start = max(0, int(row_center) - radius_pixels)
            col_start = max(0, int(col_center) - radius_pixels)
            win_size = radius_pixels * 2

            window = Window(col_start, row_start, win_size, win_size)
            try:
                data = src.read(1, window=window)
            except Exception:
                data = np.array([])

            # Sum population (LandScan uses negative sentinel values for no-data)
            valid = data[(data > 0) & (data < 1e8)]
            pop = int(np.sum(valid)) if valid.size > 0 else DEFAULT_CATCHMENT_POP

            rows.append({
                "station_id": sid,
                "population_catchment": pop,
                "catchment_area_name": f"landscan_r{POPULATION_CATCHMENT_RADIUS_KM}km",
                "data_source": "landscan_raster",
            })
            print(f"  [pop] Station {sid}: {pop:,} people (LandScan raster)")

    print(f"[pop] Source citation: {LANDSCAN_CITATION}")
    return pd.DataFrame(rows)


# ─────────────────────────── Core pipeline ───────────────────────────────────

def load_weather_enriched() -> SparkDataFrame:
    """Load weather-enriched OpenAQ Parquet."""
    spark = get_spark()
    path = str(WEATHER_PARQUET_DIR)
    sdf = spark.read.parquet(path)
    sdf = sdf.withColumn("station_id", F.col("station_id").cast("string"))
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
                        choices=["landscan", "synthetic"],
                        help="Population data source (default: synthetic)")
    parser.add_argument("--raster", type=str, default=str(LANDSCAN_RASTER_PATH),
                        help="Path to LandScan Global GeoTIFF raster file "
                             f"(default: {LANDSCAN_RASTER_PATH})")
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
        sdf.select("station_id", "station_name", "latitude", "longitude")
        .distinct()
        .toPandas()
    )
    print(f"[pop_join] {len(stations_pdf)} unique station(s) to process.\n")

    # Compute population catchment
    if args.source == "landscan":
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
