"""
Central configuration for the Dhaka Air Health Analytics project.

All paths, API keys, pollutant lists, WHO baseline values, and epidemiological
coefficients are defined here so every script draws from one source of truth.
"""

import os
from pathlib import Path

# ─────────────────────────── Project root ────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ─────────────────────────── Directory paths ─────────────────────────────────
DATA_DIR           = PROJECT_ROOT / "data"
RAW_DIR            = DATA_DIR / "raw"
PROCESSED_DIR      = DATA_DIR / "processed"
OPENAQ_PARQUET_DIR = PROCESSED_DIR / "openaq"
WEATHER_PARQUET_DIR = PROCESSED_DIR / "weather"
POPULATION_PARQUET_DIR = PROCESSED_DIR / "population_enriched"
CLEANED_PARQUET_DIR = PROCESSED_DIR / "cleaned"
POPULATION_DIR     = DATA_DIR / "population"
FIGURES_DIR        = PROJECT_ROOT / "figures"
DOCS_DIR           = PROJECT_ROOT / "docs"
REPORTS_DIR        = PROJECT_ROOT / "reports"

# Ensure key directories exist
for d in [RAW_DIR, OPENAQ_PARQUET_DIR, WEATHER_PARQUET_DIR, POPULATION_PARQUET_DIR,
          CLEANED_PARQUET_DIR, POPULATION_DIR, FIGURES_DIR,
          DOCS_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────── API keys ────────────────────────────────────────
# Set your OpenAQ API key as an environment variable or replace the placeholder.
OPENAQ_API_KEY = os.environ.get("OPENAQ_API_KEY", "YOUR_OPENAQ_API_KEY_HERE")

# ─────────────────────────── OpenAQ settings ─────────────────────────────────
OPENAQ_BASE_URL = "https://api.openaq.org/v3"
OPENAQ_PAGE_LIMIT = 1000  # max records per API page

# Dhaka bounding box (approximate metro area)
DHAKA_BBOX = {
    "lat_min": 23.65,
    "lat_max": 23.90,
    "lon_min": 90.30,
    "lon_max": 90.50,
}

# ─────────────────────────── Pollutants ──────────────────────────────────────
POLLUTANTS = ["pm25", "pm10", "no2", "o3", "so2", "co"]

# OpenAQ parameter IDs (v3 API uses numeric IDs for parameters)
OPENAQ_PARAMETER_MAP = {
    "pm25": {"id": 2, "name": "pm25",  "unit": "µg/m³"},
    "pm10": {"id": 1, "name": "pm10",  "unit": "µg/m³"},
    "no2":  {"id": 7, "name": "no2",   "unit": "µg/m³"},
    "o3":   {"id": 10, "name": "o3",   "unit": "µg/m³"},
    "so2":  {"id": 9, "name": "so2",   "unit": "µg/m³"},
    "co":   {"id": 38, "name": "co",   "unit": "µg/m³"},
}

# ─────────────────────────── Weather features ────────────────────────────────
WEATHER_FEATURES = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "surface_pressure",
]

# Rename map: Open-Meteo column → our schema column
WEATHER_RENAME = {
    "temperature_2m":       "temperature",
    "relative_humidity_2m": "humidity",
    "wind_speed_10m":       "wind_speed",
    "wind_direction_10m":   "wind_direction",
    "surface_pressure":     "pressure",
}

# ─────────────────────────── WHO baselines ───────────────────────────────────
# WHO Air Quality Guideline (2021) annual-mean recommendations (µg/m³)
WHO_GUIDELINE_ANNUAL = {
    "pm25": 5.0,
    "pm10": 15.0,
    "no2":  10.0,
    "o3":   60.0,   # peak-season 8h-mean guideline
    "so2":  40.0,   # 24h guideline
    "co":   4000.0, # 24h guideline (µg/m³ equivalent)
}

# ─────────────────── Concentration-Response Coefficients ─────────────────────
# Published CRF values used in health burden estimation (Part 2).
# Stored here for single-source-of-truth.
CRF_COEFFICIENTS = {
    "pm25_short_term": {
        "source": "WHO AQG 2021 / Orellano et al. 2020",
        "pollutant": "pm25",
        "pct_increase_per_10ug": 0.4,
        "outcome": "all-cause mortality (short-term)",
    },
    "pm25_long_term_pope": {
        "source": "Pope et al., ACS cohort study",
        "pollutant": "pm25",
        "pct_increase_per_10ug": 8.0,
        "outcome": "all-cause mortality (long-term)",
    },
    "pm25_long_term_india": {
        "source": "India diff-in-diff study, 2024",
        "pollutant": "pm25",
        "pct_increase_per_10ug": 8.6,
        "outcome": "annual mortality (long-term, South Asian context)",
    },
    "pm10_short_term": {
        "source": "Orellano et al. 2020, cited in WHO AQG",
        "pollutant": "pm10",
        "pct_increase_per_10ug": 0.4,
        "outcome": "all-cause mortality (short-term, linear CRF)",
    },
}

# ─────────────────────────── Population defaults ─────────────────────────────
# Catchment radius in km for assigning population to each station
POPULATION_CATCHMENT_RADIUS_KM = 3.0

# ─────────────────────────── Spark defaults ──────────────────────────────────
SPARK_APP_NAME = "DhakaAirHealthAnalytics"
SPARK_DRIVER_MEMORY = "4g"
SPARK_LOG_LEVEL = "WARN"

# ─────────────────────────── Data cleaning ───────────────────────────────────
# Max gap (in hours) for forward-fill before falling back to station median
MAX_FFILL_GAP_HOURS = 3

# IQR multiplier for outlier clipping
IQR_MULTIPLIER = 1.5
