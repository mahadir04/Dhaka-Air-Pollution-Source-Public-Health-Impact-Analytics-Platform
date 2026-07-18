"""
Mock OpenAQ and weather generation script to allow offline development
and verification of the entire Part 1 pipeline when API keys are not configured.

Generates a realistic 30-day dataset for Dhaka and saves it into the expected Parquet directory structure.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.config import (
    OPENAQ_PARQUET_DIR, WEATHER_PARQUET_DIR,
    POLLUTANTS, WEATHER_RENAME,
)
from pyspark.sql import functions as F
from utils.spark_session import get_spark


def generate_mock_data(days=30):
    print(f"[mock] Generating {days} days of mock data for 4 Dhaka stations...")
    
    stations = [
        {"station_id": "222094", "name": "US Diplomatic Post: Dhaka", "lat": 23.7964, "lon": 90.4243},
        {"station_id": "2796", "name": "Dhaka - CASE", "lat": 23.7260, "lon": 90.3890},
        {"station_id": "236353", "name": "Dhaka - Dhanmondi", "lat": 23.7465, "lon": 90.3760},
        {"station_id": "233592", "name": "Dhaka - Gulshan", "lat": 23.7925, "lon": 90.4150},
    ]

    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)
    timestamps = pd.date_range(start=start_date, end=end_date, freq="h")

    rows = []
    np.random.seed(42)

    for st in stations:
        for ts in timestamps:
            # Add diurnal pattern
            hour = ts.hour
            month = ts.month
            
            # Pollution increases in winter (dry season: Nov-Mar)
            seasonal_mult = 2.5 if month in [11, 12, 1, 2, 3] else 1.0
            
            # Diurnal rush-hour pattern
            diurnal_mult = 1.3 if (7 <= hour <= 9 or 17 <= hour <= 20) else 0.8
            
            # Base pollutant levels
            pm25 = np.random.lognormal(mean=3.5, sigma=0.4) * seasonal_mult * diurnal_mult
            pm10 = pm25 * np.random.uniform(1.2, 1.8)
            no2 = np.random.lognormal(mean=2.8, sigma=0.3) * diurnal_mult
            o3 = np.random.lognormal(mean=2.0, sigma=0.5) * (1.5 if 11 <= hour <= 16 else 0.5)
            so2 = np.random.lognormal(mean=1.5, sigma=0.6) * seasonal_mult  # Brick kilns active in winter
            co = np.random.lognormal(mean=6.0, sigma=0.3) * diurnal_mult

            # Add occasional missing values (before cleaning stage)
            row = {
                "station_id": st["station_id"],
                "station_name": st["name"],
                "latitude": st["lat"],
                "longitude": st["lon"],
                "timestamp": ts,
                "pm25": pm25 if np.random.rand() > 0.05 else None,
                "pm10": pm10 if np.random.rand() > 0.05 else None,
                "no2": no2 if np.random.rand() > 0.05 else None,
                "o3": o3 if np.random.rand() > 0.05 else None,
                "so2": so2 if np.random.rand() > 0.05 else None,
                "co": co if np.random.rand() > 0.05 else None,
            }
            rows.append(row)

    pdf = pd.DataFrame(rows)
    
    # Save as Spark DataFrame/Parquet
    spark = get_spark()
    sdf = spark.createDataFrame(pdf)
    sdf = (
        sdf
        .withColumn("date", F.to_date("timestamp"))
        .withColumn("year", F.year("date"))
        .withColumn("month", F.month("date"))
    )
    
    out_path = str(OPENAQ_PARQUET_DIR)
    sdf.write.mode("overwrite").partitionBy("station_id", "year", "month").parquet(out_path)
    print(f"[mock] Ingested mock data saved successfully to {out_path}")


if __name__ == "__main__":
    generate_mock_data(days=30)
