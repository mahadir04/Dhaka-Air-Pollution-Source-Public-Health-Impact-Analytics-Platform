# Unified Schema — Dhaka Air Health Analytics

All downstream analytics (source-signature analysis, exposure calculation,
health burden estimation) read from the **cleaned Parquet store** at
`data/processed/cleaned/`, which conforms to the schema below.

## Partitioning

```
data/processed/cleaned/
  └── station_id=<id>/
        └── year=<YYYY>/
              └── month=<MM>/
                    └── part-*.parquet
```

## Column Reference

### Identity & Location

| Column | Type | Description | Source |
|---|---|---|---|
| `station_id` | string | OpenAQ location identifier | OpenAQ API |
| `station_name` | string | Human-readable station name | OpenAQ API |
| `latitude` | float | Station latitude (WGS-84) | OpenAQ API |
| `longitude` | float | Station longitude (WGS-84) | OpenAQ API |
| `timestamp` | timestamp | Reading time (UTC) | OpenAQ API |

### Pollutant Concentrations

| Column | Type | Unit | Description | Source |
|---|---|---|---|---|
| `pm25` | float | µg/m³ | Fine particulate matter (≤2.5 µm) | OpenAQ |
| `pm10` | float | µg/m³ | Coarse particulate matter (≤10 µm) | OpenAQ |
| `no2` | float | µg/m³ | Nitrogen dioxide | OpenAQ |
| `o3` | float | µg/m³ | Ozone | OpenAQ |
| `so2` | float | µg/m³ | Sulfur dioxide | OpenAQ |
| `co` | float | µg/m³ | Carbon monoxide | OpenAQ |

### Weather Enrichment

| Column | Type | Unit | Description | Source |
|---|---|---|---|---|
| `temperature` | float | °C | Air temperature at 2m | Open-Meteo |
| `humidity` | float | % | Relative humidity at 2m | Open-Meteo |
| `wind_speed` | float | m/s | Wind speed at 10m | Open-Meteo |
| `wind_direction` | float | ° (0–360) | Wind direction at 10m | Open-Meteo |
| `pressure` | float | hPa | Surface pressure | Open-Meteo |

### Population

| Column | Type | Description | Source |
|---|---|---|---|
| `population_catchment` | int | Estimated population within station catchment area | LandScan Global / BBS census |

### Partition Columns

| Column | Type | Description |
|---|---|---|
| `date` | date | Calendar date (derived from timestamp) |
| `year` | int | Year (partition key) |
| `month` | int | Month (partition key) |

## Columns Added in Part 2 (downstream)

These columns are computed during the diagnostic analytics phase and
appended to copies of the cleaned dataset:

| Column | Type | Description | Added by |
|---|---|---|---|
| `exposure_score` | float | `pollutant_concentration × population_catchment` | `exposure/compute_exposure.py` |
| `source_signature` | string | Diagnosed likely dominant source (traffic / brick_kiln / construction / biomass / mixed) | `source_analysis/detect_signatures.py` |
| `attributable_risk_pct` | float | Estimated excess health risk vs. WHO guideline baseline | `health_burden/apply_crf.py` |

## Data Quality Notes

- **Missing values**: Forward-filled within-station for gaps ≤ 3 hours;
  longer gaps filled with station-level hourly median.
- **Outliers**: IQR-based clipping (1.5 × IQR) per pollutant, per station.
- **Duplicates**: Deduplicated on `(station_id, timestamp)`, keeping the
  row with the most non-null pollutant readings.
- **Timezone**: All timestamps are in UTC.
