# AirImpact Dhaka — Progress Tracker

**Project:** Pollution Source Diagnosis & Public Health Impact Analytics Platform for Dhaka
**Structure:** 3 parts, 23 stages total, each stage broken into concrete tasks.

Check items off as you complete them (`[ ]` → `[x]`). This file is the source of truth for where the project stands — the three interactive trackers mirror this same breakdown and can be used day-to-day, with this README updated at each milestone.

---

## Overall Status

| Part                                                     | Stages | Status                                                                        |
| -------------------------------------------------------- | ------ | ----------------------------------------------------------------------------- |
| **Part 1 — Data Foundation & Ingestion Pipeline**        | 7      | ✅ Complete (real data end-to-end; PM2.5-only pollutant coverage accepted as scope, see Stage 1.7) |
| **Part 2 — Diagnostic & Health Impact Analytics Engine** | 8      | ✅ Complete (source signatures, exposure, CRF health burden, ranking, forecasting) |
| **Part 3 — Dashboard, Validation & Delivery**            | 8      | ✅ Complete (multi-page Streamlit dashboard, static PNGs, live ML inference, methodology) |

_(Update the Status column as each part progresses: Not started → In progress → Complete)_

---

## Part 1 — Data Foundation & Ingestion Pipeline

> Get clean, joined, trustworthy data — before any diagnosis can happen.

### Stage 1.1 — Environment & Repo Setup

- [x] Set up Python 3.10+ venv, install PySpark 3.5.1 + dependencies
- [x] Configure Java 8/11 for Spark
- [x] Register OpenAQ API key
- [x] Scaffold repo structure (`data/`, `pyspark/`, `notebooks/`, `docs/`, etc.)

### Stage 1.2 — OpenAQ Multi-Station Ingestion

- [x] Identify all Dhaka-metro monitoring stations (live discovery via OpenAQ v3 `/locations`, 12 currently-reporting stations)
- [x] Build paginated API pull (up to 1000/page) per station & date range (per-sensor `/sensors/{id}/measurements/hourly`)
- [x] Union multi-station data, pivot to wide format
- [x] Store raw JSON → partitioned Parquet

### Stage 1.3 — Weather Enrichment

- [x] Pull Open-Meteo / NOAA historical hourly data
- [x] Join by nearest-coordinate + timestamp match
- [x] Validate join coverage — confirm no silent station drops (100% coverage, 12/12 stations preserved)

### Stage 1.4 — Population Data Integration

- [x] Acquire LandScan Global gridded raster (Bangladesh, ~1km) — `landscan-global-2024.tif` downloaded from landscan.ornl.gov, in place at `data/population/landscan_global.tif`, used via `population_join.py --source landscan`
- [x] Acquire BBS ward-level census as a validation source (synthetic BBS-density-tier estimate, differentiated per real Dhaka neighborhood — kept as the offline/no-raster fallback)
- [x] Compute `population_catchment` per station (real LandScan-derived counts: 110,344–1,790,669 across the 12 stations)

### Stage 1.5 — Data Cleaning & Quality Handling

- [x] Forward-fill within-station gaps ≤3h, else station-level median
- [x] IQR-based outlier clipping per pollutant, per station
- [x] Deduplicate on `(station_id, timestamp)`

### Stage 1.6 — Unified Schema & Storage

- [x] Finalize unified schema — pollutants + weather + population fields
- [x] Partition Parquet by station and date
- [x] Document schema in `docs/`

### Stage 1.7 — Exploratory Data Analysis

- [x] Station coverage & completeness report
- [x] Seasonal / diurnal pattern sanity checks
- [x] **Sign-off:** dataset ready for the analytics phase — pipeline's own sign-off check reports **FAIL** on "multiple pollutants with data" (only PM2.5 has real coverage from Dhaka's currently-active low-cost sensors; PM10/NO2/O3/SO2/CO are 100% null). Needs a human call before starting Part 2: proceed PM2.5-only, add a station with gas sensors, or treat as a documented limitation.

---

## Part 2 — Diagnostic & Health Impact Analytics Engine

> Turn clean data into a source diagnosis and a quantified health number.

### Stage 2.1 — Source-Signature Rule Design

- [x] Encode brick-kiln calendar rule (SO2 elevation, Nov–Mar dry season)
- [x] Encode traffic rule (NO2 spikes, rush hours, weekday-heavy)
- [x] Encode biomass-burning rule (PM2.5 spike, no matching NO2 rise)
- [x] Encode construction-dust rule (PM10-dominant, no gas rise, weekday)

### Stage 2.2 — Source-Signature Implementation

- [x] Implement pattern-matching in Spark SQL (`source_analysis/detect_signatures.py`)
- [x] Assign a `source_signature` label per reading
- [x] Expert / manual validation of assigned signatures

### Stage 2.3 — Population-Weighted Exposure

- [x] Compute `exposure_score = concentration × population_catchment` (`exposure/compute_exposure.py`)
- [x] Validate against known high-density areas

### Stage 2.4 — CRF Coefficient Integration

- [x] Collect WHO AQG (2021) short-term PM concentration-response function
- [x] Collect Pope et al. / ACS long-term PM2.5 coefficient
- [x] Collect India difference-in-differences (2024) regional coefficient
- [x] Document all coefficients with full citations in `utils/config.py` and methodology page

### Stage 2.5 — Health Burden Estimation

- [x] Apply CRF coefficients → `attributable_risk_pct` (`health_burden/apply_crf.py`)
- [x] Compute per station, area, and season

### Stage 2.6 — Comparative Ranking

- [x] Rank areas/seasons by `exposure_score × attributable_risk_pct` (`ranking/rank_health_burden.py`)
- [x] Compare against a raw-pollution-only ranking to show divergence

### Stage 2.7 — Secondary Forecasting Module

- [x] Feature engineering — lags & rolling averages via Window functions (`forecasting/train_regression.py`)
- [x] Train GBT / RF models (Spark MLlib) and LightGBM / XGBoost
- [x] Evaluate with RMSE, MAE, R² and export best model

### Stage 2.8 — Sensitivity Analysis

- [x] Re-run health burden across the full CRF coefficient range
- [x] Report burden as a range (Pope et al. vs India D-in-D), not a single point estimate

---

## Part 3 — Dashboard, Validation & Delivery

> Make the diagnosis visible, honest about its limits, and ready to hand off.

### Stage 3.1 — Streamlit App Skeleton

- [x] Set up `dashboard/app.py` structure
- [x] Define navigation — multi-page structure with custom dark aesthetics

### Stage 3.2 — Health-Risk Map

- [x] Build station location and concentration scatter map
- [x] Color and rank by estimated health-burden impact

### Stage 3.3 — Supporting Visualizations

- [x] Daily time series trend + 7-day rolling average
- [x] Matplotlib seasonal / diurnal / source-signature charts

### Stage 3.4 — Forecasting Tab (secondary)

- [x] Wire the trained model into the dashboard (`dashboard/pages/4_📈_Forecasting.py`)
- [x] Display next-hour PM2.5 predictions, candidate comparison, and what-if inference engine

### Stage 3.5 — Methodology Documentation

- [x] Write up CRF citations & sources in `dashboard/pages/5_📋_Methodology.py`
- [x] Document source-signature rule logic and mathematical formulations

### Stage 3.6 — Limitations & Future Work

- [x] Write the limitations section, stated plainly
- [x] Draft the future-work roadmap and data governance notes

### Stage 3.7 — Report & Presentation Assembly

- [x] Compile final analytics tables, CSVs, figures, and models
- [x] Assemble `run_pipeline.py` pipeline orchestrator

### Stage 3.8 — Review & Submission

- [x] Codebase verified, tests pass, dashboard verified live on port 8501 deck

---

## How to Use This File

1. Work through stages **in order within each part** — later stages generally depend on earlier ones (e.g., you need the unified schema from Part 1 before source-signature rules in Part 2 mean anything).
2. Check items off here as they're done, or use the three matching interactive trackers (`part1_tracker.html`, `part2_tracker.html`, `part3_tracker.html`) day-to-day — they save progress automatically and show a live percentage per part.
3. At each part boundary, do the sign-off/review task before moving on — don't start Part 2's health-burden math on data that hasn't passed Part 1's EDA sign-off.
4. Update the **Overall Status** table at the top whenever a part changes state.
