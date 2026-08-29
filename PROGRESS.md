# AirImpact Dhaka — Progress Tracker

**Project:** Pollution Source Diagnosis & Public Health Impact Analytics Platform for Dhaka
**Structure:** 3 parts, 23 stages total, each stage broken into concrete tasks.

Check items off as you complete them (`[ ]` → `[x]`). This file is the source of truth for where the project stands — the three interactive trackers mirror this same breakdown and can be used day-to-day, with this README updated at each milestone.

---

## Overall Status

| Part                                                     | Stages | Status                                                                        |
| -------------------------------------------------------- | ------ | ----------------------------------------------------------------------------- |
| **Part 1 — Data Foundation & Ingestion Pipeline**        | 7      | ✅ Complete (real data end-to-end; PM2.5-only pollutant coverage accepted as scope, see Stage 1.7) |
| **Part 2 — Diagnostic & Health Impact Analytics Engine** | 8      | ☐ Not started                                                                 |
| **Part 3 — Dashboard, Validation & Delivery**            | 8      | ☐ Not started                                                                 |

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

- [ ] Encode brick-kiln calendar rule (SO2 elevation, Nov–Mar dry season)
- [ ] Encode traffic rule (NO2 spikes, rush hours, weekday-heavy)
- [ ] Encode biomass-burning rule (PM2.5 spike, no matching NO2 rise)
- [ ] Encode construction-dust rule (PM10-dominant, no gas rise, weekday)

### Stage 2.2 — Source-Signature Implementation

- [ ] Implement pattern-matching in Spark SQL
- [ ] Assign a `source_signature` label per reading
- [ ] Expert / manual validation of assigned signatures

### Stage 2.3 — Population-Weighted Exposure

- [ ] Compute `exposure_score = concentration × population_catchment`
- [ ] Validate against known high-density areas

### Stage 2.4 — CRF Coefficient Integration

- [ ] Collect WHO AQG (2021) short-term PM concentration-response function
- [ ] Collect Pope et al. / ACS long-term PM2.5 coefficient
- [ ] Collect India difference-in-differences (2024) regional coefficient
- [ ] Document all coefficients with full citations

### Stage 2.5 — Health Burden Estimation

- [ ] Apply CRF coefficients → `attributable_risk_pct`
- [ ] Compute per station, area, and season

### Stage 2.6 — Comparative Ranking

- [ ] Rank areas/seasons by `exposure_score × attributable_risk_pct`
- [ ] Compare against a raw-pollution-only ranking to show divergence

### Stage 2.7 — Secondary Forecasting Module

- [ ] Feature engineering — lags & rolling averages via Window functions
- [ ] Train GBT / RF models (Spark MLlib)
- [ ] Evaluate with RMSE, MAE, R²

### Stage 2.8 — Sensitivity Analysis

- [ ] Re-run health burden across the full CRF coefficient range
- [ ] Report burden as a range, not a single point estimate

---

## Part 3 — Dashboard, Validation & Delivery

> Make the diagnosis visible, honest about its limits, and ready to hand off.

### Stage 3.1 — Streamlit App Skeleton

- [ ] Set up `dashboard/app.py` structure
- [ ] Define navigation — health-risk tab + forecasting tab

### Stage 3.2 — Health-Risk Map

- [ ] Build Folium choropleth by area / ward
- [ ] Color by estimated health-burden rank

### Stage 3.3 — Supporting Visualizations

- [ ] Plotly time series — pollutant concentration + exposure
- [ ] Matplotlib seasonal / source-signature charts

### Stage 3.4 — Forecasting Tab (secondary)

- [ ] Wire the GBT / RF model into the dashboard
- [ ] Display next-hour / next-day PM2.5 prediction

### Stage 3.5 — Methodology Documentation

- [ ] Write up CRF citations & sources
- [ ] Document source-signature rule logic

### Stage 3.6 — Limitations & Future Work

- [ ] Write the limitations section, stated plainly
- [ ] Draft the future-work roadmap

### Stage 3.7 — Report & Presentation Assembly

- [ ] Compile the final written report
- [ ] Assemble / refresh the slide deck

### Stage 3.8 — Review & Submission

- [ ] Internal review / dry run
- [ ] Final submission

---

## How to Use This File

1. Work through stages **in order within each part** — later stages generally depend on earlier ones (e.g., you need the unified schema from Part 1 before source-signature rules in Part 2 mean anything).
2. Check items off here as they're done, or use the three matching interactive trackers (`part1_tracker.html`, `part2_tracker.html`, `part3_tracker.html`) day-to-day — they save progress automatically and show a live percentage per part.
3. At each part boundary, do the sign-off/review task before moving on — don't start Part 2's health-burden math on data that hasn't passed Part 1's EDA sign-off.
4. Update the **Overall Status** table at the top whenever a part changes state.
