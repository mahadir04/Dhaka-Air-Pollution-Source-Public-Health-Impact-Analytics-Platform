# 🌍 Dhaka Air Pollution Source & Public Health Impact Analytics Platform

**Diagnosing Pollution Sources and Quantifying Population Health Burden for Dhaka using PySpark**

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PySpark](https://img.shields.io/badge/PySpark-3.5-orange.svg)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Active%20Development-brightgreen.svg)

---

## 📌 Overview

Most short-term air pollution projects on Dhaka ask the same question: *what will PM2.5 be next hour?* That's useful, but it stops at a number — it doesn't say **why** pollution is happening, **who** it's actually harming, or **where** intervention would matter most.

This project asks a different question: ***which pollution sources drive Dhaka's air quality problem, how much health burden do they cause, and which areas and seasons carry the greatest risk?***

It's a **data analytics platform**, built primarily to demonstrate distributed data processing with **PySpark**, that:

- Aggregates **multi-station, multi-pollutant** air quality data for Dhaka (PM2.5, PM10, NO₂, O₃, SO₂, CO) via the OpenAQ API
- Identifies **source signatures** — temporal fingerprints matched against known Dhaka emission activity (brick kilns, traffic, construction, biomass burning)
- Computes **population-weighted exposure**, joining pollutant readings to population density data
- Estimates **attributable public health burden** using published epidemiological concentration-response coefficients (WHO / peer-reviewed cohort studies)
- Ranks Dhaka areas and seasons by **estimated health impact**, not raw pollution numbers
- Includes a lightweight short-term forecasting module as a secondary feature

> **One-line pitch:** *Not a forecaster — a diagnosis. This project identifies what's actually driving Dhaka's air pollution, translates it into population-weighted health impact using established epidemiological evidence, and shows where the harm is concentrated.*

---

## 🎯 Objectives

| # | Objective |
|---|-----------|
| 1 | Aggregate multi-station, multi-pollutant air quality data for Dhaka at scale using PySpark |
| 2 | Identify source signatures in pollution data by matching temporal patterns to known emission activity |
| 3 | Compute population-weighted pollution exposure across Dhaka |
| 4 | Estimate attributable health burden using published epidemiological concentration-response coefficients |
| 5 | Rank areas and seasons by estimated health impact, highlighting where intervention matters most |
| 6 | (Secondary) Provide a lightweight short-term forecasting module |

---

## 🧩 How This Differs From a Typical Forecasting Project

| Typical single-station forecaster | This project |
|---|---|
| Forecasts a future pollutant value | Diagnoses causes and quantifies harm using existing data |
| "Early warning" = a predicted number | "Impact" = estimated health burden tied to real epidemiological coefficients |
| Pollution levels only | Pollution levels **weighted by population exposed** |
| No source reasoning | Explicit source-signature matching against known Dhaka emission calendars |
| ML-forecasting-centric | Public-health-analytics-centric, still PySpark-driven |
| Ranks stations by raw AQI | Ranks areas/seasons by estimated attributable health burden |

---

## 🏗 System Architecture

```text
OpenAQ API (multi-station, multi-pollutant, Dhaka)
                          │
                          ▼
Weather Enrichment (temperature, humidity, wind speed/direction, pressure)
                          │
                          ▼
Population Data (WorldPop / census, ward-level)
                          │
                          ▼
PySpark Ingestion & Preprocessing
(cleaning, imputation, outlier handling, joins)
                          │
                          ▼
Source-Signature Analysis (Spark SQL)
(temporal fingerprint matching: brick kilns, traffic, construction, biomass burning)
                          │
                          ▼
Population-Weighted Exposure Calculation
(pollutant concentration × population in catchment)
                          │
                          ▼
Health Burden Estimation
(published concentration-response coefficients applied)
                          │
                          ▼
Comparative Ranking (areas & seasons by estimated health burden)
```

---

## 🧠 Technology Stack

| Layer | Tools |
|---|---|
| **Distributed compute** | Apache Spark, PySpark, Spark SQL, Window functions, MLlib |
| **Language** | Python 3.10+ |
| **Population data** | WorldPop gridded population rasters, or Bangladesh census ward-level tables |
| **Health burden methodology** | Published concentration-response coefficients (WHO Global Air Quality Guidelines, peer-reviewed cohort studies) |
| **ML (secondary forecasting)** | Gradient Boosted Trees, Random Forest (Spark MLlib) |
| **Visualization** | Plotly, Folium (choropleth health-risk map), Matplotlib |
| **Dashboard** | Streamlit |
| **Environment** | Google Colab (Free Tier), GitHub |

---

## 📂 Dataset

### 1. Ground Station Data — OpenAQ (Primary)
[OpenAQ](https://openaq.org/) v3 REST API, filtered to Dhaka-area stations.

| Property | Value |
|---|---|
| Coverage | All available Dhaka-metro monitoring stations |
| Pollutants | PM2.5, PM10, NO₂, O₃, SO₂, CO |
| Granularity | Hourly, per-station |
| Format | JSON (raw API) → Parquet (processed, partitioned by station and date) |
| Licensing | CC BY 4.0 |

### 2. Reference Dataset — Kaggle Single-Station PM2.5
[Hourly PM2.5, Dhaka, 2016–2023](https://www.kaggle.com/datasets/kishorsakib099413/air-quality-index-hourly-25-2016-2023) — used only for long-horizon backtesting of the secondary forecasting module, not for the primary source/health analysis.

### 3. Weather Enrichment
| Feature | Unit | Purpose |
|---|---|---|
| Temperature, Humidity | °C, % | Source-signature disambiguation, exposure context |
| Wind speed / direction | m/s, 0–360° | Distinguishing local emission vs. wind-transported pollution |
| Atmospheric pressure | hPa | Seasonal pattern context |

### 4. Population Data
| Source | Description |
|---|---|
| WorldPop | Free gridded population density rasters, ~100m resolution, usable for Dhaka |
| Bangladesh census (BBS) | Ward-level population figures, alternative/validation source |

### 5. Health Burden Reference Coefficients
| Source | Finding used |
|---|---|
| WHO Global Air Quality Guidelines (2021) | Meta-analytic short-term concentration-response functions for PM2.5/PM10 and mortality |
| Orellano et al. (2020) | ~0.4% increase in all-cause mortality per 10 µg/m³ rise in PM10 |
| Pope et al. | ~8% increase in long-term mortality risk per 10 µg/m³ rise in PM2.5 |
| India difference-in-differences study (2024) | ~8.6% higher annual mortality per 10 µg/m³ rise in annual PM2.5 |

---

## 🔬 Core Methodology

### 1. Data Aggregation (PySpark)
Multi-station OpenAQ pulls are unioned and pivoted to wide format, joined with weather and population data.

### 2. Source-Signature Analysis (diagnostic)
Temporal/pollutant patterns are matched against known Dhaka emission activity calendars using Spark SQL aggregations.

### 3. Population-Weighted Exposure
```text
exposure_score(station, time) = pollutant_concentration × population_in_catchment
```

### 4. Health Burden Estimation
Published concentration-response coefficients are applied to compute estimated attributable risk percentage per station/area/season relative to the WHO guideline baseline.

### 5. Comparative Ranking
Areas and seasons are ranked by estimated health burden, not raw pollution.

### 6. (Secondary) Short-Term Forecasting
A lightweight Gradient Boosted Trees / Random Forest regression model (Spark MLlib) is used for a supporting forecasting module.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Java 8/11 (required by Spark)
- OpenAQ API key (free)

### Installation

```bash
git clone https://github.com/mahadir04/Dhaka-Air-Pollution-Source-Public-Health-Impact-Analytics-Platform.git
cd Dhaka-Air-Pollution-Source-Public-Health-Impact-Analytics-Platform

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### Run the pipeline

```bash
python pyspark/ingest.py --source openaq --city dhaka --days 365
python pyspark/weather_join.py
python pyspark/population_join.py --source worldpop
python pyspark/preprocess.py
python source_analysis/detect_signatures.py
python exposure/compute_exposure.py
python health_burden/apply_crf.py
python ranking/rank_health_burden.py
python forecasting/train_regression.py --model gbt
streamlit run dashboard/app.py
```

---

## 🌟 Key Contributions

- Multi-station, multi-pollutant Dhaka air quality dataset assembled via PySpark
- Rule-based source-signature diagnostics matching pollution patterns to known Dhaka emission activity
- Population-weighted exposure calculation
- Application of published epidemiological concentration-response coefficients to estimate attributable health burden
- Health-burden-based area and seasonal ranking
- A lightweight secondary forecasting module retained to demonstrate Spark MLlib competency

---

## 📚 Future Work

- [ ] Validate estimated health burden against real hospital admission / respiratory ED visit data for Dhaka
- [ ] Formalize source-signature diagnostics into a proper source-apportionment model
- [ ] Refine population-weighting with higher-resolution local census data
- [ ] Extend health burden estimation to morbidity outcomes
- [ ] Compare estimated burden rankings against actual policy interventions

---

## 👨‍💻 Author

Undergraduate Data Analytics Project — Department of Computer Science and Engineering

Built with Apache Spark, PySpark, Spark MLlib, and applied public health analytics methodology.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## ⭐ Support

If you find this project useful, consider starring the repository and contributing improvements via pull requests.
