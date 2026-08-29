"""
Stage 1.7 — Exploratory Data Analysis

Generates a comprehensive data-quality and pattern report on the cleaned
dataset, producing:
  1. Station coverage & data completeness report
  2. Seasonal patterns (monthly averages per pollutant)
  3. Diurnal patterns (hourly averages per pollutant)
  4. Pollutant correlation matrix
  5. Summary statistics table
  6. Sign-off checklist

All plots are saved to figures/.

Usage
-----
    python notebooks/01_eda.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for script mode

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.config import CLEANED_PARQUET_DIR, FIGURES_DIR, POLLUTANTS, WEATHER_RENAME
from utils.spark_session import get_spark, stop_spark


# ─────────────────────────── Styling ─────────────────────────────────────────

COLORS = {
    "pm25": "#E74C3C",
    "pm10": "#E67E22",
    "no2":  "#3498DB",
    "o3":   "#2ECC71",
    "so2":  "#9B59B6",
    "co":   "#1ABC9C",
}

plt.rcParams.update({
    "figure.facecolor": "#FAFAFA",
    "axes.facecolor": "#FAFAFA",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
})


# ─────────────────────────── Load data ───────────────────────────────────────

def load_cleaned_data() -> pd.DataFrame:
    """Load cleaned Parquet into pandas (via Spark for consistency)."""
    spark = get_spark()
    sdf = spark.read.parquet(str(CLEANED_PARQUET_DIR))
    pdf = sdf.toPandas()
    # Spark's partition-column type inference can turn an all-numeric
    # station_id into a float on read (e.g. "6240023" -> 6240023.0) — undo it.
    pdf["station_id"] = pdf["station_id"].apply(lambda x: str(int(float(x))))
    print(f"[EDA] Loaded {len(pdf):,} rows, {len(pdf.columns)} columns")

    # Ensure timestamp is datetime
    if "timestamp" in pdf.columns:
        pdf["timestamp"] = pd.to_datetime(pdf["timestamp"], utc=True)

    return pdf


# ─────────────────────────── 1. Coverage report ──────────────────────────────

def station_coverage_report(pdf: pd.DataFrame) -> None:
    """Print and save a station-level data completeness report."""
    print("\n" + "=" * 60)
    print("  1. STATION COVERAGE & COMPLETENESS")
    print("=" * 60)

    pollutant_cols = [c for c in POLLUTANTS if c in pdf.columns]

    report_rows = []
    for sid, group in pdf.groupby("station_id"):
        name = group["station_name"].iloc[0] if "station_name" in group.columns else sid
        n_rows = len(group)
        date_min = group["timestamp"].min()
        date_max = group["timestamp"].max()
        days = (date_max - date_min).days + 1

        row = {
            "station_id": sid,
            "station_name": name,
            "rows": n_rows,
            "date_from": str(date_min.date()),
            "date_to": str(date_max.date()),
            "days_span": days,
        }
        for p in pollutant_cols:
            non_null = group[p].notna().sum()
            pct = non_null / n_rows * 100 if n_rows > 0 else 0
            row[f"{p}_coverage_%"] = round(pct, 1)

        report_rows.append(row)

    report = pd.DataFrame(report_rows)
    print(report.to_string(index=False))

    # Save
    report.to_csv(FIGURES_DIR / "station_coverage_report.csv", index=False)
    print(f"\nSaved → {FIGURES_DIR / 'station_coverage_report.csv'}")


# ─────────────────────────── 2. Seasonal patterns ────────────────────────────

def plot_seasonal_patterns(pdf: pd.DataFrame) -> None:
    """Monthly average concentration per pollutant across all stations."""
    print("\n" + "=" * 60)
    print("  2. SEASONAL PATTERNS")
    print("=" * 60)

    pdf["month"] = pdf["timestamp"].dt.month
    pollutant_cols = [c for c in POLLUTANTS if c in pdf.columns and pdf[c].notna().any()]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for idx, p in enumerate(pollutant_cols):
        ax = axes[idx] if idx < len(axes) else axes[-1]
        monthly = pdf.groupby("month")[p].agg(["mean", "std"]).reset_index()

        ax.bar(monthly["month"], monthly["mean"],
               color=COLORS.get(p, "#888"), alpha=0.8,
               yerr=monthly["std"], capsize=3)
        ax.set_title(f"{p.upper()} — Monthly Average")
        ax.set_xlabel("Month")
        ax.set_ylabel("µg/m³")
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
                           rotation=45, ha="right")

    # Hide unused axes
    for idx in range(len(pollutant_cols), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Seasonal Patterns — Monthly Average Concentration", fontsize=16, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "seasonal_patterns.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {FIGURES_DIR / 'seasonal_patterns.png'}")


# ─────────────────────────── 3. Diurnal patterns ─────────────────────────────

def plot_diurnal_patterns(pdf: pd.DataFrame) -> None:
    """Hourly average concentration per pollutant (all stations combined)."""
    print("\n" + "=" * 60)
    print("  3. DIURNAL PATTERNS")
    print("=" * 60)

    pdf["hour"] = pdf["timestamp"].dt.hour
    pollutant_cols = [c for c in POLLUTANTS if c in pdf.columns and pdf[c].notna().any()]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for idx, p in enumerate(pollutant_cols):
        ax = axes[idx] if idx < len(axes) else axes[-1]
        hourly = pdf.groupby("hour")[p].agg(["mean", "std"]).reset_index()

        ax.plot(hourly["hour"], hourly["mean"],
                color=COLORS.get(p, "#888"), linewidth=2, marker="o", markersize=4)
        ax.fill_between(hourly["hour"],
                        hourly["mean"] - hourly["std"],
                        hourly["mean"] + hourly["std"],
                        alpha=0.15, color=COLORS.get(p, "#888"))
        ax.set_title(f"{p.upper()} — Hourly Average")
        ax.set_xlabel("Hour of Day (UTC)")
        ax.set_ylabel("µg/m³")
        ax.set_xticks(range(0, 24, 3))

    for idx in range(len(pollutant_cols), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Diurnal Patterns — Hourly Average Concentration", fontsize=16, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "diurnal_patterns.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {FIGURES_DIR / 'diurnal_patterns.png'}")


# ─────────────────────────── 4. Correlation matrix ───────────────────────────

def plot_correlation_matrix(pdf: pd.DataFrame) -> None:
    """Pollutant + weather correlation heatmap."""
    print("\n" + "=" * 60)
    print("  4. CORRELATION MATRIX")
    print("=" * 60)

    weather_cols = [c for c in WEATHER_RENAME.values() if c in pdf.columns]
    analysis_cols = [c for c in POLLUTANTS if c in pdf.columns] + weather_cols
    analysis_cols = [c for c in analysis_cols if pdf[c].notna().any()]

    if len(analysis_cols) < 2:
        print("  Not enough numeric columns for correlation matrix.")
        return

    corr = pdf[analysis_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(analysis_cols)))
    ax.set_yticks(range(len(analysis_cols)))
    ax.set_xticklabels([c.upper() for c in analysis_cols], rotation=45, ha="right")
    ax.set_yticklabels([c.upper() for c in analysis_cols])

    # Add correlation values as text
    for i in range(len(analysis_cols)):
        for j in range(len(analysis_cols)):
            val = corr.iloc[i, j]
            color = "white" if abs(val) > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=8, color=color)

    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Pollutant & Weather Correlation Matrix", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "correlation_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {FIGURES_DIR / 'correlation_matrix.png'}")


# ─────────────────────────── 5. Summary statistics ───────────────────────────

def summary_statistics(pdf: pd.DataFrame) -> None:
    """Print descriptive statistics for all numeric columns."""
    print("\n" + "=" * 60)
    print("  5. SUMMARY STATISTICS")
    print("=" * 60)

    pollutant_cols = [c for c in POLLUTANTS if c in pdf.columns]
    weather_cols = [c for c in WEATHER_RENAME.values() if c in pdf.columns]
    num_cols = pollutant_cols + weather_cols

    if "population_catchment" in pdf.columns:
        num_cols.append("population_catchment")

    stats = pdf[num_cols].describe().T
    stats["null_%"] = (pdf[num_cols].isnull().sum() / len(pdf) * 100).round(1)
    print(stats.to_string())

    stats.to_csv(FIGURES_DIR / "summary_statistics.csv")
    print(f"\nSaved → {FIGURES_DIR / 'summary_statistics.csv'}")


# ─────────────────────────── 6. Sign-off checklist ───────────────────────────

def signoff_checklist(pdf: pd.DataFrame) -> None:
    """Automated dataset readiness checks."""
    print("\n" + "=" * 60)
    print("  6. SIGN-OFF CHECKLIST")
    print("=" * 60)

    checks = []

    # Check 1: Multiple stations
    n_stations = pdf["station_id"].nunique()
    checks.append(("Multiple stations present", n_stations >= 2, f"{n_stations} stations"))

    # Check 2: Multiple pollutants with data
    pollutant_cols = [c for c in POLLUTANTS if c in pdf.columns and pdf[c].notna().any()]
    checks.append(("Multiple pollutants with data", len(pollutant_cols) >= 3,
                    f"{len(pollutant_cols)} pollutants: {pollutant_cols}"))

    # Check 3: Weather columns present
    weather_cols = [c for c in WEATHER_RENAME.values() if c in pdf.columns]
    checks.append(("Weather enrichment present", len(weather_cols) >= 3,
                    f"{len(weather_cols)} weather cols"))

    # Check 4: Population catchment present
    has_pop = "population_catchment" in pdf.columns and pdf["population_catchment"].notna().any()
    checks.append(("Population catchment present", has_pop, ""))

    # Check 5: No excessive nulls (< 20% in any pollutant)
    max_null_pct = 0
    for p in pollutant_cols:
        null_pct = pdf[p].isnull().sum() / len(pdf) * 100
        max_null_pct = max(max_null_pct, null_pct)
    checks.append(("Null rate < 20% (worst pollutant)", max_null_pct < 20,
                    f"worst: {max_null_pct:.1f}%"))

    # Check 6: Date range > 7 days
    date_span = (pdf["timestamp"].max() - pdf["timestamp"].min()).days
    checks.append(("Date range ≥ 7 days", date_span >= 7, f"{date_span} days"))

    # Check 7: Total rows > 100
    checks.append(("Sufficient data volume (>100 rows)", len(pdf) > 100,
                    f"{len(pdf):,} rows"))

    # Print results
    all_pass = True
    for name, passed, detail in checks:
        icon = "✓" if passed else "✗"
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        extra = f"  ({detail})" if detail else ""
        print(f"  [{icon}] {status}: {name}{extra}")

    print()
    if all_pass:
        print("  ★ ALL CHECKS PASSED — dataset is ready for Part 2 analytics.")
    else:
        print("  ⚠ Some checks failed — review before proceeding to Part 2.")


# ─────────────────────────── Main ────────────────────────────────────────────

def main():
    print(f"{'=' * 60}")
    print(f"  Exploratory Data Analysis — Part 1 Sign-Off")
    print(f"{'=' * 60}\n")

    pdf = load_cleaned_data()

    station_coverage_report(pdf)
    plot_seasonal_patterns(pdf)
    plot_diurnal_patterns(pdf)
    plot_correlation_matrix(pdf)
    summary_statistics(pdf)
    signoff_checklist(pdf)

    stop_spark()

    print(f"\n{'=' * 60}")
    print(f"  EDA complete. Figures saved in: {FIGURES_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
