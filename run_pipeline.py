"""
Orchestration Script — AirImpact Dhaka Analytics Platform

Executes the complete diagnostic, health impact, and forecasting pipeline:
  1. Validates processed data foundation
  2. Runs source-signature detection (Spark SQL)
  3. Computes population-weighted exposure
  4. Estimates public health burden using epidemiological CRFs
  5. Computes comparative health-impact ranking
  6. Trains and evaluates PM2.5 forecasting models (Spark MLlib / boosting)
  7. Exports unified dashboard dataset (outputs/enriched_data.parquet)
  8. Optionally launches the Streamlit dashboard

Usage:
    python run_pipeline.py
    python run_pipeline.py --skip-training
    python run_pipeline.py --launch-dashboard
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import (
    CLEANED_PARQUET_DIR,
    OUTPUTS_DIR,
    FIGURES_DIR,
    MODEL_DIR,
    MODEL_SKLEARN_DIR,
)


def print_banner():
    banner = """
======================================================================
  🌍 AirImpact Dhaka — Diagnostic & Health Impact Analytics Platform
======================================================================
    """
    print(banner)


def export_dashboard_dataset():
    """Export cleaned data to outputs/enriched_data.parquet for Streamlit."""
    print("\n📦 Exporting dashboard dataset to outputs/enriched_data.parquet...")
    try:
        import pandas as pd
        if CLEANED_PARQUET_DIR.exists():
            df = pd.read_parquet(CLEANED_PARQUET_DIR)
            # Add convenient aliases and temporal features
            if "station_name" in df.columns and "name" not in df.columns:
                df["name"] = df["station_name"]
            if "timestamp" in df.columns and "ts" not in df.columns:
                df["ts"] = df["timestamp"]
            
            ts_series = pd.to_datetime(df["timestamp"])
            df["hour"] = ts_series.dt.hour
            df["month"] = ts_series.dt.month
            df["season"] = df["month"].map(
                lambda m: "Winter" if m in [12, 1, 2]
                else ("Pre-Monsoon" if m in [3, 4, 5]
                else ("Monsoon" if m in [6, 7, 8, 9]
                else "Post-Monsoon"))
            )
            
            output_parquet = OUTPUTS_DIR / "enriched_data.parquet"
            df.to_parquet(output_parquet, index=False)
            print(f"  ✓ Saved {len(df):,} rows → {output_parquet}")
            return True
        else:
            print("  ⚠️ Cleaned parquet directory not found.")
            return False
    except Exception as e:
        print(f"  ⚠️ Could not export dashboard parquet: {e}")
        return False


def run_stage(step_num, title, script_path, args=None):
    print(f"\n[{step_num}/6] 🚀 Running: {title}")
    cmd = [sys.executable, str(script_path)]
    if args:
        cmd.extend(args)
    res = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if res.returncode != 0:
        print(f"  ⚠️ Warning: {title} exited with code {res.returncode}")
        return False
    print(f"  ✓ {title} completed successfully.")
    return True


def main():
    print_banner()

    parser = argparse.ArgumentParser(description="Run the AirImpact Dhaka analytics pipeline.")
    parser.add_argument("--skip-training", action="store_true", help="Skip forecasting model training")
    parser.add_argument("--launch-dashboard", action="store_true", help="Launch Streamlit dashboard upon completion")
    args = parser.parse_args()

    # Ensure output directories exist
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_SKLEARN_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Source signature analysis
    run_stage(1, "Source-Signature Detection", PROJECT_ROOT / "source_analysis" / "detect_signatures.py")

    # 2. Population-weighted exposure
    run_stage(2, "Population-Weighted Exposure Calculation", PROJECT_ROOT / "exposure" / "compute_exposure.py")

    # 3. Health burden estimation
    run_stage(3, "Health Burden Estimation (CRFs)", PROJECT_ROOT / "health_burden" / "apply_crf.py")

    # 4. Health-impact ranking
    run_stage(4, "Comparative Health-Impact Ranking", PROJECT_ROOT / "ranking" / "rank_health_burden.py")

    # 5. Model training
    if not args.skip_training:
        run_stage(5, "PM2.5 Forecasting Model Training & Export", PROJECT_ROOT / "forecasting" / "train_regression.py")
    else:
        print("\n[5/6] ⏭️ Skipping model training (--skip-training specified).")

    # 6. Export dashboard dataset
    export_dashboard_dataset()

    print("\n" + "=" * 70)
    print("  🎉 End-to-End Analytics Pipeline Complete!")
    print("  Artifacts available in:")
    print(f"    - Outputs: {OUTPUTS_DIR}")
    print(f"    - Figures: {FIGURES_DIR}")
    print("=" * 70 + "\n")

    if args.launch_dashboard:
        print("🚀 Launching Streamlit dashboard...")
        streamlit_bin = PROJECT_ROOT / ".venv" / "Scripts" / "streamlit.exe"
        if streamlit_bin.exists():
            cmd = [str(streamlit_bin), "run", "dashboard/app.py"]
        else:
            cmd = [sys.executable, "-m", "streamlit", "run", "dashboard/app.py"]
        subprocess.run(cmd, cwd=str(PROJECT_ROOT))


if __name__ == "__main__":
    main()
