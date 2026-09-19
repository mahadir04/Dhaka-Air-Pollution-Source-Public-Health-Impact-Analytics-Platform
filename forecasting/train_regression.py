"""
PM2.5 Forecasting — Train & Export Model.

Exact same architecture as notebook Cells 8.1–8.2b:
  Features: pm25_lag1, pm25_lag24, pm25_roll24, hour, dow, month, season_enc,
            + weather features (temp, humidity, wind, blh)
  Models:   Persistence, LinearRegression, RandomForest, GBT (Spark MLlib)
            + LightGBM, XGBoost (sklearn)
  Split:    Chronological 80/20
  Winner:   Best RMSE via train_and_compare_models() harness
  Export:   Spark model → outputs/model/, sklearn → outputs/model_sklearn/

Usage:
    python forecasting/train_regression.py
    python forecasting/train_regression.py --model gbt
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from pyspark.sql import functions as F, Window
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import GBTRegressor, RandomForestRegressor, LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from utils.config import (
    CLEANED_PARQUET_DIR, OUTPUTS_DIR, MODEL_DIR, MODEL_SKLEARN_DIR, FIGURES_DIR
)
from utils.spark_session import get_spark

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Try importing optional boosting libraries ────────────────────────────────
try:
    from lightgbm import LGBMRegressor
    HAVE_LIGHTGBM = True
except ImportError:
    HAVE_LIGHTGBM = False

try:
    from xgboost import XGBRegressor
    HAVE_XGBOOST = True
except ImportError:
    HAVE_XGBOOST = False


# ── Evaluation helpers (same as notebook) ────────────────────────────────────

def evaluate_regression(preds_sdf, label_col="target", pred_col="prediction"):
    return {
        m: RegressionEvaluator(
            labelCol=label_col, predictionCol=pred_col, metricName=m
        ).evaluate(preds_sdf)
        for m in ["rmse", "mae", "r2"]
    }


def evaluate_regression_pd(y_true, y_pred):
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae":  float(mean_absolute_error(y_true, y_pred)),
        "r2":   float(r2_score(y_true, y_pred)),
    }


# ── Model comparison harness (identical to notebook Cell 8.2a) ───────────────

def train_and_compare_models(train_sdf, test_sdf, feature_cols,
                              label_col="target", primary_metric="rmse",
                              lower_is_better=True):
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="_features",
                                 handleInvalid="skip")

    candidates = {
        "LinearRegression": LinearRegression(
            featuresCol="_features", labelCol=label_col, maxIter=200),
        "RandomForest": RandomForestRegressor(
            featuresCol="_features", labelCol=label_col,
            numTrees=100, maxDepth=8, seed=42),
        "GBT": GBTRegressor(
            featuresCol="_features", labelCol=label_col,
            maxIter=80, maxDepth=5, stepSize=0.1, subsamplingRate=0.8, seed=42),
    }

    results, fitted_models, model_kind = [], {}, {}

    # Naive persistence baseline
    if "pm25_lag1" in test_sdf.columns:
        pers_preds = test_sdf.withColumn("prediction", F.col("pm25_lag1"))
        m = evaluate_regression(pers_preds, label_col)
        m["model"] = "Persistence (naive)"
        results.append(m)
        print(f"   Persistence (naive): RMSE={m['rmse']:.3f}  MAE={m['mae']:.3f}  R²={m['r2']:.4f}")

    for name, estimator in candidates.items():
        pipeline = Pipeline(stages=[assembler, estimator])
        print(f"🔄 Training {name}...")
        fitted = pipeline.fit(train_sdf)
        preds = fitted.transform(test_sdf)
        m = evaluate_regression(preds, label_col)
        m["model"] = name
        results.append(m)
        fitted_models[name] = fitted
        model_kind[name] = "spark"
        print(f"   {name}: RMSE={m['rmse']:.3f}  MAE={m['mae']:.3f}  R²={m['r2']:.4f}")

    # LightGBM / XGBoost (pandas/sklearn)
    if HAVE_LIGHTGBM or HAVE_XGBOOST:
        train_pdf = train_sdf.select(feature_cols + [label_col]).toPandas()
        test_pdf  = test_sdf.select(feature_cols + [label_col]).toPandas()
        X_train, y_train = train_pdf[feature_cols], train_pdf[label_col]
        X_test,  y_test  = test_pdf[feature_cols],  test_pdf[label_col]

        if HAVE_LIGHTGBM:
            print("🔄 Training LightGBM...")
            lgbm = LGBMRegressor(
                n_estimators=300, max_depth=5, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=-1,
            )
            lgbm.fit(X_train, y_train)
            m = evaluate_regression_pd(y_test, lgbm.predict(X_test))
            m["model"] = "LightGBM"
            results.append(m)
            fitted_models["LightGBM"] = lgbm
            model_kind["LightGBM"] = "sklearn"
            print(f"   LightGBM: RMSE={m['rmse']:.3f}  MAE={m['mae']:.3f}  R²={m['r2']:.4f}")

        if HAVE_XGBOOST:
            print("🔄 Training XGBoost...")
            xgb = XGBRegressor(
                n_estimators=300, max_depth=5, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, random_state=42,
                tree_method="hist", verbosity=0,
            )
            xgb.fit(X_train, y_train)
            m = evaluate_regression_pd(y_test, xgb.predict(X_test))
            m["model"] = "XGBoost"
            results.append(m)
            fitted_models["XGBoost"] = xgb
            model_kind["XGBoost"] = "sklearn"
            print(f"   XGBoost: RMSE={m['rmse']:.3f}  MAE={m['mae']:.3f}  R²={m['r2']:.4f}")

    results_df = (
        pd.DataFrame(results)[["model", "rmse", "mae", "r2"]]
        .sort_values(primary_metric, ascending=lower_is_better)
        .reset_index(drop=True)
    )
    results_df.to_csv(OUTPUTS_DIR / "model_comparison.csv", index=False)

    winner_name = results_df.iloc[0]["model"]
    print(f"\n🏆 Best model by {primary_metric.upper()}: {winner_name}")
    print(results_df.to_string(index=False))
    return (
        fitted_models.get(winner_name),
        model_kind.get(winner_name, "spark"),
        winner_name,
        results_df,
    )


def main():
    spark = get_spark()

    print("=" * 60)
    print("  PM2.5 Forecasting — Model Training & Export")
    print("=" * 60 + "\n")

    if not HAVE_LIGHTGBM:
        print("⚠️ lightgbm not installed — skipping (pip install lightgbm)")
    if not HAVE_XGBOOST:
        print("⚠️ xgboost not installed — skipping (pip install xgboost)")

    # Load data
    sdf = spark.read.parquet(str(CLEANED_PARQUET_DIR))
    sdf = sdf.withColumn("station_id", F.col("station_id").cast("string"))

    if "pm25" not in sdf.columns or sdf.filter(F.col("pm25").isNotNull()).count() == 0:
        print("⚠️ PM2.5 column not available — cannot train forecasting model.")
        return

    # Add time features
    sdf = sdf.withColumn("hour", F.hour("timestamp"))
    sdf = sdf.withColumn("dow", F.dayofweek("timestamp"))
    sdf = sdf.withColumn("month", F.month("timestamp"))
    sdf = sdf.withColumn("season",
        F.when(F.col("month").isin(12, 1, 2), "Winter")
         .when(F.col("month").isin(3, 4, 5), "Pre-Monsoon")
         .when(F.col("month").isin(6, 7, 8, 9), "Monsoon")
         .otherwise("Post-Monsoon")
    )

    # Pick station with most PM2.5 readings (same as notebook Cell 8.1)
    name_col = "station_name" if "station_name" in sdf.columns else "station_id"
    best_station = (
        sdf.filter(F.col("pm25").isNotNull())
        .groupBy("station_id", name_col)
        .count()
        .orderBy(F.desc("count"))
        .first()
    )
    print(f"📍 Forecasting station: {best_station[name_col]} ({best_station['count']:,} readings)")

    # Feature engineering (identical to notebook Cell 8.1)
    w_lag = Window.partitionBy("station_id").orderBy("timestamp")

    ml_sdf = (
        sdf
        .filter(
            (F.col("station_id") == best_station["station_id"]) &
            F.col("pm25").isNotNull()
        )
        .withColumn("pm25_lag1",   F.lag("pm25", 1).over(w_lag))
        .withColumn("pm25_lag24",  F.lag("pm25", 24).over(w_lag))
        .withColumn("pm25_roll24", F.avg("pm25").over(w_lag.rowsBetween(-23, 0)))
        .withColumn("target",      F.lead("pm25", 1).over(w_lag))
        .withColumn("season_enc",
            F.when(F.col("season") == "Winter",      0)
             .when(F.col("season") == "Pre-Monsoon", 1)
             .when(F.col("season") == "Monsoon",     2)
             .otherwise(3)
        )
    )

    # Weather column mapping — notebook uses temp_c/rh_pct/wind_speed_kmh/blh_m,
    # but the cleaned parquet from pyspark/ pipeline uses temperature/humidity/wind_speed.
    weather_map = {
        "temp_c": "temperature", "rh_pct": "humidity",
        "wind_speed_kmh": "wind_speed", "blh_m": "pressure",
    }
    for nb_col, pipe_col in weather_map.items():
        if nb_col not in ml_sdf.columns and pipe_col in ml_sdf.columns:
            ml_sdf = ml_sdf.withColumn(nb_col, F.col(pipe_col))

    FEATURE_COLS = [
        "pm25_lag1", "pm25_lag24", "pm25_roll24",
        "hour", "dow", "month", "season_enc",
        "temp_c", "rh_pct", "wind_speed_kmh", "blh_m",
    ]
    available_feats = [c for c in FEATURE_COLS if c in ml_sdf.columns]

    ml_sdf = ml_sdf.select(["timestamp", "target"] + available_feats).dropna()
    total_rows = ml_sdf.count()
    print(f"✅ ML dataset: {total_rows:,} rows  |  {len(available_feats)} features: {available_feats}")

    if total_rows < 100:
        print("⚠️ Not enough data for training. Aborting.")
        return

    # ── Chronological 80/20 split (same as notebook Cell 8.2b) ────────────────
    train_n = int(total_rows * 0.80)
    w_row = Window.orderBy("timestamp")
    ml_rn = ml_sdf.withColumn("_rn", F.row_number().over(w_row))
    train_sdf = ml_rn.filter(F.col("_rn") <= train_n).drop("_rn", "timestamp")
    test_sdf  = ml_rn.filter(F.col("_rn") > train_n)

    print(f"✅ Train: {train_sdf.count():,} rows | Test: {test_sdf.count():,} rows")

    # ── Train all models ──────────────────────────────────────────────────────
    best_model, best_model_kind, best_model_name, comparison_df = train_and_compare_models(
        train_sdf, test_sdf.drop("timestamp"), available_feats,
        label_col="target", primary_metric="rmse", lower_is_better=True,
    )

    if best_model is None:
        print("\n⚠️ Naive persistence beat every trained model.")
        metadata = {
            "model_name": "Persistence (naive)", "model_kind": "persistence",
            "features": [], "metrics": {"rmse": None, "mae": None, "r2": None},
        }
        with open(OUTPUTS_DIR / "model_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        return

    # ── Generate predictions ──────────────────────────────────────────────────
    if best_model_kind == "spark":
        test_preds_sdf = best_model.transform(test_sdf).orderBy("timestamp")
        rmse = RegressionEvaluator(labelCol="target", predictionCol="prediction", metricName="rmse").evaluate(test_preds_sdf)
        mae  = RegressionEvaluator(labelCol="target", predictionCol="prediction", metricName="mae").evaluate(test_preds_sdf)
        r2   = RegressionEvaluator(labelCol="target", predictionCol="prediction", metricName="r2").evaluate(test_preds_sdf)
        test_pd = test_preds_sdf.select("timestamp", "target", "prediction").toPandas()
    else:
        test_pd = test_sdf.orderBy("timestamp").toPandas()
        preds = best_model.predict(test_pd[available_feats])
        test_pd["prediction"] = preds
        rmse = float(np.sqrt(mean_squared_error(test_pd["target"], preds)))
        mae  = float(mean_absolute_error(test_pd["target"], preds))
        r2   = float(r2_score(test_pd["target"], preds))
        test_pd = test_pd[["timestamp", "target", "prediction"]]

    test_pd["timestamp"] = pd.to_datetime(test_pd["timestamp"], utc=True)
    test_pd.to_csv(OUTPUTS_DIR / "test_predictions.csv", index=False)

    # ── Save metrics ──────────────────────────────────────────────────────────
    pd.DataFrame([{
        "model": best_model_name, "rmse": round(rmse, 3),
        "mae": round(mae, 3), "r2": round(r2, 4),
        "n_train": train_n, "n_test": total_rows - train_n,
        "features": str(available_feats),
    }]).to_csv(OUTPUTS_DIR / "forecast_metrics.csv", index=False)

    # ── Save feature importance ───────────────────────────────────────────────
    importances = None
    if best_model_kind == "spark":
        stage = best_model.stages[-1]
        if hasattr(stage, "featureImportances"):
            importances = stage.featureImportances.toArray()
    else:
        importances = getattr(best_model, "feature_importances_", None)

    if importances is not None:
        feat_imp_df = pd.DataFrame({
            "feature": available_feats, "importance": importances
        }).sort_values("importance", ascending=False)
        feat_imp_df.to_csv(OUTPUTS_DIR / "feature_importance.csv", index=False)

    # ── Save model ────────────────────────────────────────────────────────────
    model_spark_path  = str(MODEL_DIR)
    model_sklearn_path = str(MODEL_SKLEARN_DIR)

    metadata = {
        "model_name": best_model_name,
        "model_kind": best_model_kind,
        "features": available_feats,
        "target": "target",
        "label": "Next-hour PM2.5 (µg/m³)",
        "metrics": {
            "rmse": round(rmse, 3),
            "mae": round(mae, 3),
            "r2": round(r2, 4),
        },
        "training_rows": train_n,
        "test_rows": total_rows - train_n,
    }

    if best_model_kind == "spark":
        if os.path.exists(model_spark_path):
            shutil.rmtree(model_spark_path)
        best_model.write().overwrite().save(model_spark_path)
        metadata["model_path"] = model_spark_path
        print(f"\n✅ Spark MLlib model saved → {model_spark_path}")
    else:
        import joblib
        os.makedirs(model_sklearn_path, exist_ok=True)
        sklearn_file = os.path.join(model_sklearn_path, "model.joblib")
        joblib.dump(best_model, sklearn_file)
        metadata["model_path"] = sklearn_file
        print(f"\n✅ sklearn model ({best_model_name}) saved → {sklearn_file}")

    with open(OUTPUTS_DIR / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"✅ Model metadata saved → outputs/model_metadata.json")
    print(f"   Model: {best_model_name} ({best_model_kind})")
    print(f"   RMSE={rmse:.3f}  MAE={mae:.3f}  R²={r2:.4f}")

    # ── Visualization ─────────────────────────────────────────────────────────
    plot_pd = test_pd.head(2000)

    # Forecast vs actual
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(plot_pd["timestamp"], plot_pd["target"], linewidth=1.2, label="Actual PM2.5")
    ax.plot(plot_pd["timestamp"], plot_pd["prediction"], linewidth=1.2, linestyle="--",
            label=f"{best_model_name} prediction")
    ax.axhline(15, linestyle=":", linewidth=1.2, label="WHO 24-h = 15 µg/m³")
    ax.set_title(f"PM2.5 Forecast vs Actual — {best_model_name}\n"
                 f"RMSE={rmse:.2f} | MAE={mae:.2f} | R²={r2:.3f}")
    ax.set_xlabel("Date"); ax.set_ylabel("PM2.5 (µg/m³)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "forecast_vs_actual.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Feature importance
    if importances is not None:
        fig, ax = plt.subplots(figsize=(10, max(4, len(available_feats) * 0.45)))
        feat_imp_sorted = feat_imp_df.sort_values("importance", ascending=True)
        ax.barh(feat_imp_sorted["feature"], feat_imp_sorted["importance"])
        ax.set_title(f"Feature Importance — {best_model_name}")
        ax.set_xlabel("Importance")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "feature_importance.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # Model comparison
    fig, ax = plt.subplots(figsize=(9, 5))
    plot_cmp = comparison_df.sort_values("rmse", ascending=True)
    ax.barh(plot_cmp["model"], plot_cmp["rmse"])
    ax.set_title("Next-Hour PM2.5 Model Comparison")
    ax.set_xlabel("Test RMSE (µg/m³) — lower is better")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Actual vs predicted scatter
    sample = plot_pd.sample(min(3000, len(plot_pd)), random_state=42)
    max_val = float(max(sample["target"].max(), sample["prediction"].max()))
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(sample["target"], sample["prediction"], alpha=0.35, s=16)
    ax.plot([0, max_val], [0, max_val], linestyle="--", linewidth=1.2)
    ax.set_title(f"Actual vs Predicted PM2.5 — {best_model_name} (R²={r2:.3f})")
    ax.set_xlabel("Actual PM2.5 (µg/m³)")
    ax.set_ylabel("Predicted PM2.5 (µg/m³)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "actual_vs_predicted.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\n{'=' * 60}")
    print(f"  Forecasting complete. Model + predictions exported.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
