"""
run_pipeline.py
----------------
Master pipeline runner for the AI-Based Forest Fire Detection system.

Steps
-----
  1. Download + preprocess latest NASA FIRMS global data
  2. Run AI predictions  (DBSCAN · XGBoost · Random Forest)
  3. Update forecast context and run Prophet + LSTM 7-day forecast
  4. Insert predictions into PostgreSQL/PostGIS

Usage
-----
  python run_pipeline.py
"""

import os
import sys
import subprocess
from datetime import datetime

# run_pipeline.py lives in the project root — ROOT_DIR is its own directory
ROOT_DIR   = os.path.dirname(os.path.abspath(__file__))
SCRIPT_DIR = os.path.join(ROOT_DIR, "scripts")


def run_step(label, script_path):
    """Run a Python script as a subprocess. Raises RuntimeError on failure."""
    print("\n" + "=" * 70)
    print(label)
    print("=" * 70)

    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Script not found: {script_path}")

    result = subprocess.run(
        [sys.executable, script_path],
        cwd=ROOT_DIR,
    )

    if result.returncode != 0:
        raise RuntimeError(f"{label} failed (exit code {result.returncode}).")

    print(f"\n{label} — completed successfully.")


def main():
    start = datetime.now()

    print("\n" + "=" * 70)
    print("AI-BASED FOREST FIRE DETECTION — WORLD 1-DAY PIPELINE")
    print(f"Started at: {start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Project root: {ROOT_DIR}")
    print("=" * 70)

    run_step(
        "STEP 1 — Download and preprocess NASA FIRMS global data",
        os.path.join(ROOT_DIR, "pipeline_nasa_firms.py"),
    )

    run_step(
        "STEP 2 — AI predictions (DBSCAN · XGBoost · Random Forest)",
        os.path.join(SCRIPT_DIR, "prediction_pipeline.py"),
    )

    run_step(
        "STEP 3 — Update forecast context + Prophet + LSTM 7-day forecast",
        os.path.join(SCRIPT_DIR, "update_context_and_forecast.py"),
    )

    run_step(
        "STEP 4 — Insert predictions into PostgreSQL/PostGIS",
        os.path.join(SCRIPT_DIR, "insert_predictions_to_postgres.py"),
    )

    end     = datetime.now()
    elapsed = int((end - start).total_seconds())

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print(f"Finished at : {end.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Elapsed     : {elapsed // 60}m {elapsed % 60}s")
    print("=" * 70)
    print("\nGenerated outputs:")
    print("  data/latest_raw_firms.csv")
    print("  data/latest_clean_firms.csv")
    print("  data/latest_predictions.csv")
    print("  data/latest_forecast.csv")
    print("  PostgreSQL table updated: fire_predictions")


if __name__ == "__main__":
    main()