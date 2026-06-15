"""
pipeline_nasa_firms.py
-----------------------
Step 1 of the wildfire detection pipeline.

Downloads the latest 1-day global wildfire data from NASA FIRMS
(VIIRS SNPP, VIIRS NOAA-20, MODIS), merges the sources, applies
feature engineering, and saves two outputs:

  data/latest_raw_firms.csv    — merged raw download
  data/latest_clean_firms.csv  — feature-engineered clean data

Old timestamped raw/clean files from previous runs are deleted
automatically so the data/ folder stays clean.
"""

import os
import sys
import glob

import numpy as np
import pandas as pd
import requests
from io import StringIO
from datetime import datetime

ROOT_DIR   = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)

from config import (
    DATA_DIR,
    LATEST_RAW_FIRMS_PATH,
    LATEST_CLEAN_FIRMS_PATH,
)

os.makedirs(DATA_DIR, exist_ok=True)

# ============================================================
# NASA FIRMS CONFIG
# ============================================================

MAP_KEY = "8d9da6178ccfc604031991e562289047"
AREA    = "world"
DAYS    = 1

SOURCES = [
    "VIIRS_SNPP_NRT",
    "VIIRS_NOAA20_NRT",
    "MODIS_NRT",
]


# ============================================================
# CLEANUP — delete previous timestamped files
# ============================================================

def delete_old_timestamped_files():
    """
    Removes all firms_raw_* and firms_clean_* timestamped CSV files
    from data/ so only the latest_* files remain after each run.
    """
    patterns = [
        os.path.join(DATA_DIR, "firms_raw_world_1day_*.csv"),
        os.path.join(DATA_DIR, "firms_clean_world_1day_*.csv"),
    ]
    deleted = 0
    for pattern in patterns:
        for path in glob.glob(pattern):
            try:
                os.remove(path)
                deleted += 1
            except OSError as e:
                print(f"  Warning: could not delete {path}: {e}")
    if deleted:
        print(f"  Deleted {deleted} old timestamped file(s) from data/")
    else:
        print("  No old timestamped files to delete.")


# ============================================================
# DOWNLOAD
# ============================================================

def download_firms_data(source):
    url = (
        f"https://firms.modaps.eosdis.nasa.gov/api/area/csv"
        f"/{MAP_KEY}/{source}/{AREA}/{DAYS}"
    )
    print(f"\n  Downloading {source}...")

    try:
        response = requests.get(url, timeout=120)
    except requests.exceptions.RequestException as e:
        print(f"  Request failed for {source}: {e}")
        return pd.DataFrame()

    if response.status_code != 200:
        print(f"  HTTP {response.status_code} for {source}: {response.text[:200]}")
        return pd.DataFrame()

    try:
        df = pd.read_csv(StringIO(response.text))
    except Exception as e:
        print(f"  Could not parse CSV for {source}: {e}")
        return pd.DataFrame()

    if df.empty:
        print(f"  {source}: no detections returned.")
        return pd.DataFrame()

    df["source"] = source
    print(f"  {source}: {len(df):,} detections")
    return df


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def ensure_column(df, col, default=np.nan):
    if col not in df.columns:
        df[col] = default
    return df


def clean_and_engineer_features(df):
    df = df.copy()

    if df.empty:
        print("No data to clean.")
        return df

    # Ensure all expected columns exist
    for col in [
        "latitude", "longitude", "bright_ti4", "bright_ti5",
        "brightness", "bright_t31", "scan", "track", "frp",
        "acq_date", "acq_time", "daynight", "confidence",
        "satellite", "instrument", "version", "source",
    ]:
        df = ensure_column(df, col)

    # Numeric coercion
    for col in ["latitude", "longitude", "bright_ti4", "bright_ti5",
                "brightness", "bright_t31", "scan", "track", "frp"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Unified brightness columns (VIIRS uses ti4/ti5; MODIS uses brightness/bright_t31)
    df["brightness_main"]      = df["bright_ti4"].fillna(df["brightness"])
    df["brightness_secondary"] = df["bright_ti5"].fillna(df["bright_t31"])
    df["brightness"]           = df["brightness"].fillna(df["bright_ti4"])
    df["bright_t31"]           = df["bright_t31"].fillna(df["bright_ti5"])

    # Datetime parsing
    df["acq_date"] = pd.to_datetime(df["acq_date"], errors="coerce")
    df["acq_time"] = (
        df["acq_time"].astype(str)
        .str.replace(".0", "", regex=False)
        .str.zfill(4)
    )
    df["hour"]   = pd.to_numeric(df["acq_time"].str[:2], errors="coerce").fillna(0).astype(int)
    df["minute"] = pd.to_numeric(df["acq_time"].str[2:], errors="coerce").fillna(0).astype(int)
    df["datetime"] = pd.to_datetime(
        df["acq_date"].dt.strftime("%Y-%m-%d")
        + " "
        + df["hour"].astype(str).str.zfill(2)
        + ":"
        + df["minute"].astype(str).str.zfill(2),
        errors="coerce",
    )

    # Temporal features
    df["year"]        = df["datetime"].dt.year
    df["month"]       = df["datetime"].dt.month
    df["day"]         = df["datetime"].dt.day
    df["day_of_year"] = df["datetime"].dt.dayofyear
    df["week_of_year"] = df["datetime"].dt.isocalendar().week.astype("Int64")

    # Day/night flag
    df["is_day"]   = df["daynight"].astype(str).str.upper().eq("D").astype(int)
    df["is_night"] = df["daynight"].astype(str).str.upper().eq("N").astype(int)

    # Confidence numeric
    _confidence_map = {
        "l": 30, "low": 30, "n": 60, "nominal": 60,
        "medium": 60, "h": 90, "high": 90,
    }
    conf = df["confidence"].astype(str).str.lower().str.strip()
    df["confidence_num"] = conf.map(_confidence_map).fillna(
        pd.to_numeric(conf, errors="coerce")
    )
    median_conf = df["confidence_num"].median()
    df["confidence_num"] = df["confidence_num"].fillna(
        median_conf if pd.notna(median_conf) else 0
    )
    df["confidence_level"] = df["confidence_num"]

    # FRP
    df["frp"]     = df["frp"].fillna(0)
    df["frp_log"] = np.log1p(df["frp"].clip(lower=0))

    # Scan / track / area
    df["scan"]  = df["scan"].fillna(0)
    df["track"] = df["track"].fillna(0)
    area = (df["scan"] * df["track"]).replace(0, np.nan)
    df["frp_per_km2"] = (df["frp"] / area).replace([np.inf, -np.inf], np.nan).fillna(0)

    # Brightness anomalies
    df["brightness_anomaly"] = (df["brightness_main"] - df["brightness_main"].mean()).fillna(0)
    df["bright_t31_anomaly"] = (df["brightness_secondary"] - df["brightness_secondary"].mean()).fillna(0)

    # Satellite encoding
    _sat_map = {"N": 0, "1": 1, "A": 2, "T": 3}
    df["satellite_code"] = (
        df["satellite"].astype(str).map(_sat_map).fillna(-1).astype(int)
    )

    # Season (0=winter, 1=spring, 2=summer, 3=autumn)
    _season_map = {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1,
                   6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}
    df["season"] = df["month"].map(_season_map).fillna(-1).astype(int)

    # Fire intensity category
    df["frp_category"] = pd.cut(
        df["frp"],
        bins=[-1, 10, 50, 150, float("inf")],
        labels=["Low", "Moderate", "High", "Extreme"],
    )

    # Cyclic hour features
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    # Interaction features
    df["lat_lon_interaction"]    = df["latitude"] * df["longitude"]
    df["brightness_scan_ratio"]  = df["brightness"] / (df["scan"] + 1e-6)
    df["track_scan_ratio"]       = df["track"] / (df["scan"] + 1e-6)

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["latitude", "longitude", "datetime"])

    return df


# ============================================================
# MAIN
# ============================================================

def main():
    print("\n" + "=" * 70)
    print("NASA FIRMS WORLD 1-DAY DOWNLOAD + PREPROCESSING")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Delete old timestamped files from previous runs
    print("\nCleaning up old data files...")
    delete_old_timestamped_files()

    # Download from all sources
    print("\nDownloading from NASA FIRMS...")
    all_data = [download_firms_data(src) for src in SOURCES]
    all_data = [d for d in all_data if not d.empty]

    if not all_data:
        raise ValueError("No FIRMS data downloaded from any source. Check your API key and network.")

    df = pd.concat(all_data, ignore_index=True)
    print(f"\nMerged raw shape: {df.shape}")

    # Save raw
    df.to_csv(LATEST_RAW_FIRMS_PATH, index=False)
    print(f"Raw data saved  : {LATEST_RAW_FIRMS_PATH}")

    # Clean + engineer features
    print("\nApplying feature engineering...")
    df_clean = clean_and_engineer_features(df)
    print(f"Clean shape     : {df_clean.shape}")

    # Save clean
    df_clean.to_csv(LATEST_CLEAN_FIRMS_PATH, index=False)
    print(f"Clean data saved: {LATEST_CLEAN_FIRMS_PATH}")

    print(f"\nNASA FIRMS step completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()