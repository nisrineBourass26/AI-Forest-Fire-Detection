import os
import glob
import pandas as pd


# ============================================================
# PATHS
# ============================================================

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPTS_DIR)

INPUT_DIR = os.path.join(ROOT_DIR, "data", "forecast_recent_raw")
OUTPUT_PATH = os.path.join(ROOT_DIR, "data", "forecast_context.csv")

CONTEXT_DAYS_TO_KEEP = 90


# ============================================================
# LOAD FILES
# ============================================================

csv_files = glob.glob(os.path.join(INPUT_DIR, "*.csv"))

if not csv_files:
    raise FileNotFoundError(f"No CSV files found in: {INPUT_DIR}")

print(f"Found {len(csv_files)} recent forecast raw file(s):")
for f in csv_files:
    print(" -", os.path.basename(f))


# ============================================================
# PROCESS FILES
# ============================================================

daily_parts = []

confidence_map = {
    "l": 30,
    "low": 30,
    "n": 60,
    "nominal": 60,
    "medium": 60,
    "h": 90,
    "high": 90,
}

for file in csv_files:
    print(f"\nReading: {os.path.basename(file)}")

    df = pd.read_csv(file, low_memory=False)

    required_cols = ["acq_date", "frp", "confidence", "daynight"]
    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise ValueError(f"{file} is missing columns: {missing}")

    df["ds"] = pd.to_datetime(df["acq_date"], errors="coerce")
    df["frp"] = pd.to_numeric(df["frp"], errors="coerce").fillna(0)

    conf = df["confidence"].astype(str).str.lower().str.strip()
    df["confidence_num"] = conf.map(confidence_map)
    df["confidence_num"] = df["confidence_num"].fillna(
        pd.to_numeric(conf, errors="coerce")
    )
    df["confidence_num"] = df["confidence_num"].fillna(0)

    df["is_night"] = (
        df["daynight"]
        .astype(str)
        .str.upper()
        .eq("N")
        .astype(int)
    )

    df = df.dropna(subset=["ds"])

    daily = (
        df.groupby("ds")
        .agg(
            y=("frp", "size"),
            avg_frp=("frp", "mean"),
            max_frp=("frp", "max"),
            avg_confidence=("confidence_num", "mean"),
            night_fire_count=("is_night", "sum"),
        )
        .reset_index()
    )

    daily_parts.append(daily)

if not daily_parts:
    raise ValueError("No daily data was created from the recent raw files.")


# ============================================================
# COMBINE + KEEP LAST 90 DAYS
# ============================================================

combined = pd.concat(daily_parts, ignore_index=True)

context = (
    combined.groupby("ds")
    .agg(
        y=("y", "sum"),
        avg_frp=("avg_frp", "mean"),
        max_frp=("max_frp", "max"),
        avg_confidence=("avg_confidence", "mean"),
        night_fire_count=("night_fire_count", "sum"),
    )
    .reset_index()
)

context = context.sort_values("ds")

# Fill missing dates between first and last date
full_dates = pd.date_range(context["ds"].min(), context["ds"].max(), freq="D")

context = context.set_index("ds").reindex(full_dates).reset_index()
context = context.rename(columns={"index": "ds"})

context["y"] = context["y"].fillna(0).astype(int)
context["avg_frp"] = context["avg_frp"].fillna(0)
context["max_frp"] = context["max_frp"].fillna(0)
context["avg_confidence"] = context["avg_confidence"].fillna(0)
context["night_fire_count"] = context["night_fire_count"].fillna(0).astype(int)

# Keep only the latest 90 days
context = context.tail(CONTEXT_DAYS_TO_KEEP).copy()

context.to_csv(OUTPUT_PATH, index=False)


# ============================================================
# SUMMARY
# ============================================================

print("\nforecast_context.csv created successfully.")
print("Saved to:", OUTPUT_PATH)

print("\nPreview:")
print(context.head())

print("\nLast rows:")
print(context.tail())

print("\nSummary:")
print("Rows:", len(context))
print("Date range:", context["ds"].min().date(), "→", context["ds"].max().date())
print("Average fires/day:", round(context["y"].mean(), 2))
print("Max fires/day:", int(context["y"].max()))