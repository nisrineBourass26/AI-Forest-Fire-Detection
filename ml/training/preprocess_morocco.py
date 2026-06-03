import os
import pandas as pd
import numpy as np

# ============================================================
# PATHS
# ============================================================

RAW_PATH = r"C:\Users\Lenovo\Desktop\Projets\PFE\data\fire_archive_SV-C2_755060.csv"
OUTPUT_DIR = r"C:\Users\Lenovo\Desktop\Projets\PFE\data"

os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_DAILY = os.path.join(OUTPUT_DIR, "prophetData_morocco.csv")
OUTPUT_CLEAN = os.path.join(OUTPUT_DIR, "fires_morocco_clean.csv")


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(RAW_PATH)

print("[INFO] Raw shape:", df.shape)
print("[INFO] Columns:", df.columns.tolist())


# ============================================================
# BASIC CLEANING
# ============================================================

df["acq_date"] = pd.to_datetime(df["acq_date"], errors="coerce")
df = df.dropna(subset=["acq_date", "latitude", "longitude", "frp"])

df["frp"] = pd.to_numeric(df["frp"], errors="coerce")
df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")

df = df.dropna(subset=["frp"])


# ============================================================
# MOROCCO SAFETY FILTER
# ============================================================

df = df[
    (df["latitude"] >= 21) & (df["latitude"] <= 36.5) &
    (df["longitude"] >= -17.5) & (df["longitude"] <= -0.5)
].copy()

print("[INFO] After Morocco coordinate filter:", df.shape)


# ============================================================
# REMOVE EXACT DUPLICATES
# ============================================================

dedup_cols = [
    "latitude",
    "longitude",
    "acq_date",
    "acq_time",
    "satellite",
    "instrument",
    "frp"
]

df = df.drop_duplicates(subset=dedup_cols).copy()

print("[INFO] After duplicate removal:", df.shape)


# ============================================================
# FEATURE CLEANING
# ============================================================

df["is_night"] = (df["daynight"] == "N").astype(int)

df["confidence"] = df["confidence"].fillna(df["confidence"].median())
df["frp"] = df["frp"].clip(lower=0)

df["year"] = df["acq_date"].dt.year
df["month"] = df["acq_date"].dt.month
df["day_of_year"] = df["acq_date"].dt.dayofyear


# ============================================================
# SAVE CLEAN DETECTION DATA
# ============================================================

df.to_csv(OUTPUT_CLEAN, index=False)
print("[INFO] Clean detection file saved:", OUTPUT_CLEAN)


# ============================================================
# DAILY AGGREGATION FOR PROPHET + LSTM
# ============================================================

daily = df.groupby("acq_date").agg(
    y=("frp", "count"),                     # daily fire detections
    avg_frp=("frp", "mean"),
    max_frp=("frp", "max"),
    total_frp=("frp", "sum"),
    avg_confidence=("confidence", "mean"),
    night_fire_count=("is_night", "sum"),
).reset_index()

daily = daily.rename(columns={"acq_date": "ds"})


# ============================================================
# FILL MISSING DAYS
# ============================================================

full_dates = pd.date_range(
    start=daily["ds"].min(),
    end=daily["ds"].max(),
    freq="D"
)

daily = daily.set_index("ds").reindex(full_dates).reset_index()
daily = daily.rename(columns={"index": "ds"})

# Days without detections = 0 fires
daily["y"] = daily["y"].fillna(0)

# For FRP/confidence features:
# If no fires, values should be 0
for col in ["avg_frp", "max_frp", "total_frp", "avg_confidence", "night_fire_count"]:
    daily[col] = daily[col].fillna(0)


# ============================================================
# EXTRA TIME FEATURES
# ============================================================

daily["year"] = daily["ds"].dt.year
daily["month"] = daily["ds"].dt.month
daily["day_of_year"] = daily["ds"].dt.dayofyear
daily["week_of_year"] = daily["ds"].dt.isocalendar().week.astype(int)

daily["day_sin"] = np.sin(2 * np.pi * daily["day_of_year"] / 365.25)
daily["day_cos"] = np.cos(2 * np.pi * daily["day_of_year"] / 365.25)
daily["month_sin"] = np.sin(2 * np.pi * daily["month"] / 12)
daily["month_cos"] = np.cos(2 * np.pi * daily["month"] / 12)


# ============================================================
# SAVE FINAL DAILY FILE
# ============================================================

daily.to_csv(OUTPUT_DAILY, index=False)

print("\n[DONE]")
print("Daily forecasting file saved:", OUTPUT_DAILY)
print("Rows:", len(daily))
print("Date range:", daily["ds"].min().date(), "→", daily["ds"].max().date())
print("\nTarget y statistics:")
print(daily["y"].describe())

print("\nFirst rows:")
print(daily.head())

print("\nLast rows:")
print(daily.tail())