import os
import glob
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_DIR = os.path.join(ROOT_DIR, "data", "global_historical")
OUTPUT_PATH = os.path.join(ROOT_DIR, "data", "prophetData_global.csv")

CHUNK_SIZE = 200_000

csv_files = glob.glob(os.path.join(INPUT_DIR, "*.csv"))

if not csv_files:
    raise FileNotFoundError(f"No CSV files found in {INPUT_DIR}")

print(f"Found {len(csv_files)} files:")
for f in csv_files:
    print(" -", os.path.basename(f))

daily_parts = []

needed = ["acq_date", "frp", "confidence", "daynight"]

confidence_map = {
    "low": 30,
    "nominal": 60,
    "medium": 60,
    "high": 90,
}

for file in csv_files:
    print(f"\nProcessing {os.path.basename(file)}...")

    chunk_id = 0

    for chunk in pd.read_csv(
        file,
        usecols=lambda c: c in needed,
        chunksize=CHUNK_SIZE,
        low_memory=False,
    ):
        chunk_id += 1
        print(f"  chunk {chunk_id} | rows: {len(chunk)}")

        missing = [col for col in needed if col not in chunk.columns]
        if missing:
            raise ValueError(f"{file} is missing columns: {missing}")

        chunk["acq_date"] = pd.to_datetime(chunk["acq_date"], errors="coerce")
        chunk["frp"] = pd.to_numeric(chunk["frp"], errors="coerce")

        conf = chunk["confidence"].astype(str).str.lower().str.strip()
        chunk["confidence_num"] = conf.map(confidence_map)
        chunk["confidence_num"] = chunk["confidence_num"].fillna(
            pd.to_numeric(conf, errors="coerce")
        )

        chunk["is_night"] = (
            chunk["daynight"]
            .astype(str)
            .str.upper()
            .eq("N")
            .astype(int)
        )

        chunk = chunk.dropna(subset=["acq_date", "frp"])

        daily_chunk = (
            chunk
            .groupby("acq_date")
            .agg(
                y=("frp", "size"),
                frp_sum=("frp", "sum"),
                max_frp=("frp", "max"),
                confidence_sum=("confidence_num", "sum"),
                confidence_count=("confidence_num", "count"),
                night_fire_count=("is_night", "sum"),
            )
            .reset_index()
        )

        daily_parts.append(daily_chunk)

print("\nCombining daily summaries...")

daily_all = pd.concat(daily_parts, ignore_index=True)

daily = (
    daily_all
    .groupby("acq_date")
    .agg(
        y=("y", "sum"),
        frp_sum=("frp_sum", "sum"),
        max_frp=("max_frp", "max"),
        confidence_sum=("confidence_sum", "sum"),
        confidence_count=("confidence_count", "sum"),
        night_fire_count=("night_fire_count", "sum"),
    )
    .reset_index()
)

daily["avg_frp"] = daily["frp_sum"] / daily["y"]
daily["avg_confidence"] = daily["confidence_sum"] / daily["confidence_count"]

daily = daily.drop(columns=["frp_sum", "confidence_sum", "confidence_count"])

daily = daily.rename(columns={"acq_date": "ds"})
daily = daily.sort_values("ds")

full_dates = pd.date_range(daily["ds"].min(), daily["ds"].max(), freq="D")

daily = daily.set_index("ds").reindex(full_dates).reset_index()
daily = daily.rename(columns={"index": "ds"})

daily["y"] = daily["y"].fillna(0).astype(int)
daily["avg_frp"] = daily["avg_frp"].fillna(0)
daily["max_frp"] = daily["max_frp"].fillna(0)
daily["avg_confidence"] = daily["avg_confidence"].fillna(0)
daily["night_fire_count"] = daily["night_fire_count"].fillna(0).astype(int)

daily.to_csv(OUTPUT_PATH, index=False)

print("\nSaved final global forecasting dataset to:")
print(OUTPUT_PATH)

print("\nPreview:")
print(daily.head())

print("\nDate range:")
print(daily["ds"].min(), "→", daily["ds"].max())

print("\nRows:", len(daily))
print("Average fires/day:", daily["y"].mean())
print("Max fires/day:", daily["y"].max())