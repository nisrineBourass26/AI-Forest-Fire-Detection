import os
import pandas as pd
from sqlalchemy import create_engine, text

# -----------------------------------------
# 1. CONFIG
# -----------------------------------------
DB_USER = "postgres"
DB_PASSWORD = "postgres123"
DB_HOST = "localhost"
DB_PORT = "5433"
DB_NAME = "forest_fire_pfe"

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_FILE = os.path.join(ROOT_DIR, "data", "latest_predictions.csv")

TABLE_NAME = "fire_predictions"

# -----------------------------------------
# 2. READ PREDICTION CSV
# -----------------------------------------
print("Loading prediction CSV...")

if not os.path.exists(CSV_FILE):
    raise FileNotFoundError(f"Prediction file not found: {CSV_FILE}")

df = pd.read_csv(CSV_FILE, low_memory=False)

print(f"Loaded {len(df)} rows.")

columns_to_keep = [
    "latitude",
    "longitude",
    "datetime",
    "satellite",
    "instrument",
    "source",
    "confidence",
    "daynight",
    "frp",
    "predicted_frp",
    "predicted_risk",
    "risk_confidence",
    "cluster_id",
    "prediction_time",
    "prediction_scope",
    "prediction_period",
    "source_clean_file",
]

available_columns = [col for col in columns_to_keep if col in df.columns]
df = df[available_columns].copy()

required_columns = [
    "latitude",
    "longitude",
    "datetime",
    "frp",
    "predicted_frp",
    "predicted_risk",
    "risk_confidence",
    "cluster_id",
    "prediction_time",
]

missing_required = [col for col in required_columns if col not in df.columns]

if missing_required:
    raise ValueError(f"Missing required columns in latest_predictions.csv: {missing_required}")

df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
df["frp"] = pd.to_numeric(df["frp"], errors="coerce")
df["predicted_frp"] = pd.to_numeric(df["predicted_frp"], errors="coerce")
df["risk_confidence"] = pd.to_numeric(df["risk_confidence"], errors="coerce")
df["cluster_id"] = pd.to_numeric(df["cluster_id"], errors="coerce").fillna(-1).astype(int)

df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
df["prediction_time"] = pd.to_datetime(df["prediction_time"], errors="coerce")

df = df.dropna(subset=["latitude", "longitude", "datetime"])

print(f"Rows after cleaning for DB: {len(df)}")

# -----------------------------------------
# 3. CONNECT TO POSTGRESQL
# -----------------------------------------
print("Connecting to PostgreSQL...")

engine = create_engine(
    f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

with engine.begin() as conn:
    print("Connected successfully.")

    # -----------------------------------------
    # 4. ENABLE POSTGIS
    # -----------------------------------------
    print("Checking PostGIS extension...")
    conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis;"))

    # -----------------------------------------
    # 5. CLEAR OLD LATEST PREDICTIONS
    # -----------------------------------------
    print(f"Clearing old data from {TABLE_NAME}...")

    conn.execute(text(f"DROP TABLE IF EXISTS {TABLE_NAME};"))

# -----------------------------------------
# 6. INSERT DATA
# -----------------------------------------
print(f"Inserting latest prediction batch into {TABLE_NAME} table...")

df.to_sql(
    TABLE_NAME,
    engine,
    if_exists="replace",
    index=False
)

print("Data inserted successfully.")

# -----------------------------------------
# 7. CREATE GEOMETRY COLUMN + INDEXES
# -----------------------------------------
with engine.begin() as conn:
    print("Creating geometry column...")

    conn.execute(text(f"""
    ALTER TABLE {TABLE_NAME}
    ADD COLUMN geom geometry(Point, 4326);
    """))

    print("Updating geometry values...")

    conn.execute(text(f"""
    UPDATE {TABLE_NAME}
    SET geom = ST_SetSRID(ST_MakePoint(longitude, latitude), 4326);
    """))

    print("Creating indexes...")

    conn.execute(text(f"""
    CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_geom
    ON {TABLE_NAME}
    USING GIST (geom);
    """))

    conn.execute(text(f"""
    CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_risk
    ON {TABLE_NAME} (predicted_risk);
    """))

    conn.execute(text(f"""
    CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_datetime
    ON {TABLE_NAME} (datetime);
    """))

    conn.execute(text(f"""
    CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_cluster
    ON {TABLE_NAME} (cluster_id);
    """))

print("All latest world 1-day predictions successfully inserted into PostgreSQL/PostGIS.")