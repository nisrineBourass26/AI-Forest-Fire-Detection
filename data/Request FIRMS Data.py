import requests
import pandas as pd
from io import StringIO

# -----------------------------
# CONFIG
# -----------------------------
API_KEY = "8d9da6178ccfc604031991e562289047" 
PRODUCT = "VIIRS_SNPP_NRT"  # or MODIS, VIIRS_NOAA20, etc.
DAYS = 3  # number of last days (1–7)
AREA = "world"  # or bounding box "(-13,27,-1,36)"
OUTPUT_CSV = "fires_data.csv"

# -----------------------------
# BUILD URL
# -----------------------------
url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{API_KEY}/{PRODUCT}/{AREA}/{DAYS}"

print("Requesting:", url)

# -----------------------------
# REQUEST DATA
# -----------------------------
response = requests.get(url)

if response.status_code != 200:
    print("Error:", response.text)
    exit()

print("Data downloaded successfully.")

# -----------------------------
# CONVERT CSV TEXT → DATAFRAME
# -----------------------------
csv_text = response.text
df = pd.read_csv(StringIO(csv_text))

print("Number of rows downloaded:", len(df))

# -----------------------------
# SAVE TO CSV
# -----------------------------
df.to_csv(OUTPUT_CSV, index=False)
print(f"CSV saved as {OUTPUT_CSV}")