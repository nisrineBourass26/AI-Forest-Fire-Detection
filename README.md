# AI-Forest-Fire-Detection
AI-Forest-Fire-Detection PFE 

# 🌲 **AI-Based Forest Fire Detection Using NASA Satellite Data**

### **Final Year Project –EST / AI Specialty – 2025–2026**
## 🚀 **Project Overview**

This project aims to build an **AI-powered early detection and prediction system** for forest fires using **NASA FIRMS satellite data** (MODIS & VIIRS).
By combining **data engineering, geospatial analysis, and machine learning**, the system detects fire hotspots, analyzes their evolution, and predicts future risk levels.

This transforms forest fire management from **reactive** to **proactive**, helping protect biodiversity, property, and communities.

---

## 👩‍💻 **Team Members**

| Name                | Role                                                      |
| ------------------- | --------------------------------------------------------- |
| **Nisrine Bourass** | Data preprocessing, data engineering, feature preparation |
| **Manar Nebach**    | AI modeling, prediction system                            |
|   Nisrine Bourass   | Web dashboard, visualization, alerts system               |
| **Manar Nebach**    | Embedded system Rassberry Pico                            |
---

## 🎯 **Objectives**

### **General Objective**

Develop an AI system that detects and predicts forest fires in near real-time using NASA satellite data.

### **Specific Objectives**

* Collect thermal data from **MODIS & VIIRS** (NASA FIRMS)
* Clean, preprocess, and merge satellite datasets
* Perform spatial & temporal fire analysis
* Detect hotspots and propagation patterns
* Train ML/DL models to **predict fire risk**
* Build a **web dashboard** to visualize fires & predictions
* Integrate an early warning alert system

---

## 🛰 **Data Sources**

* **NASA FIRMS NRT3 / NRT4**

  * MODIS: 1 km resolution
  * VIIRS: 375 m resolution
  * Provides real-time thermal anomalies, fire radiative power, confidence levels, etc.

---

## 🧰 **Tech Stack**

### **Data & Geospatial**

* Python (Pandas, NumPy, GeoPandas)
* PostgreSQL + PostGIS
* QGIS (mapping & validation)

### **Machine Learning**

* Scikit-learn
* XGBoost
* LSTM / ConvLSTM (Time-Series prediction)

### **Web Application**

* Streamlit / Dash
* Leaflet.js or Mapbox (interactive maps)

---

## 🗂 **Repository Structure**

```text
AI-Forest-Fire-Detection/
│
├── data/
│   ├── raw/           # Raw NASA FIRMS data
│   ├── cleaned/       # Cleaned/processed datasets
│
├── scripts/
│   ├── data_download/ # Scripts to fetch data from NASA API
│   ├── preprocessing/ # Cleaning, merging, feature engineering
│
├── models/
│   ├── training/      # Model training notebooks/scripts
│   ├── saved_models/  # Final models
│
├── web/
│   ├── dashboard/     # Frontend code (Streamlit/Dash)
│
└── docs/
    ├── reports/       # PFE documentation
    └── diagrams/      # Architecture, workflow, UML
```

---

## 📌 **Main Features**

### ✔ **1. Satellite Data Collection**

Automatic retrieval of MODIS & VIIRS fire data using NASA FIRMS API.

### ✔ **2. Data Preprocessing**

* Cleaning
* Filtering low-confidence detections
* Merging MODIS + VIIRS
* Feature engineering (brightness, FRP categories, fire frequency…)

### ✔ **3. Spatio-Temporal Analysis**

* Hotspot detection (DBSCAN)
* Fire propagation tracking
* Density maps
* Regional stability index

### ✔ **4. Prediction Models**

* Fire risk classification
* Time-series prediction (LSTM / ConvLSTM)

### ✔ **5. Web Dashboard**

* Real-time fire detection map
* Predictions visualization
* Alerts system

---

## 📅 **Timeline**

The development is divided into 9 phases:

1. Technical preparation
2. Data acquisition
3. Preprocessing
4. Spatio-temporal analysis
5. Modeling
6. Dashboard
7. Robotic simulation (optional)
8. Testing & validation
9. Final report writing

---

