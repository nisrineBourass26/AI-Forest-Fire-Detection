"""
web/dashboard.py
-----------------
AI-Based Forest Fire Detection — Global Monitoring Dashboard

Features
--------
- Multi-page Dash app (Overview, Map, Analytics, Alerts, Forecast, Data)
- XGBoost + Random Forest ensemble risk predictions
- Prophet + LSTM ensemble 7-day forecast
- DBSCAN hotspot clustering
- EN / FR bilingual interface
- Performance optimizations:
    * Data loaded once per page refresh, not on every callback
    * Map capped at 25,000 points (risk-prioritized)
    * Confidence scatter capped at 10,000 points
    * Interval refresh every 5 minutes (not 1 minute)
    * Country geocoding uses cache file
"""

import os
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from dash import Dash, dcc, html, dash_table, Input, Output, ctx
import dash_bootstrap_components as dbc
from datetime import datetime

# Resolve paths relative to project root
WEB_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(WEB_DIR)
sys.path.insert(0, ROOT_DIR)

from config import (
    LATEST_PREDICTIONS_PATH,
    LATEST_PREDICTIONS_COUNTRIES,
    LATEST_FORECAST_PATH,
)

DATA_PATH          = os.environ.get("PREDICTIONS_PATH", LATEST_PREDICTIONS_PATH)
FORECAST_PATH      = os.environ.get("FORECAST_PATH",    LATEST_FORECAST_PATH)
COUNTRY_CACHE_PATH = LATEST_PREDICTIONS_COUNTRIES
NASA_LOGO_PATH     = os.path.join(WEB_DIR, "assets", "nasa.png")
EMERGENCY_PHONE    = os.environ.get("EMERGENCY_PHONE", "112")

MINIMUM_COLUMNS = ["latitude", "longitude", "predicted_risk"]
RISK_ORDER      = ["Low", "Moderate", "High", "Extreme"]

RISK_COLORS = {
    "Low":     "#22C55E",
    "Moderate":"#FACC15",
    "High":    "#F97316",
    "Extreme": "#B91C1C",
    "Unknown": "#94A3B8",
}

RISK_FR = {
    "Low": "Faible", "Moderate": "Modéré",
    "High": "Élevé", "Extreme": "Extrême", "Unknown": "Inconnu",
}

TEXT = {
    "en": {
        "title": "AI-Based Forest Fire Detection and Early Warning Dashboard",
        "subtitle": "Global satellite monitoring · XGBoost + RF risk prediction · Prophet + LSTM forecasting · DBSCAN hotspot detection",
        "intro": "Overview", "map": "Global Map", "analytics": "Analytics",
        "alerts": "Alerts", "forecast": "7-Day Forecast", "data": "Records",
        "status": "System status: Operational",
        "last_update": "Last refresh",
        "map_view": "Map view", "country": "Country", "risk": "Risk level",
        "cluster": "Cluster", "confidence": "Min confidence",
        "total": "Total detections", "high": "High risk", "extreme": "Extreme risk",
        "clusters": "Active clusters", "avg_frp": "Avg predicted FRP",
        "intro_title": "Global Wildfire Intelligence Platform",
        "intro_text": (
            "This dashboard shows the latest output of the AI wildfire detection pipeline: "
            "NASA FIRMS satellite detections, XGBoost + Random Forest ensemble risk predictions, "
            "DBSCAN spatial hotspot clustering, and a Prophet + LSTM 7-day fire activity forecast."
        ),
        "notifications": "Priority Alerts",
        "forecast_title": "7-Day Global Fire Activity Forecast",
        "forecast_subtitle": "Prophet + LSTM ensemble forecast — Global fire activity index",
        "forecast_index": "Forecast index", "forecast_model": "Model",
        "forecast_date": "Date", "forecast_value": "Value",
        "forecast_missing": "No forecast file found. Run scripts/update_context_and_forecast.py first.",
        "map_title": "Interactive Global Fire Risk Map",
        "risk_dist": "Risk Level Distribution",
        "frp_dist": "Predicted FRP Distribution",
        "conf_chart": "Model Confidence vs Fire Intensity",
        "cluster_chart": "Top 10 Hotspot Clusters",
        "records": "Prediction Records",
        "no_alerts": "No high or extreme risk detections match current filters.",
        "copy": "Copy coordinates", "open_maps": "Open in Google Maps",
        "call_responsible": "Call emergency team",
        "all": "All", "auto": "Auto fit", "world": "Full world",
        "no_data": "No data available.", "no_match": "No detections match current filters.",
    },
    "fr": {
        "title": "Tableau de Bord de Détection des Feux de Forêt par IA",
        "subtitle": "Surveillance satellitaire mondiale · Prédiction XGBoost + RF · Prévision Prophet + LSTM · Détection DBSCAN",
        "intro": "Vue générale", "map": "Carte mondiale", "analytics": "Analyse",
        "alerts": "Alertes", "forecast": "Prévision 7 jours", "data": "Enregistrements",
        "status": "État du système : opérationnel",
        "last_update": "Dernier rafraîchissement",
        "map_view": "Vue carte", "country": "Pays", "risk": "Niveau de risque",
        "cluster": "Cluster", "confidence": "Confiance min.",
        "total": "Détections totales", "high": "Risque élevé", "extreme": "Risque extrême",
        "clusters": "Clusters actifs", "avg_frp": "FRP prédite moyenne",
        "intro_title": "Plateforme mondiale de surveillance des incendies",
        "intro_text": (
            "Ce tableau de bord affiche les dernières sorties du pipeline IA : "
            "détections NASA FIRMS, prédictions d'ensemble XGBoost + Random Forest, "
            "clustering DBSCAN et prévision Prophet + LSTM sur 7 jours."
        ),
        "notifications": "Alertes prioritaires",
        "forecast_title": "Prévision mondiale de l'activité des feux sur 7 jours",
        "forecast_subtitle": "Ensemble Prophet + LSTM — Indice d'activité mondiale des feux",
        "forecast_index": "Indice prévu", "forecast_model": "Modèle",
        "forecast_date": "Date", "forecast_value": "Valeur",
        "forecast_missing": "Aucun fichier de prévision trouvé. Exécutez scripts/update_context_and_forecast.py.",
        "map_title": "Carte mondiale interactive du risque d'incendie",
        "risk_dist": "Distribution des niveaux de risque",
        "frp_dist": "Distribution de la FRP prédite",
        "conf_chart": "Confiance du modèle vs intensité du feu",
        "cluster_chart": "Top 10 clusters de foyers",
        "records": "Enregistrements",
        "no_alerts": "Aucune détection à risque élevé ou extrême ne correspond aux filtres.",
        "copy": "Copier les coordonnées", "open_maps": "Ouvrir dans Google Maps",
        "call_responsible": "Appeler l'équipe d'urgence",
        "all": "Tous", "auto": "Ajuster aux données", "world": "Monde entier",
        "no_data": "Aucune donnée disponible.", "no_match": "Aucune détection ne correspond aux filtres.",
    },
}


# ============================================================
# COLORS
# ============================================================

def colors():
    return {
        "bg":      "#020617",
        "card":    "#0F172A",
        "card2":   "#1E293B",
        "text":    "#F8FAFC",
        "muted":   "#CBD5E1",
        "border":  "#334155",
        "primary": "#38BDF8",
        "map":     "carto-darkmatter",
    }


# ============================================================
# DATA UTILITIES
# ============================================================

def normalize_risk(value):
    value = str(value).strip()
    mapping = {
        "0": "Low", "1": "Moderate", "2": "High", "3": "Extreme",
        "low": "Low", "faible": "Low",
        "moderate": "Moderate", "medium": "Moderate", "modéré": "Moderate", "modere": "Moderate",
        "high": "High", "élevé": "High", "eleve": "High",
        "extreme": "Extreme", "extrême": "Extreme",
    }
    return mapping.get(value.lower(), value if value in RISK_ORDER else "Unknown")


def country_name_from_code(code):
    try:
        import pycountry
        country = pycountry.countries.get(alpha_2=str(code).upper())
        return country.name if country else str(code).upper()
    except Exception:
        return str(code).upper()


def add_country_column(df):
    """Add display_country using existing column, cache, or reverse geocoder."""
    for col in ["country", "Country", "COUNTRY", "admin_country", "country_name"]:
        if col in df.columns:
            df["display_country"] = (
                df[col].astype(str)
                .replace({"nan": "Unknown", "None": "Unknown", "": "Unknown"})
            )
            return df

    if os.path.exists(COUNTRY_CACHE_PATH):
        try:
            cached = pd.read_csv(COUNTRY_CACHE_PATH, low_memory=False)
            if {"latitude", "longitude", "display_country"}.issubset(cached.columns):
                cached = cached[["latitude", "longitude", "display_country"]].copy()
                cached["lat_r"] = cached["latitude"].round(4)
                cached["lon_r"] = cached["longitude"].round(4)
                cached = cached.drop_duplicates(["lat_r", "lon_r"])

                df["lat_r"] = df["latitude"].round(4)
                df["lon_r"] = df["longitude"].round(4)
                df = df.merge(cached[["lat_r", "lon_r", "display_country"]], on=["lat_r", "lon_r"], how="left")
                df = df.drop(columns=["lat_r", "lon_r"])

                if df["display_country"].notna().mean() > 0.90:
                    df["display_country"] = df["display_country"].fillna("Unknown")
                    return df
                df = df.drop(columns=["display_country"], errors="ignore")
        except Exception:
            pass

    try:
        import reverse_geocoder as rg
        coords  = list(zip(df["latitude"].astype(float), df["longitude"].astype(float)))
        results = rg.search(coords, mode=1)
        df["country_code"]    = [r.get("cc", "Unknown") for r in results]
        df["display_country"] = df["country_code"].apply(country_name_from_code)
        try:
            df.to_csv(COUNTRY_CACHE_PATH, index=False)
        except Exception:
            pass
    except Exception:
        df["display_country"] = "Unknown"

    return df


def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Missing: {DATA_PATH}. Run run_pipeline.py first."
        )

    df = pd.read_csv(DATA_PATH, low_memory=False)

    missing = [c for c in MINIMUM_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Prediction CSV missing required columns: {missing}")

    df["latitude"]  = pd.to_numeric(df["latitude"],  errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude"])
    df = df[df["latitude"].between(-90, 90) & df["longitude"].between(-180, 180)].copy()

    df["frp"]           = pd.to_numeric(df.get("frp", 0), errors="coerce").fillna(0)
    df["predicted_frp"] = pd.to_numeric(df.get("predicted_frp", df["frp"]), errors="coerce").fillna(0)

    df["risk_confidence"] = pd.to_numeric(df.get("risk_confidence", 1.0), errors="coerce").fillna(0)
    if df["risk_confidence"].max() > 1:
        df["risk_confidence"] /= 100.0
    df["risk_confidence"] = df["risk_confidence"].clip(0, 1)

    df["cluster_id"] = pd.to_numeric(df.get("cluster_id", -1), errors="coerce").fillna(-1).astype(int)

    if "prediction_time" in df.columns:
        df["prediction_time"] = pd.to_datetime(df["prediction_time"], errors="coerce")
    elif "datetime" in df.columns:
        df["prediction_time"] = pd.to_datetime(df["datetime"], errors="coerce")
    elif "acq_date" in df.columns:
        df["prediction_time"] = pd.to_datetime(df["acq_date"], errors="coerce")
    else:
        df["prediction_time"] = pd.NaT

    df["predicted_risk"] = df["predicted_risk"].apply(normalize_risk)
    df = add_country_column(df)

    return df


def load_forecast_data():
    if not os.path.exists(FORECAST_PATH):
        return pd.DataFrame()
    try:
        forecast = pd.read_csv(FORECAST_PATH, low_memory=False)
    except Exception:
        return pd.DataFrame()

    if "ds" not in forecast.columns:
        return pd.DataFrame()

    forecast["ds"] = pd.to_datetime(forecast["ds"], errors="coerce")
    forecast = forecast.dropna(subset=["ds"]).sort_values("ds").reset_index(drop=True)

    # Support both ensemble output (forecast_value) and Prophet-only (prophet_forecast / yhat)
    for candidate in ["forecast_value", "final_forecast", "prophet_forecast", "ensemble_pred", "yhat"]:
        if candidate in forecast.columns:
            forecast["forecast_value"] = pd.to_numeric(forecast[candidate], errors="coerce")
            break
    else:
        return pd.DataFrame()

    for src, dst in [("forecast_lower", "forecast_lower"), ("prophet_lower", "forecast_lower"),
                     ("forecast_upper", "forecast_upper"), ("prophet_upper", "forecast_upper")]:
        if src in forecast.columns and dst not in forecast.columns:
            forecast[dst] = pd.to_numeric(forecast[src], errors="coerce")

    if "forecast_lower" not in forecast.columns:
        forecast["forecast_lower"] = np.nan
    if "forecast_upper" not in forecast.columns:
        forecast["forecast_upper"] = np.nan

    if "forecast_model" not in forecast.columns:
        forecast["forecast_model"] = "Prophet + LSTM"

    forecast["forecast_value"] = forecast["forecast_value"].fillna(0).clip(lower=0)
    return forecast


# ============================================================
# LAYOUT HELPERS
# ============================================================

def card_style(c):
    return {
        "backgroundColor": c["card"],
        "border":          f"1px solid {c['border']}",
        "borderRadius":    "14px",
        "boxShadow":       "0 8px 18px rgba(0,0,0,0.18)",
    }


def section_card(title, content, c):
    return dbc.Card([
        dbc.CardHeader(
            title, className="fw-bold",
            style={"backgroundColor": c["card2"], "color": c["text"],
                   "borderBottom": f"1px solid {c['border']}"},
        ),
        dbc.CardBody(content),
    ], style=card_style(c))


def metric_card(label, value, color, c):
    return dbc.Col(
        dbc.Card(
            dbc.CardBody([
                html.Div(label, className="small fw-semibold mb-2", style={"color": c["muted"]}),
                html.H3(value, className="fw-bold mb-0", style={"color": color}),
            ]),
            style={**card_style(c), "borderLeft": f"5px solid {color}"},
        ),
        lg=2, md=4, sm=6,
    )


def empty_figure(message, c, height=420):
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper",
                       showarrow=False, font=dict(color=c["muted"], size=14))
    fig.update_layout(paper_bgcolor=c["card"], plot_bgcolor=c["card"],
                      font_color=c["text"], height=height,
                      margin=dict(l=0, r=0, t=20, b=20))
    return fig


# ============================================================
# CHART BUILDERS
# ============================================================

def build_map(df, map_view, c, t, height=720):
    if df.empty:
        return empty_figure(t["no_match"], c, height=height)

    plot_df = df.copy()

    if len(plot_df) > 25000:
        risk_rank = {"Extreme": 4, "High": 3, "Moderate": 2, "Low": 1, "Unknown": 0}
        plot_df["_rank"] = plot_df["predicted_risk"].map(risk_rank).fillna(0)
        plot_df = plot_df.sort_values(
            ["_rank", "risk_confidence", "predicted_frp"], ascending=False
        ).head(25000).drop(columns=["_rank"])

    plot_df["marker_size"] = np.log1p(plot_df["predicted_frp"].clip(lower=0)).clip(2.5, 7.5)

    center = {"lat": 15, "lon": 0}
    zoom   = 0.75

    if map_view == "auto" and not plot_df.empty:
        center = {"lat": float(plot_df["latitude"].mean()), "lon": float(plot_df["longitude"].mean())}
        span   = max(
            float(plot_df["latitude"].max()  - plot_df["latitude"].min()),
            float(plot_df["longitude"].max() - plot_df["longitude"].min()),
        )
        zoom = (0.8 if span > 140 else 1.2 if span > 90 else 2.0 if span > 50
                else 3.0 if span > 25 else 4.0 if span > 10 else 5.5 if span > 4
                else 7.0 if span > 1 else 9.0)

    hover_cols = {
        "display_country": True,
        "latitude":        ":.4f",
        "longitude":       ":.4f",
        "frp":             ":.2f",
        "predicted_frp":   ":.2f",
        "risk_confidence": ":.1%",
        "cluster_id":      True,
        "prediction_time": True,
        "marker_size":     False,
    }
    hover_cols = {k: v for k, v in hover_cols.items() if k in plot_df.columns}

    fig = px.scatter_mapbox(
        plot_df, lat="latitude", lon="longitude",
        color="predicted_risk", size="marker_size",
        color_discrete_map=RISK_COLORS,
        category_orders={"predicted_risk": RISK_ORDER},
        size_max=8, center=center, zoom=zoom, opacity=0.75,
        hover_data=hover_cols,
    )
    fig.update_layout(
        mapbox_style=c["map"],
        mapbox=dict(center=center, zoom=zoom, bearing=0, pitch=0),
        paper_bgcolor=c["card"], plot_bgcolor=c["card"], font_color=c["text"],
        height=height, margin=dict(l=0, r=0, t=0, b=0),
        legend_title_text="Risk", dragmode="pan", uirevision="keep-map-position",
    )
    return fig


def build_risk_chart(df, lang, c, t):
    if df.empty:
        return empty_figure(t["no_data"], c)
    counts = df["predicted_risk"].value_counts().reindex(RISK_ORDER).fillna(0).reset_index()
    counts.columns = ["Risk", "Count"]
    counts["Display"] = counts["Risk"].map(RISK_FR) if lang == "fr" else counts["Risk"]
    fig = px.bar(counts, x="Display", y="Count", color="Risk", color_discrete_map=RISK_COLORS)
    fig.update_layout(paper_bgcolor=c["card"], plot_bgcolor=c["card"],
                      font_color=c["text"], showlegend=False)
    return fig


def build_frp_chart(df, c, t):
    if df.empty:
        return empty_figure(t["no_data"], c)
    fig = px.histogram(df, x="predicted_frp", nbins=40, color_discrete_sequence=["#38BDF8"])
    fig.update_layout(
        paper_bgcolor=c["card"], plot_bgcolor=c["card"], font_color=c["text"],
        xaxis_title="Predicted FRP (MW)", yaxis_title="Detections",
    )
    return fig


def build_confidence_chart(df, c, t):
    if df.empty:
        return empty_figure(t["no_data"], c)
    sample = df.sample(10000, random_state=42) if len(df) > 10000 else df
    hover = {k: True for k in ["display_country", "latitude", "longitude", "cluster_id"] if k in sample.columns}
    fig = px.scatter(
        sample, x="predicted_frp", y="risk_confidence",
        color="predicted_risk", color_discrete_map=RISK_COLORS,
        hover_data=hover,
    )
    fig.update_layout(
        paper_bgcolor=c["card"], plot_bgcolor=c["card"], font_color=c["text"],
        xaxis_title="Predicted FRP (MW)", yaxis_title="Ensemble confidence",
    )
    return fig


def build_cluster_chart(df, c, t):
    if df.empty:
        return empty_figure(t["no_data"], c)
    cdf = df[df["cluster_id"] != -1]["cluster_id"].value_counts().head(10).reset_index()
    if cdf.empty:
        return empty_figure("No active clusters found.", c)
    cdf.columns = ["Cluster ID", "Detections"]
    cdf["Cluster ID"] = cdf["Cluster ID"].astype(str)
    fig = px.bar(cdf, x="Cluster ID", y="Detections", color_discrete_sequence=["#A855F7"])
    fig.update_layout(paper_bgcolor=c["card"], plot_bgcolor=c["card"], font_color=c["text"])
    return fig


def build_forecast_chart(forecast_df, c, t, height=430):
    if forecast_df.empty:
        return empty_figure(t["forecast_missing"], c, height=height)

    fig = go.Figure()

    has_bounds = (
        "forecast_lower" in forecast_df.columns
        and "forecast_upper" in forecast_df.columns
        and forecast_df["forecast_lower"].notna().any()
    )

    if has_bounds:
        fig.add_trace(go.Scatter(
            x=forecast_df["ds"], y=forecast_df["forecast_upper"],
            mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=forecast_df["ds"], y=forecast_df["forecast_lower"],
            mode="lines", fill="tonexty", fillcolor="rgba(56, 189, 248, 0.18)",
            line=dict(width=0), name="Confidence interval", hoverinfo="skip",
        ))

    # Prophet baseline (dashed) — if available separately
    if "prophet_forecast" in forecast_df.columns:
        fig.add_trace(go.Scatter(
            x=forecast_df["ds"], y=forecast_df["prophet_forecast"],
            mode="lines", line=dict(width=1.5, color="#94A3B8", dash="dash"),
            name="Prophet baseline",
            hovertemplate="Date: %{x|%Y-%m-%d}<br>Prophet: %{y:.4f}<extra></extra>",
        ))

    # Final ensemble line
    fig.add_trace(go.Scatter(
        x=forecast_df["ds"], y=forecast_df["forecast_value"],
        mode="lines+markers",
        line=dict(width=3, color="#38BDF8"),
        marker=dict(size=8, color="#38BDF8"),
        name="Prophet + LSTM ensemble",
        hovertemplate="Date: %{x|%Y-%m-%d}<br>Forecast: %{y:.4f}<extra></extra>",
    ))

    fig.update_layout(
        paper_bgcolor=c["card"], plot_bgcolor=c["card"], font_color=c["text"],
        height=height, margin=dict(l=20, r=20, t=40, b=40),
        xaxis_title=t["forecast_date"], yaxis_title=t["forecast_value"],
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(gridcolor=c["border"])
    fig.update_yaxes(gridcolor=c["border"], rangemode="tozero")
    return fig


def build_alerts(df, lang, c, with_actions=False):
    t = TEXT[lang]
    alerts_df = df[df["predicted_risk"].isin(["High", "Extreme"])].copy()

    if alerts_df.empty:
        return dbc.Alert(t["no_alerts"], color="success", className="mb-0")

    alerts_df = alerts_df.sort_values(
        ["risk_confidence", "predicted_frp"], ascending=False
    ).head(10)

    items = []
    for _, row in alerts_df.iterrows():
        color    = RISK_COLORS.get(row["predicted_risk"], RISK_COLORS["Unknown"])
        risk     = RISK_FR[row["predicted_risk"]] if lang == "fr" else row["predicted_risk"]
        location = f"{row['latitude']:.6f}, {row['longitude']:.6f}"
        country  = row.get("display_country", "Unknown")

        actions = html.Div()
        if with_actions:
            actions = html.Div([
                dcc.Clipboard(
                    content=location, title=t["copy"],
                    style={"display": "inline-block", "fontSize": "18px",
                           "cursor": "pointer", "marginRight": "12px"},
                ),
                html.A(t["open_maps"],
                       href=f"https://www.google.com/maps/search/?api=1&query={location}",
                       target="_blank", className="btn btn-sm btn-outline-info me-2"),
                html.A(t["call_responsible"], href=f"tel:{EMERGENCY_PHONE}",
                       className="btn btn-sm btn-outline-danger"),
            ], className="mt-2")

        # Show XGBoost vs RF risk if available (shows ensemble transparency)
        model_detail = ""
        if "xgb_predicted_risk" in row and "rf_predicted_risk" in row:
            xgb = row.get("xgb_predicted_risk", "—")
            rf  = row.get("rf_predicted_risk",  "—")
            model_detail = f"XGBoost: {xgb} | Random Forest: {rf}"

        items.append(html.Div([
            html.Div(f"{risk.upper()} RISK DETECTION", className="fw-bold mb-1", style={"color": color}),
            html.Div(f"Country: {country} | Location: {location}", className="small", style={"color": c["text"]}),
            html.Div(f"Ensemble FRP: {row['predicted_frp']:.2f} MW | Confidence: {row['risk_confidence']:.1%}",
                     className="small", style={"color": c["text"]}),
            html.Div(model_detail, className="small", style={"color": c["muted"]}) if model_detail else html.Div(),
            html.Div(f"Cluster: {int(row['cluster_id'])} | Time: {row.get('prediction_time', '—')}",
                     className="small", style={"color": c["muted"]}),
            actions,
        ], className="p-3 mb-2 rounded",
           style={"backgroundColor": c["card2"], "borderLeft": f"5px solid {color}"}))

    return html.Div(items)


def build_table(df, lang, c):
    if df.empty:
        return html.Div(TEXT[lang]["no_data"], style={"color": c["muted"]})

    possible_cols = [
        "prediction_time", "display_country", "latitude", "longitude",
        "frp", "xgb_predicted_risk", "rf_predicted_risk", "predicted_risk",
        "predicted_frp", "risk_confidence", "cluster_id",
        "satellite", "instrument", "source",
    ]
    cols     = [col for col in possible_cols if col in df.columns]
    table_df = df[cols].copy().sort_values("prediction_time", ascending=False) \
               if "prediction_time" in df.columns else df[cols].copy()

    if "prediction_time" in table_df.columns:
        table_df["prediction_time"] = table_df["prediction_time"].astype(str)
    for col in ["latitude", "longitude"]:
        if col in table_df.columns:
            table_df[col] = table_df[col].round(4)
    for col in ["frp", "predicted_frp"]:
        if col in table_df.columns:
            table_df[col] = table_df[col].round(2)
    if "risk_confidence" in table_df.columns:
        table_df["risk_confidence"] = (table_df["risk_confidence"] * 100).round(1).astype(str) + "%"
    if lang == "fr" and "predicted_risk" in table_df.columns:
        table_df["predicted_risk"] = table_df["predicted_risk"].map(RISK_FR).fillna(table_df["predicted_risk"])

    return dash_table.DataTable(
        data=table_df.to_dict("records"),
        columns=[{"name": col.replace("_", " ").upper(), "id": col} for col in table_df.columns],
        page_size=12, sort_action="native", filter_action="native",
        style_table={"overflowX": "auto", "maxHeight": "560px", "overflowY": "auto"},
        style_header={"backgroundColor": c["card2"], "color": c["text"],
                      "fontWeight": "bold", "border": f"1px solid {c['border']}"},
        style_cell={"backgroundColor": c["card"], "color": c["text"],
                    "border": f"1px solid {c['border']}", "padding": "9px",
                    "fontSize": "12px", "textAlign": "center",
                    "fontFamily": "Arial", "minWidth": "110px"},
    )


# ============================================================
# PAGE BUILDERS
# ============================================================

def intro_page(df, kpis, t, c, lang, map_view):
    return html.Div([
        kpis,
        dbc.Row([
            dbc.Col([
                section_card(
                    t["intro_title"],
                    html.Div([
                        html.P(t["intro_text"], style={"color": c["muted"], "fontSize": "15px"}),
                        html.Hr(style={"borderColor": c["border"]}),
                        html.H6("System modules" if lang == "en" else "Modules du système",
                                style={"color": c["text"]}),
                        html.Ul([
                            html.Li("NASA FIRMS world data acquisition (VIIRS + MODIS)"),
                            html.Li("XGBoost + Random Forest ensemble risk classification"),
                            html.Li("Predicted Fire Radiative Power (FRP) regression"),
                            html.Li("DBSCAN spatial hotspot clustering"),
                            html.Li("Prophet + LSTM 7-day global fire activity forecasting"),
                            html.Li("Global early warning and emergency decision support"),
                        ], style={"color": c["muted"], "fontSize": "14px"}),
                    ]),
                    c,
                )
            ], lg=5),
            dbc.Col([
                section_card(
                    t["map_title"],
                    dcc.Graph(
                        figure=build_map(df.head(5000), map_view, c, t, height=430),
                        config={"displayModeBar": False, "scrollZoom": True, "responsive": True},
                    ),
                    c,
                )
            ], lg=7),
        ], className="mb-4"),

        dbc.Row([
            dbc.Col([
                section_card(
                    t["forecast_title"],
                    dcc.Graph(figure=build_forecast_chart(load_forecast_data(), c, t, height=340)),
                    c,
                )
            ], lg=12)
        ], className="mb-4"),

        dbc.Row([
            dbc.Col([section_card(t["notifications"], build_alerts(df, lang, c, with_actions=True), c)])
        ]),
    ])


def forecast_page(forecast_df, t, c):
    if forecast_df.empty:
        return section_card(
            t["forecast_title"],
            html.Div(t["forecast_missing"], style={"color": c["muted"]}), c,
        )

    avg_val   = forecast_df["forecast_value"].mean()
    max_val   = forecast_df["forecast_value"].max()
    model_name = str(forecast_df.get("forecast_model", pd.Series(["Prophet + LSTM"])).iloc[0])
    start_date = forecast_df["ds"].min().strftime("%Y-%m-%d")

    cards = dbc.Row([
        metric_card(t["forecast_model"],  model_name,        c["primary"],        c),
        metric_card(t["forecast_index"],  f"{avg_val:.3f}",  "#22C55E",           c),
        metric_card("Max",                f"{max_val:.3f}",  "#F97316",           c),
        metric_card(t["forecast_date"],   start_date,        "#A855F7",           c),
    ], className="g-3 mb-4")

    table_df = forecast_df.copy()
    table_df["ds"] = table_df["ds"].dt.strftime("%Y-%m-%d")
    for col in ["forecast_value", "forecast_lower", "forecast_upper", "prophet_forecast", "lstm_correction"]:
        if col in table_df.columns:
            table_df[col] = pd.to_numeric(table_df[col], errors="coerce").round(4)

    visible_cols = [c_ for c_ in [
        "ds", "prophet_forecast", "lstm_correction", "forecast_value",
        "forecast_lower", "forecast_upper", "forecast_model", "generated_at",
    ] if c_ in table_df.columns]

    return html.Div([
        cards,
        section_card(
            t["forecast_title"],
            html.Div([
                html.P(t["forecast_subtitle"], className="mb-3", style={"color": c["muted"]}),
                dcc.Graph(figure=build_forecast_chart(forecast_df, c, t, height=460),
                          config={"displayModeBar": True, "responsive": True}),
            ]),
            c,
        ),
        html.Div(className="mb-4"),
        section_card(
            t["data"],
            dash_table.DataTable(
                data=table_df[visible_cols].to_dict("records"),
                columns=[{"name": col.replace("_", " ").upper(), "id": col} for col in visible_cols],
                page_size=10, style_table={"overflowX": "auto"},
                style_header={"backgroundColor": c["card2"], "color": c["text"],
                              "fontWeight": "bold", "border": f"1px solid {c['border']}"},
                style_cell={"backgroundColor": c["card"], "color": c["text"],
                            "border": f"1px solid {c['border']}", "padding": "9px",
                            "fontSize": "12px", "textAlign": "center",
                            "fontFamily": "Arial", "minWidth": "130px"},
            ),
            c,
        ),
    ])


# ============================================================
# APP INIT
# ============================================================

app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP],
    suppress_callback_exceptions=True,
)

app.title = "AI Forest Fire Detection Dashboard"

app.layout = html.Div([
    dcc.Location(id="url"),
    dcc.Store(id="language-store", data="en"),
    # Refresh every 5 minutes (was 1 minute — reduces server load significantly)
    dcc.Interval(id="interval-update", interval=5 * 60 * 1000, n_intervals=0),
    html.Div(id="app-shell"),
])


@app.callback(
    Output("language-store", "data"),
    Input("lang-en", "n_clicks"),
    Input("lang-fr", "n_clicks"),
    prevent_initial_call=True,
)
def toggle_language(_, __):
    return "fr" if ctx.triggered_id == "lang-fr" else "en"


@app.callback(
    Output("app-shell", "children"),
    Input("language-store", "data"),
    Input("url", "pathname"),
)
def render_shell(lang, pathname):
    """Renders the static shell: navbar, filters, and page skeleton.
    Data is NOT reloaded here — only on page content render."""
    df = load_data()
    c  = colors()
    t  = TEXT[lang]

    clusters = sorted(df["cluster_id"].unique())
    cluster_options = [{"label": f"{t['all']} clusters", "value": "all"}] + [
        {"label": "Noise (-1)" if cl == -1 else f"Cluster {cl}", "value": int(cl)}
        for cl in clusters
    ]

    countries       = sorted([x for x in df["display_country"].dropna().unique() if str(x).strip()])
    country_options = [{"label": t["all"], "value": "all"}] + [
        {"label": str(co), "value": str(co)} for co in countries
    ]

    logo = html.Div(
        "NASA",
        style={"width": "54px", "height": "54px", "borderRadius": "50%",
               "backgroundColor": "#DC2626", "color": "white",
               "display": "flex", "alignItems": "center", "justifyContent": "center",
               "fontWeight": "bold", "fontSize": "14px", "marginRight": "12px"},
    )
    if os.path.exists(NASA_LOGO_PATH):
        logo = html.Img(src="/assets/nasa.png",
                        style={"height": "54px", "width": "54px",
                               "objectFit": "contain", "marginRight": "12px"})

    page = pathname or "/"

    return html.Div([
        dbc.Navbar(
            dbc.Container([
                html.Div([logo,
                          dbc.NavbarBrand("AI Forest Fire Detection", className="fw-bold",
                                          style={"color": "white"})],
                         className="d-flex align-items-center"),
                dbc.Nav([
                    dbc.NavLink(t["intro"],    href="/",          active=page == "/",          style={"color": "white"}),
                    dbc.NavLink(t["map"],      href="/map",       active=page == "/map",       style={"color": "white"}),
                    dbc.NavLink(t["analytics"],href="/analytics", active=page == "/analytics", style={"color": "white"}),
                    dbc.NavLink(t["alerts"],   href="/alerts",    active=page == "/alerts",    style={"color": "white"}),
                    dbc.NavLink(t["forecast"], href="/forecast",  active=page == "/forecast",  style={"color": "white"}),
                    dbc.NavLink(t["data"],     href="/data",      active=page == "/data",      style={"color": "white"}),
                ], navbar=True, className="ms-auto"),
                dbc.DropdownMenu(
                    label="EN / FR",
                    children=[
                        dbc.DropdownMenuItem("English",  id="lang-en"),
                        dbc.DropdownMenuItem("Français", id="lang-fr"),
                    ],
                    color="secondary", size="sm", className="ms-2",
                ),
            ], fluid=True),
            color="dark", dark=True, sticky="top", className="mb-4",
        ),

        dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1(t["title"],    className="fw-bold mb-2", style={"color": c["text"]}),
                    html.P(t["subtitle"],  className="mb-0",         style={"color": c["muted"]}),
                ], lg=8),
                dbc.Col([
                    dbc.Card(dbc.CardBody([
                        html.Div(t["status"],       className="fw-bold", style={"color": "#22C55E"}),
                        html.Div(id="last-refresh-text", className="small", style={"color": c["muted"]}),
                        html.Div(f"Source: {os.path.basename(DATA_PATH)}",
                                 className="small mt-1", style={"color": c["muted"]}),
                    ]), style=card_style(c))
                ], lg=4),
            ], className="mb-4"),

            # Filters
            dbc.Card(dbc.CardBody(dbc.Row([
                dbc.Col([
                    html.Label(t["map_view"], className="small fw-bold", style={"color": c["text"]}),
                    dcc.Dropdown(id="map-view-filter",
                                 options=[{"label": t["world"], "value": "world"},
                                          {"label": t["auto"],  "value": "auto"}],
                                 value="world", clearable=False,
                                 persistence=True, persistence_type="session"),
                ], lg=2, md=6, className="mb-3"),

                dbc.Col([
                    html.Label(t["country"], className="small fw-bold", style={"color": c["text"]}),
                    dcc.Dropdown(id="country-filter", options=country_options, value="all",
                                 clearable=False, searchable=True,
                                 persistence=True, persistence_type="session"),
                ], lg=3, md=6, className="mb-3"),

                dbc.Col([
                    html.Label(t["risk"], className="small fw-bold", style={"color": c["text"]}),
                    dcc.Dropdown(id="risk-filter",
                                 options=[{"label": r, "value": r} for r in RISK_ORDER],
                                 value=RISK_ORDER, multi=True,
                                 persistence=True, persistence_type="session"),
                ], lg=3, md=6, className="mb-3"),

                dbc.Col([
                    html.Label(t["cluster"], className="small fw-bold", style={"color": c["text"]}),
                    dcc.Dropdown(id="cluster-filter", options=cluster_options, value="all",
                                 clearable=False, searchable=True,
                                 persistence=True, persistence_type="session"),
                ], lg=2, md=6, className="mb-3"),

                dbc.Col([
                    html.Label(t["confidence"], className="small fw-bold", style={"color": c["text"]}),
                    dcc.Slider(id="confidence-filter", min=0, max=1, step=0.01, value=0.0,
                               marks={0: "0%", 0.5: "50%", 1: "100%"},
                               tooltip={"placement": "bottom", "always_visible": True},
                               persistence=True, persistence_type="session"),
                ], lg=2, md=12, className="mb-3"),
            ])), style=card_style(c), className="mb-4"),

            html.Div(id="page-content"),
        ], fluid=True),
    ], style={"backgroundColor": c["bg"], "minHeight": "100vh", "paddingBottom": "30px"})


@app.callback(
    Output("last-refresh-text", "children"),
    Input("language-store", "data"),
    Input("interval-update", "n_intervals"),
)
def update_refresh_text(lang, _):
    t = TEXT[lang]
    return f"{t['last_update']}: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"


@app.callback(
    Output("page-content", "children"),
    Input("language-store",    "data"),
    Input("url",               "pathname"),
    Input("interval-update",   "n_intervals"),
    Input("map-view-filter",   "value"),
    Input("country-filter",    "value"),
    Input("risk-filter",       "value"),
    Input("cluster-filter",    "value"),
    Input("confidence-filter", "value"),
)
def render_page(lang, pathname, _n, map_view, country, risks, cluster, min_conf):
    df = load_data()
    c  = colors()
    t  = TEXT[lang]

    risks    = risks or []
    filtered = df[
        df["predicted_risk"].isin(risks) &
        (df["risk_confidence"] >= float(min_conf))
    ].copy()

    if country != "all":
        filtered = filtered[filtered["display_country"].astype(str) == str(country)]
    if cluster != "all":
        filtered = filtered[filtered["cluster_id"] == int(cluster)]

    total           = len(filtered)
    high            = len(filtered[filtered["predicted_risk"] == "High"])
    extreme         = len(filtered[filtered["predicted_risk"] == "Extreme"])
    active_clusters = filtered[filtered["cluster_id"] != -1]["cluster_id"].nunique()
    avg_frp         = filtered["predicted_frp"].mean() if total else 0

    kpis = dbc.Row([
        metric_card(t["total"],    f"{total:,}",          c["primary"],        c),
        metric_card(t["high"],     f"{high:,}",           RISK_COLORS["High"], c),
        metric_card(t["extreme"],  f"{extreme:,}",        RISK_COLORS["Extreme"], c),
        metric_card(t["clusters"], f"{active_clusters:,}","#A855F7",           c),
        metric_card(t["avg_frp"],  f"{avg_frp:.1f} MW",  "#22C55E",           c),
    ], className="g-3 mb-4")

    page = pathname or "/"

    if page == "/":
        return intro_page(filtered, kpis, t, c, lang, map_view)

    if page == "/map":
        return section_card(
            t["map_title"],
            dcc.Graph(figure=build_map(filtered, map_view, c, t),
                      config={"displayModeBar": True, "scrollZoom": True, "responsive": True},
                      style={"height": "72vh"}),
            c,
        )

    if page == "/analytics":
        return html.Div([
            kpis,
            dbc.Row([
                dbc.Col(section_card(t["risk_dist"],    dcc.Graph(figure=build_risk_chart(filtered, lang, c, t)), c), lg=6),
                dbc.Col(section_card(t["frp_dist"],     dcc.Graph(figure=build_frp_chart(filtered, c, t)),        c), lg=6),
            ], className="mb-4"),
            dbc.Row([
                dbc.Col(section_card(t["conf_chart"],   dcc.Graph(figure=build_confidence_chart(filtered, c, t)), c), lg=6),
                dbc.Col(section_card(t["cluster_chart"],dcc.Graph(figure=build_cluster_chart(filtered, c, t)),    c), lg=6),
            ]),
        ])

    if page == "/alerts":
        return section_card(t["alerts"], build_alerts(filtered, lang, c, with_actions=True), c)

    if page == "/forecast":
        return forecast_page(load_forecast_data(), t, c)

    if page == "/data":
        return section_card(t["records"], build_table(filtered, lang, c), c)

    return intro_page(filtered, kpis, t, c, lang, map_view)


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=8050)