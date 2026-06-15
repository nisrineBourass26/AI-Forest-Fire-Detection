"""
============================================================
AI Forest Fire Detection — Email Alert System
============================================================
Sends an automatic email whenever the pipeline detects
new Extreme or High confidence fire detections.

SETUP (one time only):
1. Go to your Gmail account
2. Enable 2-Factor Authentication
3. Go to: Google Account → Security → App Passwords
4. Create an App Password for "Mail"
5. Copy the 16-character password
6. Fill in the config below

Place this file in: AI-Forest-Fire-Detection/scripts/
============================================================
"""

import os
import smtplib
import ssl
import json
import pandas as pd
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

# ============================================================
# CONFIGURATION — fill these in
# ============================================================
SENDER_EMAIL    = "nisrinebourass27@gmail.com"          # Your Gmail address
APP_PASSWORD    = "ybyr kxup ykmr mkso"            # Gmail App Password (16 chars)
RECIPIENT_EMAIL = "nisrinebourass27@gmail.com"            # Who receives the alert (can be same as sender)

# Alert thresholds
MIN_CONFIDENCE  = 0.80    # Only alert if confidence >= 80%
RISK_LEVELS     = ["Extreme", "High"]   # Which risk levels trigger alert
MAX_ALERTS_PER_EMAIL = 10   # Show top N detections in the email

# Paths
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR    = os.path.dirname(SCRIPTS_DIR)
DATA_PATH   = os.path.join(ROOT_DIR, "data", "latest_predictions.csv")
SENT_LOG    = os.path.join(ROOT_DIR, "data", "alert_sent_log.json")

# ============================================================
# HELPERS
# ============================================================

def _continent_fallback(lat, lon):
    """Return readable region name when country lookup fails."""
    try:
        lat, lon = float(lat), float(lon)
    except Exception:
        return "Unknown region"
    if lat > 60:
        if lon < -30:  return "Northern Canada / Arctic"
        if lon < 60:   return "Northern Europe / Russia"
        return "Northern Russia / Siberia"
    if lat > 35:
        if lon < -100: return "North America (West)"
        if lon < -50:  return "North America (East)"
        if lon < 20:   return "Europe"
        if lon < 60:   return "Middle East"
        if lon < 100:  return "Central Asia"
        return "East Asia"
    if lat > 10:
        if lon < -80:  return "Central America"
        if lon < -30:  return "Caribbean"
        if lon < 20:   return "North Africa"
        if lon < 60:   return "Arabian Peninsula"
        if lon < 100:  return "South Asia"
        return "Southeast Asia"
    if lat > -10:
        if lon < -40:  return "Northern South America"
        if lon < 20:   return "Central Africa"
        if lon < 50:   return "East Africa"
        return "Maritime Southeast Asia"
    if lat > -35:
        if lon < -40:  return "Southern South America"
        if lon < 35:   return "Southern Africa"
        return "Australia / Oceania"
    return "Southern Ocean"


def resolve_country(row):
    """Get best available country name for a detection row."""
    country = str(row.get("display_country", "")).strip()
    if country and country not in ("Unknown", "nan", "None", ""):
        return country
    # Try reverse geocoder
    try:
        import reverse_geocoder as rg
        results = rg.search([(float(row["latitude"]), float(row["longitude"]))], mode=1)
        cc = results[0].get("cc", "") if results else ""
        if cc:
            try:
                import pycountry
                c = pycountry.countries.get(alpha_2=cc.upper())
                if c: return c.name
            except Exception:
                pass
            return cc
    except Exception:
        pass
    # Continent fallback
    return _continent_fallback(row.get("latitude", 0), row.get("longitude", 0))


def load_predictions():
    if not os.path.exists(DATA_PATH):
        print(f"[Alert] No predictions file found at {DATA_PATH}")
        return pd.DataFrame()
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df["risk_confidence"] = pd.to_numeric(df["risk_confidence"], errors="coerce").fillna(0)
    if df["risk_confidence"].max() > 1:
        df["risk_confidence"] /= 100.0
    return df


def get_critical_detections(df):
    """Return High/Extreme detections above confidence threshold."""
    if df.empty:
        return pd.DataFrame()
    result = df[
        (df["predicted_risk"].isin(RISK_LEVELS)) &
        (df["risk_confidence"] >= MIN_CONFIDENCE)
    ].copy().sort_values(["predicted_risk", "risk_confidence", "predicted_frp"],
                          ascending=False).head(MAX_ALERTS_PER_EMAIL)
    # Resolve Unknown country names
    result["display_country"] = result.apply(resolve_country, axis=1)
    return result


def load_sent_log():
    if not os.path.exists(SENT_LOG):
        return {}
    try:
        with open(SENT_LOG, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def save_sent_log(log):
    os.makedirs(os.path.dirname(SENT_LOG), exist_ok=True)
    with open(SENT_LOG, "w") as f:
        json.dump(log, f, indent=2)


def _snapshot(df):
    """
    Build a comparable snapshot of current fire situation.
    Returns a dict with key statistics used to detect meaningful changes.
    """
    if df.empty:
        return {
            "extreme_count": 0,
            "high_count": 0,
            "total_critical": 0,
            "max_frp": 0.0,
            "countries": [],
            "top_clusters": [],
        }

    extreme = df[df["predicted_risk"] == "Extreme"]
    high    = df[df["predicted_risk"] == "High"]

    countries = sorted(df["display_country"].dropna().unique().tolist()) \
                if "display_country" in df.columns else []

    clustered = df[df["cluster_id"] != -1]
    top_clusters = clustered["cluster_id"].value_counts().head(5).index.tolist() \
                   if not clustered.empty else []

    return {
        "extreme_count":  int(len(extreme)),
        "high_count":     int(len(high)),
        "total_critical": int(len(df)),
        "max_frp":        float(df["predicted_frp"].max()) if len(df) else 0.0,
        "countries":      countries,
        "top_clusters":   [int(c) for c in top_clusters],
    }


def detect_changes(current_df):
    """
    Compare current situation with last alert snapshot.
    Returns (should_send, reasons) where reasons is a list of change descriptions.
    """
    if current_df.empty:
        return False, []

    log      = load_sent_log()
    last     = log.get("last_snapshot", {})
    current  = _snapshot(current_df)
    reasons  = []

    # First run ever — always send
    if not last:
        reasons.append(f"First detection run: {current['total_critical']:,} critical fires found")
        return True, reasons

    prev_extreme  = last.get("extreme_count", 0)
    prev_high     = last.get("high_count", 0)
    prev_total    = last.get("total_critical", 0)
    prev_frp      = last.get("max_frp", 0.0)
    prev_countries= set(last.get("countries", []))
    curr_countries= set(current["countries"])

    # 1. New Extreme detections appeared
    if current["extreme_count"] > prev_extreme:
        delta = current["extreme_count"] - prev_extreme
        reasons.append(
            f"+{delta} new Extreme risk detections "
            f"(was {prev_extreme}, now {current['extreme_count']})"
        )

    # 2. Extreme count jumped significantly (>20% increase)
    elif prev_extreme > 0 and current["extreme_count"] > prev_extreme * 1.20:
        reasons.append(
            f"Extreme detections increased by "
            f"{(current['extreme_count']/prev_extreme - 1)*100:.0f}%"
        )

    # 3. Total critical increased significantly (>30% increase)
    if prev_total > 0 and current["total_critical"] > prev_total * 1.30:
        reasons.append(
            f"Critical fire count surged: "
            f"{prev_total:,} → {current['total_critical']:,} "
            f"(+{(current['total_critical']/prev_total - 1)*100:.0f}%)"
        )

    # 4. Max FRP jumped significantly (>50% increase)
    if prev_frp > 0 and current["max_frp"] > prev_frp * 1.50:
        reasons.append(
            f"Max fire intensity jumped: "
            f"{prev_frp:.0f} MW → {current['max_frp']:.0f} MW"
        )

    # 5. New countries entered the danger zone
    new_countries = curr_countries - prev_countries
    new_countries = {c for c in new_countries
                     if c not in ("Unknown", "") and not c.startswith("Northern")
                     and not c.startswith("Southern")}
    if new_countries:
        reasons.append(
            f"New danger zones detected: {', '.join(sorted(new_countries))}"
        )

    if reasons:
        return True, reasons

    print(f"[Alert] No significant change detected vs last alert. Skipping.")
    print(f"  Previous: {prev_extreme} Extreme, {prev_high} High, max FRP {prev_frp:.0f} MW")
    print(f"  Current:  {current['extreme_count']} Extreme, {current['high_count']} High, "
          f"max FRP {current['max_frp']:.0f} MW")
    return False, []


def mark_alert_sent(current_df):
    log = load_sent_log()
    log["last_snapshot"] = _snapshot(current_df)
    log["last_sent"]     = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log["total_sent"]    = log.get("total_sent", 0) + 1
    save_sent_log(log)


# ============================================================
# EMAIL BUILDER
# ============================================================

RISK_COLORS = {
    "Extreme": "#B91C1C",
    "High":    "#F97316",
}

def build_email_html(critical_df, total_critical, reasons=None):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    n   = len(critical_df)
    reasons = reasons or []

    # Summary by country
    by_country = critical_df.groupby(
        critical_df.get("display_country", pd.Series(["Unknown"]*len(critical_df)))
    ).size().sort_values(ascending=False)
    country_summary = " · ".join(
        f"{cnt} in {cntry}" for cntry, cnt in by_country.head(5).items()
    )

    # Change reasons HTML
    reasons_html = ""
    if reasons:
        items = "".join(
            f'<li style="margin-bottom:4px;color:rgba(255,255,255,0.85);">{r}</li>'
            for r in reasons
        )
        reasons_html = f"""
  <div style="background:rgba(0,0,0,0.2);border-radius:8px;
              padding:12px 16px;margin-top:12px;">
    <div style="font-size:10px;color:rgba(255,255,255,0.5);
                text-transform:uppercase;letter-spacing:0.8px;margin-bottom:8px;">
      Why this alert was triggered
    </div>
    <ul style="margin:0;padding-left:18px;font-size:12px;">
      {items}
    </ul>
  </div>"""

    # Build detection rows
    rows_html = ""
    for _, row in critical_df.iterrows():
        risk    = row.get("predicted_risk", "Unknown")
        color   = RISK_COLORS.get(risk, "#94A3B8")
        country = row.get("display_country", "Unknown")
        lat     = float(row.get("latitude", 0))
        lon     = float(row.get("longitude", 0))
        frp     = float(row.get("predicted_frp", 0))
        conf    = float(row.get("risk_confidence", 0))
        cluster    = int(row.get("cluster_id", -1))
        cluster_str = "—" if cluster == -1 else f"#{cluster}"
        maps_url   = f"https://www.google.com/maps/search/?api=1&query={lat:.6f},{lon:.6f}"

        rows_html += f"""
        <tr>
          <td style="padding:10px 12px;border-bottom:1px solid #1E293B;">
            <span style="color:{color};font-weight:700;font-size:13px;">{risk.upper()}</span>
          </td>
          <td style="padding:10px 12px;border-bottom:1px solid #1E293B;
                     color:#F1F5F9;font-weight:600;">{country}</td>
          <td style="padding:10px 12px;border-bottom:1px solid #1E293B;
                     color:#38BDF8;font-weight:700;">{frp:.1f} MW</td>
          <td style="padding:10px 12px;border-bottom:1px solid #1E293B;
                     color:#CBD5E1;">{conf:.0%}</td>
          <td style="padding:10px 12px;border-bottom:1px solid #1E293B;
                     color:#94A3B8;font-size:12px;">{lat:.4f}, {lon:.4f}</td>
          <td style="padding:10px 12px;border-bottom:1px solid #1E293B;
                     color:#A855F7;font-weight:600;">{cluster_str}</td>
          <td style="padding:10px 12px;border-bottom:1px solid #1E293B;">
            <a href="{maps_url}" style="color:#38BDF8;text-decoration:none;
               font-size:12px;font-weight:600;">Open Maps →</a>
          </td>
        </tr>"""

    return f"""
<!DOCTYPE html>
<html>
<body style="margin:0;padding:0;background-color:#020617;font-family:Arial,sans-serif;">
<div style="max-width:800px;margin:0 auto;padding:24px;">

  <!-- Header -->
  <div style="background:linear-gradient(135deg,#7F1D1D,#B91C1C);
              border-radius:12px;padding:24px;margin-bottom:20px;">
    <div style="font-size:11px;color:rgba(255,255,255,0.7);
                text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;">
      AI FOREST FIRE DETECTION SYSTEM
    </div>
    <div style="font-size:22px;font-weight:800;color:#FFF;margin-bottom:6px;">
      Fire Alert — {total_critical:,} Critical Detections
    </div>
    <div style="font-size:13px;color:rgba(255,255,255,0.8);">
      {country_summary}
    </div>
    <div style="font-size:11px;color:rgba(255,255,255,0.6);margin-top:8px;">
      Generated: {now}
    </div>
    {reasons_html}
  </div>

  <!-- Summary strip -->
  <div style="display:flex;gap:12px;margin-bottom:20px;">
    <div style="flex:1;background:#0A1628;border:1px solid #1E293B;
                border-left:4px solid #B91C1C;border-radius:10px;padding:14px;">
      <div style="color:#475569;font-size:10px;font-weight:700;
                  text-transform:uppercase;letter-spacing:0.5px;">Total critical</div>
      <div style="color:#B91C1C;font-size:24px;font-weight:800;">{total_critical:,}</div>
    </div>
    <div style="flex:1;background:#0A1628;border:1px solid #1E293B;
                border-left:4px solid #F97316;border-radius:10px;padding:14px;">
      <div style="color:#475569;font-size:10px;font-weight:700;
                  text-transform:uppercase;letter-spacing:0.5px;">Min confidence</div>
      <div style="color:#F97316;font-size:24px;font-weight:800;">{MIN_CONFIDENCE:.0%}</div>
    </div>
    <div style="flex:1;background:#0A1628;border:1px solid #1E293B;
                border-left:4px solid #38BDF8;border-radius:10px;padding:14px;">
      <div style="color:#475569;font-size:10px;font-weight:700;
                  text-transform:uppercase;letter-spacing:0.5px;">Max FRP</div>
      <div style="color:#38BDF8;font-size:24px;font-weight:800;">
        {critical_df["predicted_frp"].max():.0f} MW
      </div>
    </div>
  </div>

  <!-- Detection table -->
  <div style="background:#0A1628;border:1px solid #1E293B;border-radius:12px;
              overflow:hidden;margin-bottom:20px;">
    <div style="background:#0D2137;padding:14px 16px;border-bottom:1px solid #1E293B;">
      <span style="color:#F1F5F9;font-weight:700;font-size:14px;">
        Top {n} Detections (sorted by risk &amp; FRP)
      </span>
    </div>
    <table style="width:100%;border-collapse:collapse;">
      <thead>
        <tr style="border-bottom:2px solid #1E3A5F;">
          <th style="padding:10px 12px;color:#64748B;font-size:11px;
                     text-align:left;text-transform:uppercase;letter-spacing:0.5px;">Risk</th>
          <th style="padding:10px 12px;color:#64748B;font-size:11px;
                     text-align:left;text-transform:uppercase;letter-spacing:0.5px;">Country</th>
          <th style="padding:10px 12px;color:#64748B;font-size:11px;
                     text-align:left;text-transform:uppercase;letter-spacing:0.5px;">FRP</th>
          <th style="padding:10px 12px;color:#64748B;font-size:11px;
                     text-align:left;text-transform:uppercase;letter-spacing:0.5px;">Confidence</th>
          <th style="padding:10px 12px;color:#64748B;font-size:11px;
                     text-align:left;text-transform:uppercase;letter-spacing:0.5px;">Coordinates</th>
          <th style="padding:10px 12px;color:#64748B;font-size:11px;
                     text-align:left;text-transform:uppercase;letter-spacing:0.5px;">Cluster</th>
          <th style="padding:10px 12px;color:#64748B;font-size:11px;
                     text-align:left;text-transform:uppercase;letter-spacing:0.5px;">Map</th>
        </tr>
      </thead>
      <tbody>{rows_html}</tbody>
    </table>
  </div>

  <!-- Footer -->
  <div style="text-align:center;color:#334155;font-size:11px;padding:10px;">
    AI Forest Fire Detection System — NASA FIRMS · MODIS · VIIRS<br>
    This alert was generated automatically by the prediction pipeline.
  </div>

</div>
</body>
</html>"""


# ============================================================
# MAIN SEND FUNCTION
# ============================================================

def send_fire_alert():
    print("[Alert] Loading predictions...")
    df = load_predictions()

    if df.empty:
        print("[Alert] No prediction data found. Skipping.")
        return

    critical = get_critical_detections(df)
    total_critical = len(df[
        (df["predicted_risk"].isin(RISK_LEVELS)) &
        (df["risk_confidence"] >= MIN_CONFIDENCE)
    ])

    if critical.empty:
        print(f"[Alert] No detections above {MIN_CONFIDENCE:.0%} confidence. No email sent.")
        # Still save snapshot so next run can compare
        mark_alert_sent(critical)
        return

    print(f"[Alert] Found {total_critical} critical detections.")

    # Smart change detection — only alert on meaningful new developments
    should_send, reasons = detect_changes(critical)
    if not should_send:
        return

    print(f"[Alert] Changes detected:")
    for r in reasons:
        print(f"  • {r}")

    # Build email with change reasons
    subject = (f"FIRE ALERT — {total_critical:,} critical detections "
               f"({datetime.now().strftime('%Y-%m-%d %H:%M')} UTC)")
    html_body = build_email_html(critical, total_critical, reasons)

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = SENDER_EMAIL
    msg["To"]      = RECIPIENT_EMAIL
    msg.attach(MIMEText(html_body, "html"))

    print(f"[Alert] Sending email to {RECIPIENT_EMAIL}...")
    try:
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(SENDER_EMAIL, APP_PASSWORD)
            server.sendmail(SENDER_EMAIL, RECIPIENT_EMAIL, msg.as_string())
        print(f"[Alert] Email sent successfully.")
        mark_alert_sent(critical)
    except smtplib.SMTPAuthenticationError:
        print("[Alert] ERROR: Authentication failed.")
        print("  → Use a Gmail App Password, not your account password.")
        print("  → Go to: myaccount.google.com/apppasswords")
    except Exception as e:
        print(f"[Alert] ERROR sending email: {e}")


if __name__ == "__main__":
    send_fire_alert()