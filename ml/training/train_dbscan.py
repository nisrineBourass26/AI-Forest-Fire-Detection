"""
train_dbscan.py
Wildfire Hotspot Detection — DBSCAN Spatial Clustering

Inputs : data/fires_with_clusters.csv  (falls back to fires.csv)

Outputs:
    saved_models/dbscan_model.joblib
    saved_models/dbscan_params.json
    evaluation/dbscan_report.txt
    evaluation/dbscan_cluster_map.png
    evaluation/dbscan_frp_by_cluster.png
    evaluation/dbscan_cluster_sizes.png
    evaluation/dbscan_kdistance.png
    evaluation/dbscan_grid_search.csv
"""

import os
import json
import warnings
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import joblib

from sklearn.cluster import DBSCAN
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


# ── Paths ──────────────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

MODEL_DIR = os.path.join(ROOT_DIR, "ml", "saved_models")
EVAL_DIR = os.path.join(ROOT_DIR, "ml", "evaluation")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)


# Prefer fires_with_clusters.csv, otherwise fires.csv
DATA_PATH = None

for fname in ("fires_with_clusters.csv", "fires.csv"):
    path = os.path.join(ROOT_DIR, "data", fname)
    if os.path.exists(path):
        DATA_PATH = path
        break

if DATA_PATH is None:
    raise FileNotFoundError("No data file found: fires_with_clusters.csv or fires.csv")


# ── 1. Load data ───────────────────────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    print(f"[INFO] Loading data from {path}")
    df = pd.read_csv(path, low_memory=False)

    print(f"[INFO] Shape: {df.shape}")
    print(f"[INFO] Columns: {df.columns.tolist()}\n")

    return df


# ── 2. Prepare spatial features ───────────────────────────────────────────────

def prepare_spatial_features(df: pd.DataFrame):
    """
    Prepare latitude and longitude for DBSCAN.

    Important:
    - Raw coords are kept in degrees for plots.
    - Radian coords are used for haversine DBSCAN.
    """

    df = df.copy()

    required_cols = ["latitude", "longitude"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

    if "frp" in df.columns:
        df["frp"] = pd.to_numeric(df["frp"], errors="coerce").clip(lower=0)

    df = df.dropna(subset=["latitude", "longitude"])

    print(f"[INFO] Valid spatial rows: {len(df)}\n")

    coords_deg = df[["latitude", "longitude"]].values
    coords_rad = np.radians(coords_deg)

    weights = None
    if "frp" in df.columns:
        weights = np.log1p(df["frp"].fillna(0).values)
        weights = weights / (weights.max() + 1e-9)

    return df, coords_deg, coords_rad, weights


# ── 3. K-distance plot ────────────────────────────────────────────────────────

def plot_k_distance(coords_rad: np.ndarray, k: int = 4):
    """
    Computes k-nearest-neighbor distance using radians.

    Since haversine DBSCAN needs eps in radians,
    the suggested eps returned here is also in radians.
    """

    print("[INFO] Computing k-distance plot to suggest eps...")

    sample = coords_rad

    if len(coords_rad) > 50_000:
        idx = np.random.choice(len(coords_rad), 50_000, replace=False)
        sample = coords_rad[idx]

    nbrs = NearestNeighbors(
        n_neighbors=k,
        algorithm="ball_tree",
        metric="haversine",
        n_jobs=-1
    )

    nbrs.fit(sample)

    distances, _ = nbrs.kneighbors(sample)
    k_dist = np.sort(distances[:, k - 1])[::-1]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(k_dist, linewidth=1)
    ax.set_xlabel("Points sorted")
    ax.set_ylabel(f"{k}-NN distance in radians")
    ax.set_title("K-Distance Plot — use elbow as eps")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_DIR, "dbscan_kdistance.png"), dpi=150)
    plt.close()

    elbow_idx = max(1, len(k_dist) // 10)
    suggested_eps = float(np.median(k_dist[elbow_idx: elbow_idx * 2]))

    print(f"[INFO] Suggested eps: {suggested_eps:.8f} radians")
    print(f"[INFO] Approx distance: {suggested_eps * 6371:.2f} km\n")

    return suggested_eps


# ── 4. Tune DBSCAN ────────────────────────────────────────────────────────────

def tune_dbscan(coords_rad: np.ndarray, suggested_eps: float):
    """
    Tune DBSCAN using a separate tuning subset.

    DBSCAN is unsupervised, but we still avoid evaluating final quality
    on exactly the same sample used for parameter selection.
    """

    print("[INFO] Grid-searching DBSCAN eps and min_samples...")

    eps_candidates = np.unique(
        np.round(
            np.linspace(
                max(0.00001, suggested_eps * 0.5),
                suggested_eps * 2.0,
                8
            ),
            8
        )
    )

    min_samples_candidates = [3, 5, 8, 10, 15]

    sample_size = min(30_000, len(coords_rad))
    sample_idx = np.random.choice(len(coords_rad), sample_size, replace=False)
    sample = coords_rad[sample_idx]

    best_score = -1
    best_params = {
        "eps": suggested_eps,
        "min_samples": 5
    }

    results = []

    for eps in eps_candidates:
        for min_s in min_samples_candidates:

            db = DBSCAN(
                eps=eps,
                min_samples=min_s,
                algorithm="ball_tree",
                metric="haversine",
                n_jobs=-1
            )

            labels = db.fit_predict(sample)

            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            noise_pct = (labels == -1).mean() * 100

            if n_clusters < 2:
                sil = -1
                results.append((eps, eps * 6371, min_s, n_clusters, noise_pct, sil))
                continue

            valid_mask = labels != -1

            if valid_mask.sum() < 100:
                sil = -1
            else:
                try:
                    sil = silhouette_score(
                        sample[valid_mask],
                        labels[valid_mask],
                        metric="euclidean",
                        sample_size=min(5000, valid_mask.sum())
                    )
                except Exception:
                    sil = -1

            results.append((eps, eps * 6371, min_s, n_clusters, noise_pct, sil))

            print(
                f"  eps={eps:.8f} rad "
                f"({eps * 6371:.2f} km) | "
                f"min_s={min_s:2d} | "
                f"clusters={n_clusters:4d} | "
                f"noise={noise_pct:.1f}% | "
                f"sil={sil:.4f}"
            )

            if sil > best_score:
                best_score = sil
                best_params = {
                    "eps": float(eps),
                    "eps_km": float(eps * 6371),
                    "min_samples": int(min_s)
                }

    print(f"\n[INFO] Best params: {best_params}")
    print(f"[INFO] Best silhouette: {best_score:.4f}\n")

    res_df = pd.DataFrame(
        results,
        columns=[
            "eps_radians",
            "eps_km",
            "min_samples",
            "n_clusters",
            "noise_pct",
            "silhouette"
        ]
    )

    res_df.to_csv(
        os.path.join(EVAL_DIR, "dbscan_grid_search.csv"),
        index=False
    )

    return best_params


# ── 5. Fit final DBSCAN ───────────────────────────────────────────────────────

def fit_dbscan(coords_rad: np.ndarray, params: dict, weights=None):
    """
    Fit final DBSCAN on full dataset.

    If FRP weights are available, they are used as sample_weight.
    """

    print(f"[INFO] Fitting DBSCAN on full dataset ({len(coords_rad)} points)...")

    db = DBSCAN(
        eps=params["eps"],
        min_samples=params["min_samples"],
        algorithm="ball_tree",
        metric="haversine",
        n_jobs=-1
    )

    if weights is not None:
        labels = db.fit_predict(coords_rad, sample_weight=weights)
    else:
        labels = db.fit_predict(coords_rad)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_pct = (labels == -1).mean() * 100

    print(f"[INFO] Final clusters: {n_clusters}")
    print(f"[INFO] Noise points: {noise_pct:.1f}%\n")

    return db, labels


# ── 6. Evaluate clusters ──────────────────────────────────────────────────────

def evaluate_clusters(df, coords_rad, labels, report_lines):
    """
    Evaluate clusters using a separate evaluation subset.
    """

    df = df.copy()
    df["cluster"] = labels

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_pct = (labels == -1).mean() * 100

    lines = [
        "── DBSCAN Clustering Results ──",
        f"Total detections : {len(labels)}",
        f"Clusters found   : {n_clusters}",
        f"Noise points     : {(labels == -1).sum()} ({noise_pct:.1f}%)",
    ]

    valid_mask = labels != -1

    if valid_mask.sum() > 100 and n_clusters >= 2:
        valid_indices = np.where(valid_mask)[0]

        sample_n = min(10_000, len(valid_indices))
        eval_idx = np.random.choice(valid_indices, sample_n, replace=False)

        try:
            sil = silhouette_score(
                coords_rad[eval_idx],
                labels[eval_idx],
                metric="euclidean"
            )

            db_score = davies_bouldin_score(
                coords_rad[eval_idx],
                labels[eval_idx]
            )

            ch_score = calinski_harabasz_score(
                coords_rad[eval_idx],
                labels[eval_idx]
            )

            lines += [
                f"Silhouette score     : {sil:.4f}  higher is better",
                f"Davies-Bouldin score : {db_score:.4f}  lower is better",
                f"Calinski-Harabasz    : {ch_score:.2f}  higher is better",
            ]

        except Exception as e:
            lines.append(f"Metrics error: {e}")

    if "frp" in df.columns:
        cluster_stats = (
            df[df["cluster"] >= 0]
            .groupby("cluster")["frp"]
            .agg(["count", "mean", "max", "sum"])
            .rename(
                columns={
                    "count": "detections",
                    "mean": "mean_frp",
                    "max": "max_frp",
                    "sum": "total_frp"
                }
            )
            .sort_values("total_frp", ascending=False)
        )

        lines.append("\nTop 10 Hotspot Clusters by total FRP:")
        lines.append(cluster_stats.head(10).to_string())

        cluster_stats.to_csv(
            os.path.join(EVAL_DIR, "dbscan_cluster_stats.csv")
        )

    for line in lines:
        print(line)

    report_lines.extend(lines)


# ── 7. Plots ──────────────────────────────────────────────────────────────────

def plot_cluster_map(df, labels):
    print("[INFO] Plotting cluster map...")

    df = df.copy()
    df["cluster"] = labels

    fig, ax = plt.subplots(figsize=(12, 7))

    noise = df[df["cluster"] == -1]

    ax.scatter(
        noise["longitude"],
        noise["latitude"],
        s=1,
        alpha=0.3,
        label="Noise",
        rasterized=True
    )

    clustered = df[df["cluster"] >= 0]

    n_clusters = clustered["cluster"].nunique()

    colors = cm.tab20(np.linspace(0, 1, max(n_clusters, 1)))
    unique_ids = sorted(clustered["cluster"].unique())
    id_to_color = {
        cid: colors[i % len(colors)]
        for i, cid in enumerate(unique_ids)
    }

    c_array = clustered["cluster"].map(id_to_color).tolist()

    ax.scatter(
        clustered["longitude"],
        clustered["latitude"],
        s=3,
        c=c_array,
        alpha=0.6,
        rasterized=True
    )

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"DBSCAN Wildfire Hotspot Clusters — n={n_clusters}")
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(
        os.path.join(EVAL_DIR, "dbscan_cluster_map.png"),
        dpi=150
    )
    plt.close()


def plot_frp_by_cluster(df, labels, top_n=20):
    if "frp" not in df.columns:
        return

    df = df.copy()
    df["cluster"] = labels

    top = (
        df[df["cluster"] >= 0]
        .groupby("cluster")["frp"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(top.index.astype(str), top.values)
    ax.set_xlabel("Cluster ID")
    ax.set_ylabel("Total FRP")
    ax.set_title(f"Top {top_n} Hotspot Clusters — Total FRP")

    plt.tight_layout()
    plt.savefig(
        os.path.join(EVAL_DIR, "dbscan_frp_by_cluster.png"),
        dpi=150
    )
    plt.close()


def plot_cluster_sizes(labels):
    sizes = pd.Series(labels[labels >= 0]).value_counts().values

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.hist(sizes, bins=50)
    ax.set_xlabel("Cluster size")
    ax.set_ylabel("Number of clusters")
    ax.set_title("Distribution of Cluster Sizes")

    plt.tight_layout()
    plt.savefig(
        os.path.join(EVAL_DIR, "dbscan_cluster_sizes.png"),
        dpi=150
    )
    plt.close()


# ── 8. Main ───────────────────────────────────────────────────────────────────

def main():
    report_lines = []

    df = load_data(DATA_PATH)

    df, coords_deg, coords_rad, weights = prepare_spatial_features(df)

    print("=" * 60)
    print("STEP 1/4 — K-distance analysis")
    print("=" * 60)

    suggested_eps = plot_k_distance(coords_rad, k=4)

    print("=" * 60)
    print("STEP 2/4 — Grid search for best eps / min_samples")
    print("=" * 60)

    best_params = tune_dbscan(coords_rad, suggested_eps)

    print("=" * 60)
    print("STEP 3/4 — Fit final DBSCAN model")
    print("=" * 60)

    db_model, labels = fit_dbscan(
        coords_rad,
        best_params,
        weights=weights
    )

    df["cluster"] = labels

    print("=" * 60)
    print("STEP 4/4 — Evaluation and plots")
    print("=" * 60)

    evaluate_clusters(df, coords_rad, labels, report_lines)
    plot_cluster_map(df, labels)
    plot_frp_by_cluster(df, labels)
    plot_cluster_sizes(labels)

    out_csv = os.path.join(ROOT_DIR, "data", "fires_with_clusters.csv")
    df.to_csv(out_csv, index=False)

    print(f"[INFO] Enriched dataset saved → {out_csv}")

    model_path = os.path.join(MODEL_DIR, "dbscan_model.joblib")

    joblib.dump(
        {
            "model": db_model,
            "params": best_params,
            "uses_haversine": True,
            "eps_unit": "radians",
            "eps_km": best_params.get("eps_km"),
            "uses_frp_weights": weights is not None
        },
        model_path,
        compress=3
    )

    print(f"[INFO] DBSCAN model saved → {model_path}")

    with open(os.path.join(MODEL_DIR, "dbscan_params.json"), "w", encoding="utf-8") as f:
      json.dump(best_params, f, indent=2)

    report_path = os.path.join(EVAL_DIR, "dbscan_report.txt")

    with open(report_path, "w", encoding="utf-8") as f:
     f.write("\n".join(report_lines))

    print(f"\n[DONE] Reports → {EVAL_DIR}")
    print(f"[DONE] Model   → {MODEL_DIR}")


if __name__ == "__main__":
    main()