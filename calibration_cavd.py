# Calibration curve and Brier score for the CAVD percolation classifier's predicted
# probabilities, on the same three evaluation protocols as the point-estimate classifier.
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split

DATA_DIR = Path("/glade/work/ajalali/Multivalent-Ion-Conductor-ML-project/data")
OUT_DIR = Path("/glade/work/ajalali/Multivalent-Ion-Conductor-ML-project/models/baseline")

MONOVALENT = ["Li", "Na"]
MULTIVALENT = ["Mg", "Al"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_data():
    df = pd.read_parquet(DATA_DIR / "feature_table.parquet")
    drop_cols = ["cation", "filename", "IND", "percolates"]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols].to_numpy(dtype=float)
    col_mean = np.nanmean(X, axis=0)
    X = np.where(np.isnan(X), col_mean, X)
    return df, X


def make_model():
    return RandomForestClassifier(
        n_estimators=400, max_depth=12, min_samples_leaf=2,
        class_weight="balanced_subsample", n_jobs=8, random_state=42,
    )


def fit_eval(X, y, train_m, test_m, label):
    model = make_model()
    model.fit(X[train_m], y[train_m])
    proba = model.predict_proba(X[test_m])[:, 1]
    brier = float(brier_score_loss(y[test_m], proba))
    frac_pos, mean_pred = calibration_curve(y[test_m], proba, n_bins=10, strategy="quantile")
    log(f"  [{label}] Brier={brier:.4f}")
    return {
        "brier_score": brier,
        "calibration_curve_mean_predicted": mean_pred.tolist(),
        "calibration_curve_fraction_positive": frac_pos.tolist(),
    }


def main():
    t0 = time.time()
    df, X = load_data()
    y = df["percolates"].to_numpy()

    idx = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        idx, test_size=0.2, random_state=42,
        stratify=df[["cation", "percolates"]].astype(str).agg("_".join, axis=1),
    )
    random_train = np.zeros(len(df), dtype=bool); random_train[train_idx] = True
    random_test = np.zeros(len(df), dtype=bool); random_test[test_idx] = True
    mono_mask = df["cation"].isin(MONOVALENT).to_numpy()
    multi_mask = df["cation"].isin(MULTIVALENT).to_numpy()

    results = {
        "random_split": fit_eval(X, y, random_train, random_test, "random_split"),
        "mono_to_multi": fit_eval(X, y, mono_mask, multi_mask, "mono_to_multi"),
        "multi_to_mono": fit_eval(X, y, multi_mask, mono_mask, "multi_to_mono"),
        "runtime_seconds": time.time() - t0,
    }
    with open(OUT_DIR / "calibration_metrics.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    log(f"DONE in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
