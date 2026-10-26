# Broader model benchmark for CAVD percolation classification: adds SVM, k-NN, Logistic
# Regression, and MLP alongside Random Forest, on the exact same splits as train_baseline.py.
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

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


def make_models():
    return {
        "RandomForest": RandomForestClassifier(n_estimators=400, max_depth=12, min_samples_leaf=2,
                                                class_weight="balanced_subsample", n_jobs=8, random_state=42),
        "SVM": SVC(kernel="rbf", C=5.0, class_weight="balanced", random_state=42),
        "kNN": KNeighborsClassifier(n_neighbors=15, weights="distance"),
        "LogisticRegression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
        "MLP": MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", alpha=1e-3,
                              max_iter=2000, early_stopping=True, random_state=42),
    }


NEEDS_SCALING = {"SVM", "kNN", "LogisticRegression", "MLP"}


def fit_eval(name, model, X, y, train_m, test_m, scaler):
    if name in NEEDS_SCALING:
        Xtr = scaler.fit_transform(X[train_m])
        Xte = scaler.transform(X[test_m])
    else:
        Xtr, Xte = X[train_m], X[test_m]
    model.fit(Xtr, y[train_m])
    pred = model.predict(Xte)
    return {
        "n_train": int(train_m.sum()), "n_test": int(test_m.sum()),
        "accuracy": float(accuracy_score(y[test_m], pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y[test_m], pred)),
    }


def main():
    t0 = time.time()
    df, X = load_data()
    y = df["percolates"].to_numpy()
    log(f"n={len(df)}, {X.shape[1]} features")

    idx = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        idx, test_size=0.2, random_state=42,
        stratify=df[["cation", "percolates"]].astype(str).agg("_".join, axis=1),
    )
    random_train = np.zeros(len(df), dtype=bool); random_train[train_idx] = True
    random_test = np.zeros(len(df), dtype=bool); random_test[test_idx] = True
    mono_mask = df["cation"].isin(MONOVALENT).to_numpy()
    multi_mask = df["cation"].isin(MULTIVALENT).to_numpy()

    protocols = {
        "random_split": (random_train, random_test),
        "train_mono_test_multi": (mono_mask, multi_mask),
        "train_multi_test_mono": (multi_mask, mono_mask),
    }

    scaler = StandardScaler()
    models = make_models()
    results = {name: {} for name in models}

    for name, model in models.items():
        log(f"=== {name} ===")
        for pname, (train_m, test_m) in protocols.items():
            m = fit_eval(name, model, X, y, train_m, test_m, scaler)
            results[name][pname] = m
            log(f"  {pname}: acc={m['accuracy']:.3f} bal_acc={m['balanced_accuracy']:.3f}")
        # leave-one-cation-out (all 4 cations)
        per_cation = {}
        for cat in ["Li", "Na", "Mg", "Al"]:
            test_m = (df["cation"] == cat).to_numpy()
            train_m = ~test_m
            per_cation[cat] = fit_eval(name, model, X, y, train_m, test_m, scaler)
        results[name]["leave_one_cation_out"] = per_cation
        mean_bal = np.mean([per_cation[c]["balanced_accuracy"] for c in per_cation])
        log(f"  leave-one-cation-out mean balanced_accuracy={mean_bal:.3f}")

    results["runtime_seconds"] = time.time() - t0
    with open(OUT_DIR / "benchmark_models.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    log(f"DONE in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
