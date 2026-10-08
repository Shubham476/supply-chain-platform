"""
Train the booking-time delay-risk model.

Target: will this shipment arrive more than 24h late? (binary)
Only shipments with a known outcome (actual_arrival present) are used for
training/evaluation -- cancelled/unknown-status shipments have no ground
truth and are excluded.

We train and honestly compare two models:
  - Logistic Regression (baseline, interpretable)
  - Gradient Boosting (more flexible, usually a bit better recall)
and keep whichever has the higher AUC on the held-out test set, saving
both the model and the preprocessing pipeline together as one artefact so
the API can load it and call .predict_proba() directly.

Usage:
    python train_model.py --db ../data/warehouse.duckdb --out model.joblib
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import duckdb
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import (  # noqa: E402
    ALL_FEATURES,
    BOOKING_TIME_CATEGORICAL_FEATURES,
    BOOKING_TIME_NUMERIC_FEATURES,
    build_features,
    build_label,
)

RANDOM_STATE = 42


def load_training_frame(db_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        shipments = con.execute("SELECT * FROM curated_shipments").fetchdf()
        ports = con.execute("SELECT * FROM curated_ports").fetchdf()
    finally:
        con.close()

    features = build_features(shipments, ports)
    label = build_label(shipments)

    frame = features.copy()
    frame["label"] = label.values
    frame = frame.dropna(subset=["label"] + ALL_FEATURES)
    y = frame["label"].astype(int)
    X = frame[ALL_FEATURES]
    return X, y


def build_pipeline(estimator) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), BOOKING_TIME_NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), BOOKING_TIME_CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("model", estimator)])


def evaluate(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    proba = pipeline.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "auc": float(roc_auc_score(y_test, proba)),
        "n_test": int(len(y_test)),
        "positive_rate_test": float(y_test.mean()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the delay-risk model.")
    parser.add_argument("--db", default="../data/warehouse.duckdb")
    parser.add_argument("--out", default="model.joblib")
    parser.add_argument("--metrics-out", default="metrics.json")
    args = parser.parse_args()

    X, y = load_training_frame(Path(args.db))
    print(f"Training frame: {len(X)} rows, positive rate={y.mean():.3f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    candidates = {
        "logistic_regression": build_pipeline(
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
        ),
        "gradient_boosting": build_pipeline(
            GradientBoostingClassifier(random_state=RANDOM_STATE)
        ),
    }

    results = {}
    fitted = {}
    for name, pipeline in candidates.items():
        pipeline.fit(X_train, y_train)
        metrics = evaluate(pipeline, X_test, y_test)
        results[name] = metrics
        fitted[name] = pipeline
        print(f"{name}: {json.dumps(metrics, indent=2)}")

    best_name = max(results, key=lambda n: results[n]["auc"])
    best_pipeline = fitted[best_name]
    print(f"\nSelected model: {best_name} (highest AUC)")

    artefact = {
        "pipeline": best_pipeline,
        "model_name": best_name,
        "feature_columns": ALL_FEATURES,
        "on_time_threshold_hours": 24.0,
        "trained_rows": int(len(X_train)),
    }
    out_path = Path(args.out)
    joblib.dump(artefact, out_path)
    print(f"Model artefact saved to {out_path.resolve()}")

    metrics_out = {
        "selected_model": best_name,
        "candidates": results,
        "feature_columns": ALL_FEATURES,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }
    Path(args.metrics_out).write_text(json.dumps(metrics_out, indent=2))
    print(f"Metrics saved to {Path(args.metrics_out).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
