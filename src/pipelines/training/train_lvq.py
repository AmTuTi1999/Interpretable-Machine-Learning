"""train_pipeline.py – Training pipeline for all LVQ variants.

Provides a single ``LVQPipeline`` class that handles:
  - train / fit
  - evaluate (accuracy, precision, recall, F1, confusion matrix)
  - save / load  (joblib for model, JSON for metrics)
  - cross-validation
  - optional preprocessing (StandardScaler)

Usage example::

    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from glvq import GLVQ
    from train_pipeline import LVQPipeline

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = GLVQ(num_prototypes_per_class=1, learning_rate=0.01)
    pipeline = LVQPipeline(model, scale=True)

    pipeline.train(X_train, y_train, epochs=100)
    metrics = pipeline.evaluate(X_test, y_test, verbose=True)
    pipeline.save("models/glvq_iris")
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.lvq.base import BaseLVQ


def fit_and_evaluate(
    model: BaseLVQ,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    **fit_kwargs,
) -> dict[str, Any]:
    """Convenience function to fit a model and evaluate on test set."""
    print(f"[Pipeline] Training {model.__class__.__name__} …")
    t0 = time.perf_counter()
    model.fit(X_train, y_train, **fit_kwargs)
    elapsed = time.perf_counter() - t0

    train_acc = model.evaluate(X_train, y_train)
    print(f"[Pipeline] Done in {elapsed:.2f}s | " f"train accuracy: {train_acc:.2f}%")

    test_metrics = evaluate(model, X_test, y_test, verbose=True)

    return test_metrics


def evaluate(
    self,
    model: BaseLVQ,
    X_test: np.ndarray,
    y_test: np.ndarray,
    verbose: bool = True,
) -> dict[str, Any]:
    """Evaluate the fitted model on a held-out test set.

    Args:
        X_test: Test features, shape ``(n_samples, n_features)``.
        y_test: Test labels, shape ``(n_samples,)``.
        verbose: If ``True``, print a full classification report.

    Returns:
        Dictionary with accuracy, precision, recall, F1, and confusion
        matrix.
    """
    y_pred = model.predict_all(X_test)

    acc = accuracy_score(y_test, y_pred) * 100
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        "test_accuracy": round(acc, 4),
        "precision_weighted": round(float(prec), 4),
        "recall_weighted": round(float(rec), 4),
        "f1_weighted": round(float(f1), 4),
        "confusion_matrix": cm.tolist(),
    }

    if verbose:
        print(f"\n{'─'*55}")
        print(f"  Model : {self.model.__class__.__name__}")
        print(f"  Test accuracy : {acc:.2f}%")
        print(f"  Precision     : {prec:.4f}")
        print(f"  Recall        : {rec:.4f}")
        print(f"  F1 (weighted) : {f1:.4f}")
        print(f"{'─'*55}")
        print(classification_report(y_test, y_pred, zero_division=0))
        print("Confusion matrix:")
        print(cm)
        print(f"{'─'*55}\n")

    return metrics


def save_model(model: BaseLVQ, path: str | Path) -> None:
    """Save the fitted model, scaler, and metrics to disk.

    Creates three files:
        - ``<path>.joblib``  – serialised model + scaler
        - ``<path>_metrics.json``  – evaluation metrics

    Args:
        path: Base path (without extension), e.g. ``"models/glvq_iris"``.
    """
    model._check_fitted()
    base = Path(path)
    base.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model": model,
    }
    joblib.dump(payload, base.with_suffix(".joblib"))
    print(f"[Pipeline] Model saved → {base.with_suffix('.joblib')}")


def load_model(path: str | Path) -> BaseLVQ:
    """Load a fitted LVQ model previously saved with ``save_model()``.

    Args:
        path: Base path passed to ``save_model()`` (without extension),
            e.g. ``"models/glvq_iris"``.

    Returns:
        The restored fitted ``BaseLVQ`` model.

    Raises:
        FileNotFoundError: If the ``.joblib`` file does not exist at ``path``.
    """
    base = Path(path)
    joblib_path = base.with_suffix(".joblib")

    if not joblib_path.exists():
        raise FileNotFoundError(
            f"No model file found at '{joblib_path}'. "
            "Make sure the path matches what was passed to save_model()."
        )

    payload = joblib.load(joblib_path)
    model: BaseLVQ = payload["model"]

    print(f"[Pipeline] Loaded {model.__class__.__name__} " f"from {joblib_path}")
    return model
