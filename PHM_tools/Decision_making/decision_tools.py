from __future__ import annotations

"""Simple decision-making tools using shallow machine learning models."""

from typing import Iterable

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import SVC
from smolagents import tool

from utils.registry import register_tool


@register_tool("isolation_forest_detector")
@tool
def isolation_forest_detector(
    data: Iterable[Iterable[float]] | np.ndarray,
    contamination: float = 0.1,
    random_state: int | None = None,
) -> np.ndarray:
    """Detect anomalies in a feature matrix using Isolation Forest.

    Args:
        data: Samples with shape ``(N, F)``.
        contamination: Proportion of outliers in the data.
        random_state: Seed for reproducibility.

    Returns:
        Array of ``1`` for inliers and ``-1`` for outliers.
    """
    X = np.asarray(list(data))
    clf = IsolationForest(contamination=contamination, random_state=random_state)
    return clf.fit_predict(X)


@register_tool("svm_fault_classifier")
@tool
def svm_fault_classifier(
    x_train: Iterable[Iterable[float]] | np.ndarray,
    y_train: Iterable[int] | np.ndarray,
    x_test: Iterable[Iterable[float]] | np.ndarray,
    kernel: str = "rbf",
    c: float = 1.0,
) -> np.ndarray:
    """Classify fault types with a support vector machine.

    Args:
        x_train: Training feature matrix shaped ``(N, F)``.
        y_train: Training labels.
        x_test: Features to classify.
        kernel: Kernel type for the SVM.
        c: Regularization parameter.

    Returns:
        Predicted labels for ``x_test``.
    """
    clf = SVC(kernel=kernel, C=c)
    X_train = np.asarray(list(x_train))
    y_train = np.asarray(list(y_train))
    X_test = np.asarray(list(x_test))
    clf.fit(X_train, y_train)
    return clf.predict(X_test)
