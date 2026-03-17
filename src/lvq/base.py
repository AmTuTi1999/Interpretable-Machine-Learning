"""base_lvq.py – Abstract base class shared by all LVQ variants.

Extracted shared logic:
  - Prototype initialisation (mean / random)
  - Euclidean distance helpers (vectorised)
  - predict / predict_all / proba_predict / evaluate
  - Sklearn-compatible interface stubs
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import numpy as np


class BaseLVQ(ABC):
    """Abstract base for Learning Vector Quantisation classifiers.

    All LVQ variants inherit prototype initialisation, prediction, and
    evaluation from this class and only implement their own training loop.

    Args:
        num_prototypes_per_class: Number of prototypes per class.
        initialization_type: ``"mean"`` or ``"random"``.
        learning_rate: Initial learning rate (alpha_0).
    """

    def __init__(
        self,
        num_prototypes_per_class: int = 1,
        initialization_type: str = "mean",
        learning_rate: float = 0.01,
    ) -> None:
        self.num_prototypes = num_prototypes_per_class
        self.initialization_type = initialization_type
        self.alpha_zero = learning_rate

        # Set after fit()
        self.prototypes: np.ndarray | None = None
        self.proto_labels: np.ndarray | None = None

    # ── Initialisation ────────────────────────────────────────────────────────

    def initialization(
        self, train_data: np.ndarray, train_labels: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Initialise prototypes from training data.

        Args:
            train_data: Shape ``(n_samples, n_features)``.
            train_labels: Shape ``(n_samples,)``.

        Returns:
            Tuple of ``(proto_labels, prototypes)``.
        """
        labels = train_labels.astype(int)
        unique_labels = np.unique(labels)
        n_dims = train_data.shape[1]
        n_protos = self.num_prototypes * len(unique_labels)

        if self.initialization_type == "mean":
            return self._init_mean(train_data, labels, unique_labels, n_dims, n_protos)
        elif self.initialization_type == "random":
            return self._init_random(
                train_data, labels, unique_labels, n_dims, n_protos
            )
        else:
            raise ValueError(
                f"Unknown initialization_type '{self.initialization_type}'. "
                "Choose 'mean' or 'random'."
            )

    def _init_mean(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        unique_labels: np.ndarray,
        n_dims: int,
        n_protos: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Mean-based prototype initialisation."""
        proto_list: list[np.ndarray] = []
        label_list: list[int] = []

        for cls in unique_labels:
            class_data = data[labels == cls]
            mu = class_data.mean(axis=0)

            if self.num_prototypes == 1:
                proto_list.append(mu[np.newaxis, :])
                label_list.append([cls])
            else:
                # Closest (num_prototypes - 1) points to the mean
                sq_dists = np.sum((class_data - mu) ** 2, axis=1)
                nearest_idx = np.argsort(sq_dists)[1 : self.num_prototypes]
                block = np.vstack([mu, class_data[nearest_idx]])  # (num_prototypes, D)
                proto_list.append(block)
                label_list.extend([cls] * self.num_prototypes)

        prototypes = np.vstack(proto_list).reshape(n_protos, n_dims)
        proto_labels = np.array(label_list).flatten()
        return proto_labels, prototypes

    def _init_random(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        unique_labels: np.ndarray,
        n_dims: int,
        n_protos: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Random-sample prototype initialisation."""
        proto_list: list[np.ndarray] = []
        label_list: list[int] = []

        for cls in unique_labels:
            class_idx = np.flatnonzero(labels == cls)
            chosen = np.random.choice(
                class_idx, size=self.num_prototypes, replace=False
            )
            proto_list.append(data[chosen])
            label_list.extend([cls] * self.num_prototypes)

        prototypes = np.vstack(proto_list).reshape(n_protos, n_dims)
        proto_labels = np.array(label_list).flatten()
        return proto_labels, prototypes

    # ── Distance helpers ─────────────────────────────────────────────────────

    @staticmethod
    def euclidean_distances(x: np.ndarray, prototypes: np.ndarray) -> np.ndarray:
        """Vectorised squared Euclidean distances from ``x`` to all prototypes.

        Args:
            x: Shape ``(n_features,)``.
            prototypes: Shape ``(n_prototypes, n_features)``.

        Returns:
            Shape ``(n_prototypes,)`` – squared distances.
        """
        diff = prototypes - x  # (P, D)
        return np.sum(diff**2, axis=1)

    @staticmethod
    def weighted_distances(
        x: np.ndarray, prototypes: np.ndarray, weights: np.ndarray
    ) -> np.ndarray:
        """Vectorised squared weighted Euclidean distances.

        Args:
            x: Shape ``(n_features,)``.
            prototypes: Shape ``(n_prototypes, n_features)``.
            weights: Shape ``(n_features,)``.

        Returns:
            Shape ``(n_prototypes,)`` – squared weighted distances.
        """
        diff = prototypes - x  # (P, D)
        return np.sum((weights * diff) ** 2, axis=1)

    @staticmethod
    def _nearest_prototype_per_class(
        x: np.ndarray,
        prototypes: np.ndarray,
        proto_labels: np.ndarray,
        unique_labels: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (distances_to_nearest, nearest_prototypes) per class.

        Args:
            x: Query point, shape ``(n_features,)``.
            prototypes: All prototypes, shape ``(P, D)``.
            proto_labels: Labels for each prototype, shape ``(P,)``.
            unique_labels: Sorted unique class labels.

        Returns:
            Tuple ``(min_dists, nearest_protos)`` each of length
            ``len(unique_labels)``.
        """
        min_dists = []
        nearest_protos = []
        for cls in unique_labels:
            mask = proto_labels == cls
            class_protos = prototypes[mask]
            dists = np.sum((class_protos - x) ** 2, axis=1)
            idx = np.argmin(dists)
            min_dists.append(dists[idx])
            nearest_protos.append(class_protos[idx])
        return np.array(min_dists), np.array(nearest_protos)

    # ── Prediction ───────────────────────────────────────────────────────────

    def predict(self, x: np.ndarray) -> int:
        """Predict the class label for a single sample.

        Args:
            x: Shape ``(n_features,)``.

        Returns:
            Predicted class label.
        """
        dists = np.sum((self.prototypes - x) ** 2, axis=1)
        return self.proto_labels[np.argmin(dists)]

    def predict_all(
        self, data: np.ndarray, return_scores: bool = False
    ) -> np.ndarray | list:
        """Predict labels (or probability scores) for an array of samples.

        Args:
            data: Shape ``(n_samples, n_features)``.
            return_scores: If ``True``, return soft probability scores instead
                of hard labels.

        Returns:
            Array of predicted labels, or list of score arrays.
        """
        if return_scores:
            return [self.proba_predict(data[i]) for i in range(len(data))]

        # Vectorised: (N, P) distance matrix → argmin per row
        diff = data[:, np.newaxis, :] - self.prototypes[np.newaxis, :, :]  # (N, P, D)
        sq_dists = np.sum(diff**2, axis=2)  # (N, P)
        nearest = np.argmin(sq_dists, axis=1)  # (N,)
        return self.proto_labels[nearest]

    def proba_predict(self, x: np.ndarray, softmax: bool = False) -> np.ndarray:
        """Soft class scores based on distance to nearest class prototype.

        Smaller distance → higher score (scores are *inverted* distances,
        normalised to sum to 1).

        Args:
            x: Shape ``(n_features,)``.
            softmax: If ``True``, apply softmax over negative distances.

        Returns:
            Score array, shape ``(n_classes,)``.
        """
        unique_labels = np.unique(self.proto_labels)
        min_dists, _ = self._nearest_prototype_per_class(
            x, self.prototypes, self.proto_labels, unique_labels
        )
        total = min_dists.sum()
        scores = min_dists / total  # lower dist → lower raw score

        if softmax:
            exp_neg = np.exp(-min_dists)
            scores = exp_neg / exp_neg.sum()

        return scores

    def evaluate(self, test_data: np.ndarray, test_labels: np.ndarray) -> float:
        """Compute classification accuracy on a test set.

        Args:
            test_data: Shape ``(n_samples, n_features)``.
            test_labels: Shape ``(n_samples,)``.

        Returns:
            Accuracy in percent ``[0, 100]``.
        """
        predicted = self.predict_all(test_data)
        return (predicted == test_labels.flatten()).mean() * 100.0

    # ── Abstract interface ────────────────────────────────────────────────────

    @abstractmethod
    def fit(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        **kwargs,
    ) -> tuple:
        """Train the model. Must set ``self.prototypes`` and ``self.proto_labels``.

        Args:
            data: Training data, shape ``(n_samples, n_features)``.
            labels: Training labels, shape ``(n_samples,)``.
            **kwargs: Subclass-specific training arguments.
        """

    # ── Decay helper ─────────────────────────────────────────────────────────

    @staticmethod
    def _decay(rate_zero: float, epoch: int, total_epochs: int) -> float:
        """Exponential decay schedule.

        Args:
            rate_zero: Initial rate.
            epoch: Current epoch (0-indexed).
            total_epochs: Total number of epochs.

        Returns:
            Decayed rate.
        """
        return rate_zero * math.exp(-epoch / total_epochs)

    def __repr__(self) -> str:
        fitted = self.prototypes is not None
        return (
            f"{self.__class__.__name__}("
            f"num_prototypes={self.num_prototypes}, "
            f"init='{self.initialization_type}', "
            f"lr={self.alpha_zero}, "
            f"fitted={fitted})"
        )

    def _check_fitted(self) -> None:
        """Raise if the model has not been trained yet."""
        if self.prototypes is None:
            raise RuntimeError("Model is not fitted yet. Call pipeline.train() first.")
