"""rlvq.py – Relevance LVQ (RLVQ).

Combines standard LVQ prototype updates with adaptive per-feature
relevance weights (no sigmoid cost – plain nearest-prototype classification).
"""

from __future__ import annotations

import numpy as np

from base import BaseLVQ


class RLVQ(BaseLVQ):
    """LVQ with adaptive feature-relevance weights.

    Args:
        num_prototypes_per_class: Number of prototypes per class.
        initialization_type: ``"mean"`` or ``"random"``.
        learning_rate: Initial prototype learning rate.
        max_iter: Maximum number of training iterations (kept for
            backwards-compatibility; prefer passing ``max_iter`` to ``fit``).
    """

    def __init__(
        self,
        num_prototypes_per_class: int = 1,
        initialization_type: str = "mean",
        learning_rate: float = 0.05,
        max_iter: int = 100,
    ) -> None:
        super().__init__(num_prototypes_per_class, initialization_type, learning_rate)
        self.max_iter = max_iter
        self.weight: np.ndarray | None = None

    # ── Internals ─────────────────────────────────────────────────────────────

    def _init_weights(self, n_features: int) -> np.ndarray:
        """Uniform relevance weights summing to 1.

        Args:
            n_features: Number of input features.

        Returns:
            Weight vector, shape ``(n_features,)``.
        """
        return np.full(n_features, fill_value=1.0 / n_features)

    def _weight_update(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        eps: float,
    ) -> None:
        """Update relevance weights for one pass over data (in-place).

        Args:
            data: Shape ``(n_samples, n_features)``.
            labels: Shape ``(n_samples,)``.
            eps: Weight learning rate.
        """
        for xi, x_label in zip(data, labels):
            # Vectorised weighted distances: (P,)
            dists = self.weighted_distances(xi, self.prototypes, self.weight)
            nearest_idx = np.argmin(dists)
            diff_sq = (xi - self.prototypes[nearest_idx]) ** 2  # (D,)

            if x_label == self.proto_labels[nearest_idx]:
                self.weight -= eps * (self.weight * diff_sq)
            else:
                self.weight += eps * (self.weight * diff_sq)

            self.weight = np.clip(self.weight, a_min=0.0, a_max=None)
            w_sum = self.weight.sum()
            if w_sum > 0:
                self.weight /= w_sum

    def _proto_update(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        alpha: float,
    ) -> None:
        """Update prototypes for one pass over data (in-place).

        Args:
            data: Shape ``(n_samples, n_features)``.
            labels: Shape ``(n_samples,)``.
            alpha: Prototype learning rate.
        """
        for xi, x_label in zip(data, labels):
            dists = self.weighted_distances(xi, self.prototypes, self.weight)
            nearest_idx = np.argmin(dists)
            delta = xi - self.prototypes[nearest_idx]

            if x_label == self.proto_labels[nearest_idx]:
                self.prototypes[nearest_idx] += alpha * delta
            else:
                self.prototypes[nearest_idx] -= alpha * delta

    # ── Public API ────────────────────────────────────────────────────────────

    def fit(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        eps_zero: float = 0.1,
        alpha_zero: float = 0.1,
        max_iter: int | None = None,
        decay_scheme: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Train RLVQ.

        Args:
            data: Training data, shape ``(n_samples, n_features)``.
            labels: Training labels, shape ``(n_samples,)``.
            eps_zero: Initial weight learning rate.
            alpha_zero: Initial prototype learning rate.
            max_iter: Number of iterations (overrides constructor value).
            decay_scheme: Whether to use exponential learning-rate decay.

        Returns:
            Tuple ``(prototypes, proto_labels, weights)``.
        """
        n_iters = max_iter if max_iter is not None else self.max_iter
        self.proto_labels, self.prototypes = self.initialization(data, labels)
        self.prototypes = self.prototypes.astype(float)
        self.weight = self._init_weights(data.shape[1])

        for iteration in range(n_iters):
            if decay_scheme:
                eps = self._decay(eps_zero, iteration, n_iters)
                alpha = self._decay(alpha_zero, iteration, n_iters)
            else:
                eps, alpha = eps_zero, alpha_zero

            self._weight_update(data, labels, eps)
            self._proto_update(data, labels, alpha)

        return self.prototypes, self.proto_labels, self.weight
