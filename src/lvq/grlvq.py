"""grlvq.py – Generalised Relevance LVQ (GRLVQ).

Extends GLVQ with per-feature relevance weights that are learned jointly
with the prototypes.

Reference: Hammer & Villmann (2002).
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from base import BaseLVQ


class GRLVQ(BaseLVQ):
    """GLVQ with adaptive feature-relevance weights.

    Args:
        num_prototypes_per_class: Number of prototypes per class.
        initialization_type: ``"mean"`` or ``"random"``.
        prototype_update_learning_rate: Learning rate for prototype updates.
        weight_update_learning_rate: Learning rate for relevance-weight updates.
    """

    def __init__(
        self,
        num_prototypes_per_class: int = 1,
        initialization_type: str = "mean",
        prototype_update_learning_rate: float = 0.01,
        weight_update_learning_rate: float = 0.01,
    ) -> None:
        super().__init__(
            num_prototypes_per_class,
            initialization_type,
            prototype_update_learning_rate,
        )
        self.eps_zero = weight_update_learning_rate
        self.weight: np.ndarray | None = None

    # ── Internals ─────────────────────────────────────────────────────────────

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def _sigmoid_prime(x: np.ndarray) -> np.ndarray:
        s = 1.0 / (1.0 + np.exp(-x))
        return s * (1.0 - s)

    def _init_weights(self, n_features: int) -> np.ndarray:
        """Uniform relevance weights, summing to 1.

        Args:
            n_features: Number of input features.

        Returns:
            Weight vector, shape ``(n_features,)``.
        """
        return np.full(n_features, fill_value=1.0 / n_features)

    def _update(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        alpha: float,
        eps: float,
    ) -> None:
        """One pass of prototype and weight updates (in-place).

        Args:
            data: Shape ``(n_samples, n_features)``.
            labels: Shape ``(n_samples,)``.
            alpha: Prototype learning rate.
            eps: Weight learning rate.
        """
        for xi, x_label in zip(data, labels):
            mask_same = self.proto_labels == x_label
            mask_diff = ~mask_same

            # Vectorised weighted distances: (P,)
            dists = self.weighted_distances(xi, self.prototypes, self.weight)

            same_global = np.where(mask_same)[0]
            diff_global = np.where(mask_diff)[0]
            idx_a = same_global[np.argmin(dists[mask_same])]
            idx_b = diff_global[np.argmin(dists[mask_diff])]

            d_a = dists[idx_a]
            d_b = dists[idx_b]
            denom = (d_a + d_b) ** 2
            mu = (d_a - d_b) / (d_a + d_b)
            f_prime = self._sigmoid(mu) * (1.0 - self._sigmoid(mu))

            # Prototype update
            self.prototypes[idx_a] += (
                alpha * f_prime * (d_b / denom) * (xi - self.prototypes[idx_a])
            )
            self.prototypes[idx_b] -= (
                alpha * f_prime * (d_a / denom) * (xi - self.prototypes[idx_b])
            )

            # Weight gradient (vectorised over features)
            grad_w = self._sigmoid_prime(mu) * (
                (d_b / denom) * (xi - self.prototypes[idx_a]) ** 2
                - (d_a / denom) * (xi - self.prototypes[idx_b]) ** 2
            )
            self.weight -= eps * grad_w
            self.weight = np.clip(self.weight, a_min=0.0, a_max=None)
            self.weight /= self.weight.sum()

    def _cost(self, data: np.ndarray, labels: np.ndarray) -> float:
        """Mean sigmoid loss (with relevance weights).

        Args:
            data: Shape ``(n_samples, n_features)``.
            labels: Shape ``(n_samples,)``.

        Returns:
            Scalar loss.
        """
        losses = []
        for xi, x_label in zip(data, labels):
            mask_same = self.proto_labels == x_label
            mask_diff = ~mask_same
            dists = self.weighted_distances(xi, self.prototypes, self.weight)
            d_a = dists[mask_same].min()
            d_b = dists[mask_diff].min()
            mu = (d_a - d_b) / (d_a + d_b)
            losses.append(float(self._sigmoid(mu)))
        return float(np.mean(losses))

    # ── Public API ────────────────────────────────────────────────────────────

    def fit(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        epochs: int = 100,
        decay_scheme: bool = True,
        plot_loss: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Train GRLVQ.

        Args:
            data: Training data, shape ``(n_samples, n_features)``.
            labels: Training labels, shape ``(n_samples,)``.
            epochs: Number of training epochs.
            decay_scheme: Whether to use exponential learning-rate decay.
            plot_loss: If ``True``, plot the loss curve after training.

        Returns:
            Tuple ``(prototypes, proto_labels, weights, final_loss)``.
        """
        self.proto_labels, self.prototypes = self.initialization(data, labels)
        self.prototypes = self.prototypes.astype(float)
        self.weight = self._init_weights(data.shape[1])
        loss_history: list[float] = []

        for epoch in tqdm(range(epochs), desc="GRLVQ training", unit="epoch"):
            alpha = (
                self._decay(self.alpha_zero, epoch, epochs)
                if decay_scheme
                else self.alpha_zero
            )
            eps = (
                self._decay(self.eps_zero, epoch, epochs)
                if decay_scheme
                else self.eps_zero
            )
            self._update(data, labels, alpha, eps)
            err = self._cost(data, labels)
            loss_history.append(err)

        if plot_loss:
            plt.plot(loss_history)
            plt.xlabel("Epoch")
            plt.ylabel("Mean sigmoid loss")
            plt.title("GRLVQ training loss")
            plt.tight_layout()

        return self.prototypes, self.proto_labels, self.weight, loss_history[-1]
