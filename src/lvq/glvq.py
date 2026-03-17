"""glvq.py – Generalised Learning Vector Quantisation (GLVQ).

Reference: Sato & Yamada (1996).
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from base import BaseLVQ


class GLVQ(BaseLVQ):
    """Generalised LVQ with sigmoid cost function.

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
        super().__init__(num_prototypes_per_class, initialization_type, learning_rate)

    # ── Internals ─────────────────────────────────────────────────────────────

    @staticmethod
    def _sigmoid(x: float | np.ndarray) -> float | np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    def _relative_distance(
        self,
        x: np.ndarray,
        mask_same: np.ndarray,
        mask_diff: np.ndarray,
    ) -> tuple[float, int, float, int]:
        """Compute d_a, index_a, d_b, index_b for one sample.

        Args:
            x: Sample, shape ``(n_features,)``.
            mask_same: Boolean mask for same-class prototypes.
            mask_diff: Boolean mask for different-class prototypes.

        Returns:
            Tuple ``(d_a, index_a, d_b, index_b)``.
        """
        dists = self.euclidean_distances(x, self.prototypes)

        same_dists = dists[mask_same]
        index_a = np.argmin(same_dists)
        d_a = same_dists[index_a]

        diff_dists = dists[mask_diff]
        index_b = np.argmin(diff_dists)
        d_b = diff_dists[index_b]

        return d_a, index_a, d_b, index_b

    def _update(self, data: np.ndarray, labels: np.ndarray, alpha: float) -> None:
        """One pass of prototype updates over the dataset (in-place).

        Args:
            data: Shape ``(n_samples, n_features)``.
            labels: Shape ``(n_samples,)``.
            alpha: Current learning rate.
        """
        for xi, x_label in zip(data, labels):
            mask_same = self.proto_labels == x_label
            mask_diff = ~mask_same

            dists = self.euclidean_distances(xi, self.prototypes)

            same_idx_local = np.argmin(dists[mask_same])
            diff_idx_local = np.argmin(dists[mask_diff])

            same_global = np.where(mask_same)[0]
            diff_global = np.where(mask_diff)[0]
            idx_a = same_global[same_idx_local]
            idx_b = diff_global[diff_idx_local]

            d_a = dists[idx_a]
            d_b = dists[idx_b]

            denom = (d_a + d_b) ** 2
            mu = (d_a - d_b) / (d_a + d_b)
            f_prime = self._sigmoid(mu) * (1.0 - self._sigmoid(mu))

            self.prototypes[idx_a] += (
                alpha * f_prime * (d_b / denom) * (xi - self.prototypes[idx_a])
            )
            self.prototypes[idx_b] -= (
                alpha * f_prime * (d_a / denom) * (xi - self.prototypes[idx_b])
            )

    def _cost(self, data: np.ndarray, labels: np.ndarray) -> float:
        """Mean sigmoid loss over the dataset.

        Args:
            data: Shape ``(n_samples, n_features)``.
            labels: Shape ``(n_samples,)``.

        Returns:
            Scalar loss value.
        """
        losses = []
        for xi, x_label in zip(data, labels):
            mask_same = self.proto_labels == x_label
            mask_diff = ~mask_same
            dists = self.euclidean_distances(xi, self.prototypes)
            d_a = dists[mask_same].min()
            d_b = dists[mask_diff].min()
            mu = (d_a - d_b) / (d_a + d_b)
            losses.append(self._sigmoid(mu))
        return float(np.mean(losses))

    # ── Public API ────────────────────────────────────────────────────────────

    def fit(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        epochs: int = 100,
        decay_scheme: bool = True,
        plot_loss: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Train GLVQ.

        Args:
            data: Training data, shape ``(n_samples, n_features)``.
            labels: Training labels, shape ``(n_samples,)``.
            epochs: Number of training epochs.
            decay_scheme: Whether to use exponential learning-rate decay.
            plot_loss: If ``True``, plot the loss curve after training.

        Returns:
            Tuple ``(prototypes, proto_labels)``.
        """
        self.proto_labels, self.prototypes = self.initialization(data, labels)
        self.prototypes = self.prototypes.astype(float)
        loss_history: list[float] = []

        for epoch in tqdm(range(epochs), desc="GLVQ training", unit="epoch"):
            alpha = (
                self._decay(self.alpha_zero, epoch, epochs)
                if decay_scheme
                else self.alpha_zero
            )
            self._update(data, labels, alpha)
            err = self._cost(data, labels)
            loss_history.append(err)

        if plot_loss:
            plt.plot(loss_history)
            plt.xlabel("Epoch")
            plt.ylabel("Mean sigmoid loss")
            plt.title("GLVQ training loss")
            plt.tight_layout()

        return self.prototypes, self.proto_labels

    def _loss(self, unit: np.ndarray, target_class: int) -> float:
        """Sigmoid loss contribution for a single unit towards a target class.

        Args:
            unit: Sample, shape ``(n_features,)``.
            target_class: The target class label.

        Returns:
            Sigmoid of relative distance.
        """
        mask_same = self.proto_labels == target_class
        mask_diff = ~mask_same
        dists = self.euclidean_distances(unit, self.prototypes)
        d_a = dists[mask_same].min()
        d_b = dists[mask_diff].min()
        return float(self._sigmoid((d_a - d_b) / (d_a + d_b)))
