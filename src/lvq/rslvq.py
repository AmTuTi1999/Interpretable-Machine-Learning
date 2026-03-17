"""rslvq.py – Robust Soft LVQ (RSLVQ).

Prototype learning via gradient ascent on the log-likelihood ratio of a
Gaussian mixture model. Supports both continuous and categorical features.

Reference: Seo & Obermayer (2003).
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from base import BaseLVQ


class RSLVQ(BaseLVQ):
    """Robust Soft LVQ with Gaussian kernel and likelihood-ratio optimisation.

    Args:
        num_prototypes_per_class: Number of prototypes per class.
        initialization_type: ``"mean"`` or ``"random"``.
        sigma: Kernel width (bandwidth parameter).
        learning_rate: Prototype learning rate.
        max_iter: Number of training iterations.
        cat_full: If ``True``, skip the jitter on mean initialisation (useful
            for categorical / mixed data).
    """

    def __init__(
        self,
        num_prototypes_per_class: int = 1,
        initialization_type: str = "mean",
        sigma: float = 1.0,
        learning_rate: float = 0.05,
        max_iter: int = 100,
        cat_full: bool = False,
    ) -> None:
        super().__init__(num_prototypes_per_class, initialization_type, learning_rate)
        self.sigma = sigma
        self.max_iter = max_iter
        self.cat_full = cat_full

    # ── Overridden initialisation (adds optional jitter) ─────────────────────

    def initialization(
        self, train_data: np.ndarray, train_labels: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Initialise prototypes with optional small random jitter.

        Args:
            train_data: Shape ``(n_samples, n_features)``.
            train_labels: Shape ``(n_samples,)``.

        Returns:
            Tuple ``(proto_labels, prototypes)``.
        """
        proto_labels, prototypes = super().initialization(train_data, train_labels)
        if not self.cat_full:
            jitter = 0.01 * self.sigma * np.random.uniform(-1.0, 1.0)
            prototypes = prototypes + jitter * prototypes
        return proto_labels, prototypes

    # ── Gaussian kernel helpers ───────────────────────────────────────────────

    def _kernel(self, x: np.ndarray, p: np.ndarray) -> float:
        """Gaussian kernel value between ``x`` and prototype ``p``.

        Args:
            x: Shape ``(n_features,)``.
            p: Shape ``(n_features,)``.

        Returns:
            Scalar kernel value.
        """
        coef = -1.0 / (2.0 * self.sigma**2)
        return float(np.exp(coef * np.sum((x - p) ** 2)))

    def _kernel_batch(self, x: np.ndarray) -> np.ndarray:
        """Kernel values from ``x`` to all prototypes (vectorised).

        Args:
            x: Shape ``(n_features,)``.

        Returns:
            Shape ``(P,)`` kernel values.
        """
        coef = -1.0 / (2.0 * self.sigma**2)
        sq_dists = np.sum((self.prototypes - x) ** 2, axis=1)  # (P,)
        return np.exp(coef * sq_dists)

    # ── Probability helpers ───────────────────────────────────────────────────

    def _Pl(self, x: np.ndarray, proto_idx: int) -> float:
        """p(prototype j | x) – responsibility of prototype j for x.

        Args:
            x: Shape ``(n_features,)``.
            proto_idx: Index into ``self.prototypes``.

        Returns:
            Scalar probability.
        """
        k_all = self._kernel_batch(x)
        return k_all[proto_idx] / k_all.sum()

    def _Pl_y(self, x: np.ndarray, proto_idx: int, x_label: int) -> float:
        """p(prototype j | x, correct class y) – conditional responsibility.

        Args:
            x: Shape ``(n_features,)``.
            proto_idx: Index into ``self.prototypes``.
            x_label: True class of ``x``.

        Returns:
            Scalar probability.
        """
        mask_same = self.proto_labels == x_label
        k_same = self._kernel_batch(x)[mask_same]
        k_j = self._kernel(x, self.prototypes[proto_idx])
        return k_j / k_same.sum()

    # ── Categorical distance (for mixed data) ─────────────────────────────────

    @staticmethod
    def _indicator_dist(a: np.ndarray, b: np.ndarray) -> int:
        """Hamming distance (number of differing features).

        Args:
            a: Feature vector.
            b: Feature vector.

        Returns:
            Integer Hamming distance.
        """
        return int(np.sum(a != b))

    # ── Training ─────────────────────────────────────────────────────────────

    def _gradient_ascent_step(self, data: np.ndarray, labels: np.ndarray) -> None:
        """One gradient-ascent pass over all training samples (in-place).

        Args:
            data: Shape ``(n_samples, n_features)``.
            labels: Shape ``(n_samples,)``.
        """
        c = 1.0 / (self.sigma**2)

        for xi, x_label in zip(data, labels):
            for j in range(self.prototypes.shape[0]):
                d = xi - self.prototypes[j]
                if self.proto_labels[j] == x_label:
                    self.prototypes[j] += (
                        self.alpha_zero
                        * (self._Pl_y(xi, j, x_label) - self._Pl(xi, j))
                        * c
                        * d
                    )
                else:
                    self.prototypes[j] -= self.alpha_zero * self._Pl(xi, j) * c * d

    def _log_likelihood_ratio(self, data: np.ndarray, labels: np.ndarray) -> float:
        """Compute the log-likelihood ratio (training objective).

        Args:
            data: Shape ``(n_samples, n_features)``.
            labels: Shape ``(n_samples,)``.

        Returns:
            Scalar log-likelihood ratio.
        """
        log_ll = 0.0
        for xi, x_label in zip(data, labels):
            k_all = self._kernel_batch(xi)
            k_same = k_all[self.proto_labels == x_label]
            log_ll += np.log(k_same.sum() + 1e-300) - np.log(k_all.sum() + 1e-300)
        return float(log_ll)

    def fit(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        show_plot: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Train RSLVQ.

        Args:
            data: Training data, shape ``(n_samples, n_features)``.
            labels: Training labels, shape ``(n_samples,)``.
            show_plot: If ``True``, plot the log-likelihood ratio curve.

        Returns:
            Tuple ``(prototypes, proto_labels)``.
        """
        self.proto_labels, self.prototypes = self.initialization(data, labels)
        self.prototypes = self.prototypes.astype(float)
        loss_history: list[float] = []

        for _ in range(self.max_iter):
            self._gradient_ascent_step(data, labels)
            predicted = self.predict_all(data)
            acc = (predicted == labels.flatten()).mean() * 100
            lr = self._log_likelihood_ratio(data, labels)
            print(f"Acc: {acc:.2f}%   log-likelihood ratio: {lr:.4f}")
            loss_history.append(lr)

        if show_plot:
            plt.plot(loss_history)
            plt.ylabel("Log-likelihood ratio")
            plt.xlabel("Iteration")
            plt.title("RSLVQ training")
            plt.tight_layout()

        return self.prototypes, self.proto_labels

    def _loss(self, unit: np.ndarray, target_class: int) -> float:
        """Posterior probability of the target-class prototype given ``unit``.

        Args:
            unit: Sample, shape ``(n_features,)``.
            target_class: Target class label.

        Returns:
            Scalar probability in ``[0, 1]``.
        """
        k_all = self._kernel_batch(unit)
        k_same = k_all[self.proto_labels == target_class]
        return float(k_same.sum() / (k_all.sum() + 1e-300))

    def predict_cat(
        self, x: np.ndarray, prototypes: np.ndarray, proto_labels: np.ndarray
    ) -> int:
        """Predict class for a categorical / mixed input using Hamming distance.

        Args:
            x: Sample, shape ``(n_features,)``.
            prototypes: Prototype array.
            proto_labels: Labels for each prototype.

        Returns:
            Predicted class label.
        """
        dists = np.array([self._indicator_dist(x, p) for p in prototypes])
        return proto_labels[np.argmin(dists)]
