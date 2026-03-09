"""snpc.py – Soft Nearest Prototype Classifier (SNPC).

Minimises the expected misclassification rate via gradient descent on a
soft Gaussian mixture model (contrast with RSLVQ which maximises the
log-likelihood ratio).
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from base_lvq import BaseLVQ


class SNPC(BaseLVQ):
    """Soft Nearest Prototype Classifier.

    Args:
        num_prototypes_per_class: Number of prototypes per class.
        initialization_type: ``"mean"`` or ``"random"``.
        sigma: Kernel bandwidth.
        learning_rate: Prototype learning rate.
        max_iter: Number of training iterations.
        cat_full: If ``True``, skip initialisation jitter (categorical data).
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
        """Initialise with optional jitter scaled by sigma and num_prototypes.

        Args:
            train_data: Shape ``(n_samples, n_features)``.
            train_labels: Shape ``(n_samples,)``.

        Returns:
            Tuple ``(proto_labels, prototypes)``.
        """
        proto_labels, prototypes = super().initialization(train_data, train_labels)
        if not self.cat_full:
            jitter = (
                0.02 * self.sigma * self.num_prototypes * np.random.uniform(-1.0, 1.0)
            )
            prototypes = prototypes + jitter
        return proto_labels, prototypes

    # ── Gaussian kernel helpers ───────────────────────────────────────────────

    def _kernel_batch(self, x: np.ndarray) -> np.ndarray:
        """Kernel values from ``x`` to all prototypes (vectorised).

        Args:
            x: Shape ``(n_features,)``.

        Returns:
            Shape ``(P,)`` kernel values.
        """
        coef = -1.0 / (2.0 * self.sigma**2)
        sq_dists = np.sum((self.prototypes - x) ** 2, axis=1)
        return np.exp(coef * sq_dists)

    # ── Probability helpers ───────────────────────────────────────────────────

    def _Pl(self, x: np.ndarray, proto_idx: int) -> float:
        """p(prototype j | x) across all prototypes.

        Args:
            x: Shape ``(n_features,)``.
            proto_idx: Index into ``self.prototypes``.

        Returns:
            Scalar responsibility.
        """
        k_all = self._kernel_batch(x)
        return float(k_all[proto_idx] / (k_all.sum() + 1e-300))

    def _lst(self, x: np.ndarray, x_label: int) -> float:
        """Probability of misclassification of ``x`` (sum over wrong-class kernels).

        Args:
            x: Shape ``(n_features,)``.
            x_label: True class of ``x``.

        Returns:
            Scalar probability in ``[0, 1]``.
        """
        k_all = self._kernel_batch(x)
        k_wrong = k_all[self.proto_labels != x_label]
        return float(k_wrong.sum() / (k_all.sum() + 1e-300))

    # ── Training ─────────────────────────────────────────────────────────────

    def _gradient_descent_step(self, data: np.ndarray, labels: np.ndarray) -> None:
        """One gradient-descent pass over all training samples (in-place).

        Args:
            data: Shape ``(n_samples, n_features)``.
            labels: Shape ``(n_samples,)``.
        """
        c = 1.0 / (self.sigma**2)

        for xi, x_label in zip(data, labels):
            lst_val = self._lst(xi, x_label)

            for j in range(self.prototypes.shape[0]):
                d = xi - self.prototypes[j]
                p_j = self._Pl(xi, j)

                if self.proto_labels[j] == x_label:
                    self.prototypes[j] += self.alpha_zero * p_j * lst_val * c * d
                else:
                    self.prototypes[j] -= (
                        self.alpha_zero * p_j * (1.0 - lst_val) * c * d
                    )

    def _error_function(self, data: np.ndarray, labels: np.ndarray) -> float:
        """Expected misclassification rate (training objective to minimise).

        Args:
            data: Shape ``(n_samples, n_features)``.
            labels: Shape ``(n_samples,)``.

        Returns:
            Scalar expected error in ``[0, 1]``.
        """
        total = 0.0
        for xi, x_label in zip(data, labels):
            k_all = self._kernel_batch(xi)
            k_wrong = k_all[self.proto_labels != x_label]
            total += k_wrong.sum() / (k_all.sum() + 1e-300)
        return total / len(data)

    def fit(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        show_plot: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Train SNPC.

        Args:
            data: Training data, shape ``(n_samples, n_features)``.
            labels: Training labels, shape ``(n_samples,)``.
            show_plot: If ``True``, plot the error curve after training.

        Returns:
            Tuple ``(prototypes, proto_labels)``.
        """
        self.proto_labels, self.prototypes = self.initialization(data, labels)
        self.prototypes = self.prototypes.astype(float)
        loss_history: list[float] = []

        for _ in range(self.max_iter):
            self._gradient_descent_step(data, labels)
            predicted = self.predict_all(data)
            acc = (predicted == labels.flatten()).mean() * 100
            err = self._error_function(data, labels)
            print(f"Acc: {acc:.2f}%   expected error: {err:.4f}")
            loss_history.append(err)

        if show_plot:
            plt.plot(loss_history)
            plt.ylabel("Expected misclassification rate")
            plt.xlabel("Iteration")
            plt.title("SNPC training")
            plt.tight_layout()

        return self.prototypes, self.proto_labels

    def Pl_loss(self, unit: np.ndarray, target_class: int) -> float:
        """Posterior of correct-class prototypes for ``unit``.

        Args:
            unit: Shape ``(n_features,)``.
            target_class: Target class label.

        Returns:
            Scalar probability in ``[0, 1]``.
        """
        k_all = self._kernel_batch(unit)
        k_same = k_all[self.proto_labels == target_class]
        return float(k_same.sum() / (k_all.sum() + 1e-300))
