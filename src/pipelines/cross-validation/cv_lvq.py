#    def cross_validate(
#         self,
#         X: np.ndarray,
#         y: np.ndarray,
#         n_splits: int = 5,
#         verbose: bool = True,
#         **fit_kwargs,
#     ) -> dict[str, Any]:
#         """Stratified k-fold cross-validation.

#         A fresh copy of the model is created for each fold using the same
#         constructor arguments, so the original model is not modified.

#         Args:
#             X: Full feature matrix, shape ``(n_samples, n_features)``.
#             y: Labels, shape ``(n_samples,)``.
#             n_splits: Number of folds.
#             verbose: If ``True``, print per-fold accuracy.
#             **fit_kwargs: Forwarded to ``model.fit()`` for each fold.

#         Returns:
#             Dictionary with per-fold and mean/std accuracy.
#         """
#         skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
#         fold_accs: list[float] = []

#         print(f"[Pipeline] {n_splits}-fold CV for {self.model.__class__.__name__} …")

#         for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
#             X_tr, X_val = X[train_idx], X[val_idx]
#             y_tr, y_val = y[train_idx], y[val_idx]

#             # Fresh model with same hyperparameters
#             fold_model = self._clone_model()
#             fold_pipe = LVQPipeline(fold_model, scale=self.scale)
#             fold_pipe.train(X_tr, y_tr, **fit_kwargs)
#             fold_metrics = fold_pipe.evaluate(X_val, y_val, verbose=False)
#             acc = fold_metrics["test_accuracy"]
#             fold_accs.append(acc)

#             if verbose:
#                 print(f"  Fold {fold}/{n_splits} → accuracy: {acc:.2f}%")

#         mean_acc = float(np.mean(fold_accs))
#         std_acc = float(np.std(fold_accs))
#         cv_results = {
#             "cv_fold_accuracies": fold_accs,
#             "cv_mean_accuracy": round(mean_acc, 4),
#             "cv_std_accuracy": round(std_acc, 4),
#         }
#         self.metrics_.update(cv_results)

#         print(f"[Pipeline] CV result: {mean_acc:.2f}% ± {std_acc:.2f}%")
#         return cv_results
