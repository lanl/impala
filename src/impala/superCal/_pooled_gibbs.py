"""Opt-in pooled Cu Gibbs bookkeeping; default sampler path stays unchanged."""

import numpy as np


class DiagonalGibbsBatch:
    """Batch independent temperatures without changing the gamma draw order."""

    def __init__(self, model, itl, ny, a, b, indices, weights, observed):
        from .models_withlik import AbstractModel, ModelMaterialStrength

        if (
            type(model) is not ModelMaterialStrength
            or model.s2 != "gibbs"
            or model.nd != 0
            or model.stochastic
            or getattr(model.step, "__func__", None) is not AbstractModel.step
            or getattr(model.llik, "__func__", None) is not AbstractModel.llik
            or getattr(model.lik_cov_inv, "__func__", None)
            is not AbstractModel.lik_cov_inv
        ):
            raise ValueError(
                "Batched Gibbs requires plain pooled Cu diagonal likelihoods"
            )
        self.itl = np.ascontiguousarray(itl)
        self.shape = np.ascontiguousarray(itl * (ny / 2 + a + 1) - 1)
        self.b = b
        self.indices = indices
        self.weights = weights
        self.observed = observed
        if observed.ndim != 1 or weights.shape != observed.shape:
            raise ValueError(
                "Batched Gibbs requires vector observations and weights"
            )

    def update(self, dev_sq, predictions, discrepancy):
        # C order is essential: match the existing temperature-then-error order.
        scale = np.ascontiguousarray(1 / (self.itl * (self.b + dev_sq / 2)))
        log_s2 = np.log(1 / np.random.gamma(self.shape, scale))
        variance = np.ascontiguousarray(np.exp(log_s2)[:, self.indices])
        inv = 1 / variance
        ldet = (self.weights * np.log(variance)).sum(axis=1)
        covariances = [
            {"inv": row, "ldet": value} for row, value in zip(inv, ldet)
        ]
        delta = (self.observed - discrepancy) - predictions
        squared = delta * delta * inv
        likelihood = -0.5 * ldet - 0.5 * (squared * self.weights).sum(axis=1)
        return log_s2, covariances, likelihood
