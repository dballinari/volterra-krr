"""Kernel ridge regression with the Volterra reservoir kernel."""

from __future__ import annotations

import warnings

import numpy as np
from scipy import linalg
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

from ._kernel import _as_2d, volterra_kernel, zero_history_value

__all__ = ["VolterraKernelRidge"]


class VolterraKernelRidge(RegressorMixin, BaseEstimator):
    """Kernel ridge regression with the Volterra kernel for time series.

    Implements the infinite-dimensional next-generation reservoir computing
    estimator of Grigoryeva, Ting & Ortega (arXiv:2412.09800): ridge
    regression in the reproducing kernel Hilbert space of the Volterra
    kernel, which spans polynomial features of all orders on all past lags.

    Each row of ``X`` is the input at one time step, and **rows must be in
    time order**: the prediction at step ``i`` depends on the whole history
    ``X[:i+1]``. The typical one-step-ahead forecasting setup is
    ``fit(series[:-1], series[1:])``.

    Parameters
    ----------
    lag_decay : float, default=0.5
        Kernel parameter ``lambda`` of the paper, in ``(0, 1)``. Smaller values
        forget the past faster.
    theta : float, default=0.5
        Kernel parameter ``theta`` of the paper. Larger values put more weight
        on high-order monomials.
    alpha : float, default=1e-6
        Ridge regularization strength.
    washout : int, default=0
        Number of initial training samples excluded from the regression. Their
        inputs still enter the history of later samples.
    scale : bool, default=True
        If True, inputs are demeaned and divided by the largest training-set
        Euclidean norm, as in the paper, so that ``max ||x|| = 1``. Then the
        admissible parameters are ``theta < 1`` and
        ``lag_decay < sqrt(1 - theta**2)``.
    initial_value : float, default=None
        Kernel value against an empty (zero-padded) history. Defaults to
        ``1 / (1 - lag_decay**2)``.

    Attributes
    ----------
    dual_coef_ : ndarray of shape (n_samples - washout,) or (n_samples - washout, n_targets)
        Coefficients of the kernel expansion.
    X_fit_ : ndarray of shape (n_samples, n_features)
        Training inputs after scaling.
    mean_, scale_ : ndarray / float
        Scaling applied to inputs (identity if ``scale=False``).
    n_features_in_ : int
    """

    def __init__(
        self,
        lag_decay: float = 0.5,
        theta: float = 0.5,
        alpha: float = 1e-6,
        washout: int = 0,
        scale: bool = True,
        initial_value: float | None = None,
    ):
        self.lag_decay = lag_decay
        self.theta = theta
        self.alpha = alpha
        self.washout = washout
        self.scale = scale
        self.initial_value = initial_value

    # ------------------------------------------------------------------ helpers
    def _transform(self, X) -> np.ndarray:
        X = _as_2d(X, "X")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but the model was fitted with {self.n_features_in_}"
            )
        return (X - self.mean_) / self.scale_

    def _kernel(self, X, Y, boundary=None) -> np.ndarray:
        return volterra_kernel(
            X,
            Y,
            lag_decay=self.lag_decay,
            theta=self.theta,
            initial_value=self.initial_value,
            boundary=boundary,
        )

    def _k0(self) -> float:
        if self.initial_value is None:
            return zero_history_value(self.lag_decay)
        return float(self.initial_value)

    # ---------------------------------------------------------------------- API
    def fit(self, X, y):
        """Fit the model on a time series.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or (n_samples,)
            Input time series, ordered in time.
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Targets aligned with ``X``.
        """
        X = _as_2d(X, "X")
        y = np.asarray(y, dtype=float)
        if y.shape[0] != X.shape[0]:
            raise ValueError(f"X and y have different lengths: {X.shape[0]} and {y.shape[0]}")
        if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
            raise ValueError("X and y must be finite")
        washout = int(self.washout)
        if not 0 <= washout < X.shape[0]:
            raise ValueError(f"washout must be in [0, n_samples), got {self.washout}")
        if self.alpha < 0:
            raise ValueError(f"alpha must be non-negative, got {self.alpha}")

        self.n_features_in_ = X.shape[1]
        if self.scale:
            self.mean_ = X.mean(axis=0)
            norms = np.linalg.norm(X - self.mean_, axis=1)
            max_norm = norms.max()
            self.scale_ = max_norm if max_norm > 0 else 1.0
        else:
            self.mean_ = np.zeros(X.shape[1])
            self.scale_ = 1.0
        Z = (X - self.mean_) / self.scale_

        M = float(np.max(np.sum(Z**2, axis=1)))
        if self.lag_decay**2 >= 1.0 - self.theta**2 * M:
            warnings.warn(
                "lag_decay**2 >= 1 - theta**2 * max||x||**2: the Volterra kernel series is "
                "not guaranteed to converge on the training inputs. Consider smaller "
                "lag_decay or theta.",
                RuntimeWarning,
                stacklevel=2,
            )

        K = self._kernel(Z, Z)
        Kw = K[washout:, washout:]
        yw = y[washout:]
        A = Kw + self.alpha * np.eye(Kw.shape[0])
        try:
            self.dual_coef_ = linalg.solve(A, yw, assume_a="pos")
        except (linalg.LinAlgError, ValueError):
            self.dual_coef_ = linalg.lstsq(A, yw)[0]

        self.X_fit_ = Z
        self._last_train_column = K[:, -1].copy()
        self._y_last = y[-1].copy() if y.ndim > 1 else np.array(y[-1])
        self._y_ndim = y.ndim
        return self

    def predict(self, X, continue_series: bool = False):
        """Predict targets for a time series.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or (n_samples,)
            Input time series, ordered in time.
        continue_series : bool, default=False
            If True, ``X`` is treated as the continuation of the training
            series, so its history includes all training inputs. If False,
            ``X`` is a new series whose past is zero-padded (use a washout).

        Returns
        -------
        y_pred : ndarray of shape (n_samples,) or (n_samples, n_targets)
        """
        check_is_fitted(self, "dual_coef_")
        Z = self._transform(X)
        boundary = self._last_train_column if continue_series else None
        Kc = self._kernel(self.X_fit_, Z, boundary=boundary)
        return Kc[int(self.washout):].T @ self.dual_coef_

    def forecast(self, n_steps: int, start=None):
        """Autonomous multi-step forecast by feeding predictions back as inputs.

        Requires a model trained for one-step-ahead prediction, i.e.
        ``fit(series[:-1], series[1:])``, so that targets and inputs live in the
        same space. The forecast continues the training series.

        Parameters
        ----------
        n_steps : int
            Number of steps to forecast.
        start : array-like of shape (n_features,), default=None
            The first input following the training inputs. Defaults to the last
            training target (``series[-1]`` in the setup above).

        Returns
        -------
        y_forecast : ndarray of shape (n_steps, n_features)
            ``y_forecast[k]`` is the prediction of ``series[n + 1 + k]`` where
            ``series[n]`` is ``start``.
        """
        check_is_fitted(self, "dual_coef_")
        n_targets = 1 if self._y_ndim == 1 else self.dual_coef_.shape[1]
        if n_targets != self.n_features_in_:
            raise ValueError(
                "forecast requires the number of targets to equal the number of features "
                f"({n_targets} != {self.n_features_in_})"
            )
        x = self._y_last if start is None else np.asarray(start, dtype=float)
        x = np.atleast_1d(x).reshape(self.n_features_in_)

        k0 = self._k0()
        lam2, th2 = self.lag_decay**2, self.theta**2
        w = int(self.washout)
        col = self._last_train_column
        out = np.empty((n_steps, self.n_features_in_))
        for k in range(n_steps):
            z = (x - self.mean_) / self.scale_
            denom = 1.0 - th2 * (self.X_fit_ @ z)
            if np.any(denom <= 0.0):
                raise ValueError(
                    f"forecast diverged at step {k}: theta**2 * <x, z> >= 1. "
                    "The predicted trajectory left the region where the kernel is defined."
                )
            prev = np.empty_like(col)
            prev[0] = k0
            prev[1:] = col[:-1]
            col = 1.0 + lam2 * prev / denom
            y_hat = col[w:] @ self.dual_coef_
            out[k] = np.atleast_1d(y_hat)
            x = out[k]
        return out
