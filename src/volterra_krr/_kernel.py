"""Volterra reservoir kernel (Gonon, Grigoryeva & Ortega; Grigoryeva, Ting & Ortega).

For two semi-infinite input sequences ``z`` and ``z'`` the Volterra kernel is

.. math::

    K(z, z') = 1 + \\sum_{\\tau=1}^{\\infty} \\lambda^{2\\tau}
        \\prod_{t=0}^{\\tau-1} \\frac{1}{1 - \\theta^2 \\langle z_{-t}, z'_{-t} \\rangle},

which satisfies the recursion

.. math::

    K(z, z') = 1 + \\lambda^2 \\frac{K(T_1 z, T_1 z')}{1 - \\theta^2 \\langle z_0, z'_0 \\rangle}.

For finite time series, padded with zeros into the past, the Gram matrix is
computed with the recursion

.. math::

    K_{i,j} = 1 + \\frac{\\lambda^2 K_{i-1,j-1}}{1 - \\theta^2 \\langle z_i, z'_j \\rangle},

where ``K_{-1,\\cdot}`` and ``K_{\\cdot,-1}`` are the kernel values against an
all-zero history.
"""

from __future__ import annotations

import numpy as np

__all__ = ["volterra_kernel", "zero_history_value"]


def zero_history_value(lag_decay: float) -> float:
    """Kernel value between a sequence and an all-zero history.

    Solves ``k = 1 + lag_decay**2 * k``, i.e. ``k = 1 / (1 - lag_decay**2)``.
    """
    return 1.0 / (1.0 - lag_decay**2)


def _as_2d(X, name: str) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    if X.ndim != 2:
        raise ValueError(f"{name} must be 1D or 2D, got shape {X.shape}")
    return X


def volterra_kernel(
    X,
    Y=None,
    *,
    lag_decay: float = 0.5,
    theta: float = 0.5,
    initial_value: float | None = None,
    boundary=None,
) -> np.ndarray:
    """Gram matrix of the Volterra kernel between two time series.

    Row ``i`` of ``X`` (resp. row ``j`` of ``Y``) is the most recent value of
    the input history ``(..., 0, 0, X[0], ..., X[i])``. Rows must therefore be
    ordered in time.

    Parameters
    ----------
    X : array-like of shape (n_samples_X, n_features) or (n_samples_X,)
        First time series, ordered in time.
    Y : array-like of shape (n_samples_Y, n_features), default=None
        Second time series. If ``None``, ``Y = X``.
    lag_decay : float, default=0.5
        The parameter ``lambda`` of the paper, controlling how fast past lags
        are forgotten. Must satisfy ``0 < lag_decay < 1``; the paper requires
        ``lag_decay**2 < 1 - theta**2 * M`` where ``M`` bounds the squared
        norm of the inputs.
    theta : float, default=0.5
        The parameter ``theta`` of the paper, controlling the weight of higher
        order monomials. Requires ``theta**2 * <x, y> < 1`` for all inputs.
    initial_value : float, default=None
        Kernel value assigned to pairs where (at least) one history is empty.
        Defaults to ``1 / (1 - lag_decay**2)``, the exact value against an
        all-zero history.
    boundary : array-like of shape (n_samples_X,), default=None
        Kernel values ``K(X[:i+1] history, history preceding Y[0])`` for each
        ``i``. Use it when ``Y`` continues another series, e.g. the training
        series, in which case pass the last column of the training Gram matrix.
        Defaults to ``initial_value`` everywhere (``Y`` starts from zeros).

    Returns
    -------
    K : ndarray of shape (n_samples_X, n_samples_Y)
    """
    X = _as_2d(X, "X")
    Y = X if Y is None else _as_2d(Y, "Y")
    if X.shape[1] != Y.shape[1]:
        raise ValueError(
            f"X and Y must have the same number of features, got {X.shape[1]} and {Y.shape[1]}"
        )
    if not 0.0 < lag_decay < 1.0:
        raise ValueError(f"lag_decay must be in (0, 1), got {lag_decay}")
    if theta < 0.0:
        raise ValueError(f"theta must be non-negative, got {theta}")

    k0 = zero_history_value(lag_decay) if initial_value is None else float(initial_value)
    n, m = X.shape[0], Y.shape[0]

    denom = 1.0 - theta**2 * (X @ Y.T)
    if np.any(denom <= 0.0):
        raise ValueError(
            "theta**2 * <x, y> >= 1 for some inputs; reduce theta or rescale the inputs "
            "(the Volterra kernel requires theta**2 * max||x||**2 < 1)."
        )
    factor = lag_decay**2 / denom

    if boundary is None:
        boundary = np.full(n, k0)
    else:
        boundary = np.asarray(boundary, dtype=float).ravel()
        if boundary.shape[0] != n:
            raise ValueError(f"boundary must have length {n}, got {boundary.shape[0]}")

    K = np.empty((n, m))
    prev = np.empty(m)
    if m == 0:
        return K
    # Row i = -1 corresponds to the empty history of X.
    prev_row = np.full(m, k0)
    corner = k0  # K(empty X history, history preceding Y[0])
    for i in range(n):
        prev[0] = corner
        prev[1:] = prev_row[:-1]
        K[i] = 1.0 + factor[i] * prev
        prev_row = K[i]
        corner = boundary[i]
    return K
