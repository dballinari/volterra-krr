# volterra-krr

Kernel ridge regression with the **Volterra reservoir kernel**, as used in
*Infinite-dimensional next-generation reservoir computing* by L. Grigoryeva,
H. L. J. Ting and J.-P. Ortega ([arXiv:2412.09800](https://arxiv.org/abs/2412.09800)).

The Volterra kernel is a universal kernel on input sequences whose feature
space contains polynomial monomials of **every order on every past lag**. Ridge
regression in its RKHS gives a next-generation reservoir computing (NG-RC)
model with infinitely many features, at the cost of an `n × n` Gram matrix,
with only three hyperparameters.

## Install

```bash
pip install volterra-krr
```

## The kernel

For two input histories `z = (..., z_{-1}, z_0)` and `z'`,

$$
K(z, z') = 1 + \sum_{\tau=1}^{\infty} \lambda^{2\tau} \prod_{t=0}^{\tau-1}
\frac{1}{1-\theta^2\langle z_{-t}, z'_{-t}\rangle},
$$

which for zero-padded finite time series gives the Gram recursion

$$
K_{i,j} = 1 + \frac{\lambda^2 K_{i-1,j-1}}{1-\theta^2\langle z_i, z'_j\rangle},
\qquad K_{-1,\cdot} = K_{\cdot,-1} = \frac{1}{1-\lambda^2}.
$$

The kernel is well defined when `θ² · max‖z‖² < 1` and the series converges
when `λ² < 1 − θ² · max‖z‖²`. With the default scaling (`scale=True`: demean,
then divide by the largest training norm so `max‖z‖ = 1`) this means
`θ < 1` and `λ < sqrt(1 − θ²)`.

In the API, `λ` is `lag_decay`, `θ` is `theta`, and the ridge penalty is
`alpha`.

## Usage

```python
import numpy as np
from volterra_krr import VolterraKernelRidge

series = ...                     # array of shape (n_samples, n_features), in time order
train, test = series[:1000], series[1000:]

model = VolterraKernelRidge(lag_decay=0.3, theta=0.3, alpha=1e-8, washout=100)
model.fit(train[:-1], train[1:])  # one-step-ahead: x_t -> x_{t+1}

# one-step predictions on data continuing the training series
y_pred = model.predict(np.vstack([train[-1:], test[:-1]]), continue_series=True)

# autonomous multi-step forecast: predictions are fed back as inputs
forecast = model.forecast(200)    # predicts test[:200]
```

The estimator follows the scikit-learn API (`get_params`, `clone`, `score`,
`GridSearchCV`, …). Because each prediction depends on the full input history,
**rows must be in time order**; use `TimeSeriesSplit` for cross-validation.

- `predict(X)` treats `X` as a new series with a zero past. Pass
  `continue_series=True` when `X` directly follows the training inputs.
- `washout` drops the first samples from the regression, while keeping them in
  the history of later samples.
- The kernel itself is available as
  `volterra_kernel(X, Y, lag_decay=..., theta=...)`, e.g. for
  `sklearn.kernel_ridge.KernelRidge(kernel="precomputed")`.

See [`examples/lorenz.py`](examples/lorenz.py) for forecasting the Lorenz system.

## Notes on the implementation

- The ridge solution is `c = (K + αI)⁻¹ y`, which equals the paper's
  `(KᵀK + αK)⁻¹ K y` whenever `K` is invertible, and predictions are
  `ŷ = K(X_train, X_new)ᵀ c`.
- The paper initialises the Gram recursion with `1/(1−θ²)`. This package uses
  `1/(1−λ²)` by default, which is the exact kernel value against an all-zero
  history and keeps the recursion consistent with the series definition. The
  difference decays like `λ^{2i}` and is negligible after a washout. Pass
  `initial_value=1/(1-theta**2)` to reproduce the paper's choice exactly.
- Building the Gram matrix costs `O(n² d)` time and `O(n²)` memory.

## References

- L. Grigoryeva, H. L. J. Ting, J.-P. Ortega. *Infinite-dimensional
  next-generation reservoir computing.* arXiv:2412.09800.
- L. Gonon, L. Grigoryeva, J.-P. Ortega. *Reservoir kernels and Volterra
  series.* (introduces the Volterra reservoir kernel).

## License

MIT
