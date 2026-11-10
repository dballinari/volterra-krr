import numpy as np
import pytest
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from volterra_krr import VolterraKernelRidge, volterra_kernel, zero_history_value


def series_kernel(z, zp, lam, theta, n_terms=400):
    """Kernel between two zero-padded histories via the explicit series."""
    L = max(len(z), len(zp))
    zz = np.zeros((n_terms, z.shape[1]))
    zzp = np.zeros((n_terms, z.shape[1]))
    zz[: len(z)] = z[::-1]
    zzp[: len(zp)] = zp[::-1]
    assert n_terms > L
    inner = np.sum(zz * zzp, axis=1)
    factors = 1.0 / (1.0 - theta**2 * inner)
    prods = np.cumprod(factors)
    taus = np.arange(1, n_terms + 1)
    return 1.0 + np.sum(lam ** (2 * taus) * prods)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def test_gram_matches_series_definition(rng):
    X = rng.uniform(-0.5, 0.5, size=(6, 2))
    Y = rng.uniform(-0.5, 0.5, size=(4, 2))
    lam, theta = 0.6, 0.7
    K = volterra_kernel(X, Y, lag_decay=lam, theta=theta)
    for i in range(len(X)):
        for j in range(len(Y)):
            expected = series_kernel(X[: i + 1], Y[: j + 1], lam, theta)
            assert K[i, j] == pytest.approx(expected, rel=1e-10)


def test_gram_symmetric_psd(rng):
    X = rng.uniform(-0.5, 0.5, size=(60, 3))
    K = volterra_kernel(X, lag_decay=0.5, theta=0.8)
    np.testing.assert_allclose(K, K.T, atol=1e-12)
    assert np.linalg.eigvalsh(K).min() > -1e-8


def test_zero_history_value():
    K = volterra_kernel(np.zeros((1, 2)), np.ones((1, 2)) * 0.3, lag_decay=0.5, theta=0.5)
    assert K[0, 0] == pytest.approx(zero_history_value(0.5))


def test_boundary_equals_concatenated_series(rng):
    X = rng.uniform(-0.5, 0.5, size=(30, 2))
    Y = rng.uniform(-0.5, 0.5, size=(10, 2))
    kw = dict(lag_decay=0.7, theta=0.5)
    K_train = volterra_kernel(X, **kw)
    K_cont = volterra_kernel(X, Y, boundary=K_train[:, -1], **kw)
    K_full = volterra_kernel(X, np.vstack([X, Y]), **kw)
    np.testing.assert_allclose(K_cont, K_full[:, len(X):], rtol=1e-12)


def test_invalid_theta_raises():
    with pytest.raises(ValueError):
        volterra_kernel(np.ones((3, 1)) * 2, lag_decay=0.5, theta=0.9)


def lorenz(n, dt=0.01, x0=(1.0, 1.0, 1.0)):
    def f(s):
        x, y, z = s
        return np.array([10 * (y - x), x * (28 - z) - y, x * y - 8 / 3 * z])

    s = np.array(x0, dtype=float)
    out = np.empty((n, 3))
    for i in range(n):
        k1 = f(s)
        k2 = f(s + dt / 2 * k1)
        k3 = f(s + dt / 2 * k2)
        k4 = f(s + dt * k3)
        s = s + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        out[i] = s
    return out


def test_one_step_lorenz():
    # train on a long stretch so the test inputs stay inside the training attractor
    data = lorenz(4000)[500:]
    train, test = data[:3000], data[3000:3300]
    model = VolterraKernelRidge(lag_decay=0.3, theta=0.3, alpha=1e-10, washout=50)
    model.fit(train[:-1], train[1:])
    pred = model.predict(np.vstack([train[-1:], test[:-1]]), continue_series=True)
    nrmse = np.sqrt(np.mean((pred - test) ** 2)) / test.std()
    assert nrmse < 1e-3


def test_forecast_matches_manual_iteration():
    data = lorenz(700)[100:]
    model = VolterraKernelRidge(lag_decay=0.3, theta=0.3, alpha=1e-6, washout=20)
    model.fit(data[:-1], data[1:])
    fc = model.forecast(5)
    # manual: feed predictions back through predict(continue_series=True)
    inputs = [data[-1]]
    for k in range(5):
        p = model.predict(np.array(inputs), continue_series=True)[-1]
        np.testing.assert_allclose(p, fc[k], rtol=1e-9, atol=1e-9)
        inputs.append(p)


def test_forecast_short_horizon_accuracy():
    data = lorenz(4000)[500:]
    train, test = data[:3000], data[3000:]
    model = VolterraKernelRidge(lag_decay=0.3, theta=0.3, alpha=1e-10, washout=50)
    model.fit(train[:-1], train[1:])
    fc = model.forecast(100)  # forecast[k] predicts test[k]
    err = np.abs(fc - test[:100]).max() / test.std()
    assert err < 0.01


def test_univariate_and_sklearn_api():
    t = np.linspace(0, 20, 400)
    x = np.sin(t)
    model = VolterraKernelRidge(washout=10)
    model.fit(x[:-1], x[1:])
    assert model.predict(x[:-1]).shape == (399,)
    assert model.forecast(3).shape == (3, 1)
    assert model.score(x[:-1], x[1:]) > 0.99
    clone(model)
    gs = GridSearchCV(
        VolterraKernelRidge(washout=10),
        {"lag_decay": [0.3, 0.6], "theta": [0.3, 0.6]},
        cv=TimeSeriesSplit(3),
    ).fit(x[:-1, None], x[1:])
    assert set(gs.best_params_) == {"lag_decay", "theta"}
