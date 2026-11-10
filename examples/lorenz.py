"""Forecast the Lorenz system with Volterra kernel ridge regression.

Run with:  python examples/lorenz.py   (matplotlib optional, for the plot)
"""

import numpy as np

from volterra_krr import VolterraKernelRidge


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


data = lorenz(5000)[500:]
train, test = data[:3500], data[3500:]

model = VolterraKernelRidge(lag_decay=0.3, theta=0.3, alpha=1e-10, washout=100)
model.fit(train[:-1], train[1:])

one_step = model.predict(np.vstack([train[-1:], test[:-1]]), continue_series=True)
print("one-step NRMSE:", np.sqrt(np.mean((one_step - test) ** 2)) / test.std())

horizon = 500
forecast = model.forecast(horizon)
err = np.linalg.norm(forecast - test[:horizon], axis=1) / np.linalg.norm(test.std(axis=0))
valid = np.argmax(err > 0.3) if np.any(err > 0.3) else horizon
print(f"autonomous forecast stays within 30% error for {valid} steps ({valid * 0.01:.2f} time units)")

try:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 6))
    for k, ax in enumerate(axes):
        ax.plot(test[:horizon, k], label="truth")
        ax.plot(forecast[:, k], "--", label="forecast")
        ax.set_ylabel("xyz"[k])
    axes[0].legend()
    axes[-1].set_xlabel("step")
    plt.tight_layout()
    plt.show()
except ImportError:
    pass
