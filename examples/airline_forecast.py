"""
Airline passengers forecast using a Kalman filter fitted via MLE.

Preprocessing: log → lag-1 difference → lag-12 difference.
The resulting stationary series is modelled as a 1-state / 1-obs
local-level model.  Parameters are optimised with scipy L-BFGS-B
using our model's log_likelihood method directly.
A 12-step out-of-sample forecast is produced and inverse-transformed
back to passenger counts.
"""

import csv
import os

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

from state_space_rs import LinearGaussianSSM

# ── 1. Load data ──────────────────────────────────────────────────────────────

csv_path = os.path.join(os.path.dirname(__file__), "airline-passengers.csv")
months, passengers = [], []
with open(csv_path) as f:
    reader = csv.DictReader(f)
    for row in reader:
        months.append(row["month"])
        passengers.append(float(row["total_passengers"]))

raw = np.array(passengers)

# ── 2. Preprocessing: log → Δ1 → Δ12 ────────────────────────────────────────

log_raw = np.log(raw)
diff1 = np.diff(log_raw, n=1)        # length N-1
diff12 = diff1[12:] - diff1[:-12]    # length N-1-12 = N-13

y = diff12  # stationary series to model

# ── 3. Build a 1×1 local-level model and define neg-log-likelihood ────────────

SIZE_STATE = 2
SIZE_OBS = 1




def neg_log_likelihood(theta: np.ndarray) -> float:
    """Negative log-likelihood using our model's built-in method."""
    model = LinearGaussianSSM(SIZE_STATE, SIZE_OBS)
    try:
        model.set_parameters(theta)
        ll = model.log_likelihood(y.reshape(-1, SIZE_OBS)) / SIZE_OBS
        if not np.isfinite(ll):
            return 1e10
        return -ll
    except Exception:
        return 1e10


# ── 4. Optimise ───────────────────────────────────────────────────────────────

# Parameter layout for 1×1 model (6 scalars):
#   [initial_mean, initial_cov_dec(diag), transition, observation,
#    q_dec(diag), r_dec(diag)]
# Bounds enforce stationarity (|transition| < 1) and positive-definite
# covariances (decomposition diagonals bounded away from zero).


model = LinearGaussianSSM(SIZE_STATE, SIZE_OBS)
x0 = model.get_parameters()

bounds = [
    (None, None) for _ in range(len(x0))
]

result = minimize(
    neg_log_likelihood, x0, method="TNC",
    options={"maxiter": 100, "ftol": 1e-12},
)
print(f"Optimization: success={result.success}, nll={result.fun:.4f}")
print(f"Optimal params: {result.x}")

# ── 5. Forecast 12 steps ahead ───────────────────────────────────────────────

fitted = LinearGaussianSSM(SIZE_STATE, SIZE_OBS)
fitted.set_parameters(result.x)

observations = y.reshape(-1, SIZE_OBS)
forecast_dists = fitted.forecast(observations, 24)

forecast_means = np.array([d.mean[0] for d in forecast_dists])
forecast_vars = np.array([d.cov[0, 0] for d in forecast_dists])
forecast_std = np.sqrt(forecast_vars)

# 90% prediction interval in transformed space
z90 = 1.645
fc_upper = forecast_means + z90 * forecast_std
fc_lower = forecast_means - z90 * forecast_std

# ── 6. Invert transforms ─────────────────────────────────────────────────────

n_fc = 24
n_total = len(raw)


def invert_forecast(fc_vals: np.ndarray) -> np.ndarray:
    """Invert Δ12 → Δ1 → log for a vector of 12 forecast values."""
    n_diff1 = len(diff1)
    extended_diff1 = list(diff1)
    for h in range(n_fc):
        # diff12_fc[h] = diff1[n_diff1 + h] - diff1[n_diff1 + h - 12]
        new_val = fc_vals[h] + extended_diff1[n_diff1 + h - 12]
        extended_diff1.append(new_val)

    forecast_diff1 = np.array(extended_diff1[n_diff1:])

    # Undo Δ1: log_raw[t+1] = log_raw[t] + diff1[t]
    log_forecast = np.empty(n_fc)
    prev_log = log_raw[-1]
    for h in range(n_fc):
        prev_log = prev_log + forecast_diff1[h]
        log_forecast[h] = prev_log

    return np.exp(log_forecast)


fc_point = invert_forecast(forecast_means)
fc_hi = invert_forecast(fc_upper)
fc_lo = invert_forecast(fc_lower)

# ── 7. Plot ───────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(range(n_total), raw, color="black", label="Historical")

fc_x = range(n_total, n_total + n_fc)
ax.plot(fc_x, fc_point, color="steelblue", linewidth=2, label="Point forecast")
ax.fill_between(fc_x, fc_lo, fc_hi, color="steelblue", alpha=0.25, label="90% interval")

ax.set_xlabel("Month index")
ax.set_ylabel("Total passengers")
ax.set_title("Airline Passengers — 12-month out-of-sample forecast (Kalman MLE)")
ax.legend()
ax.grid(True, alpha=0.3)

fig.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), "airline_forecast.png")
fig.savefig(out_path, dpi=150)
print(f"Plot saved to {out_path}")
plt.close(fig)
