"""Head-to-head: HistGradientBoosting quantiles (production) vs L1-regularized
linear quantile regression (QuantileRegressor, i.e. lasso's pinball-loss cousin).

Replicates train_q.py's exact honest split (last 35 days TEST, 168+24 h gap so
no stacked row straddles it) and the same anchor-blend post-processing, then
scores both families on the metrics that matter for a banded forecaster:

  - median MAE (deg F)        how tight the central line is
  - pinball loss (deg F)      mean over the 5 quantiles -- the proper band score
  - cover50 / cover90         empirical coverage of the 50% and 90% bands
  - band90 width (deg F)      sharpness of the 90% band

Nothing is written; this only prints a comparison so train_q.py stays untouched.
The linear model trains on a subsample of TRAIN (LP cost scales worse than
trees); the TEST set is always full. If linear wins even subsampled+untuned,
that is a strong signal to tune alpha and consider a horizon split.
"""

import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import QuantileRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import featuresq

CF = 1.8                 # deg C error -> deg F error (scale only), matches train_q
TEST_DAYS = 35
MAX_ITER = 500
TAU = 8.0                # anchor-blend decay scale, matches train_q
LIN_TRAIN_SUBSAMPLE = 25_000
LIN_ALPHA = 1e-3         # L1 strength on standardized features; untuned on purpose
SEED = 11

buoy = pd.read_csv("data/buoy.csv", index_col=0, parse_dates=True)
wx = pd.read_csv("data/weather.csv", index_col=0, parse_dates=True)

X, y, t, h = featuresq.stack(buoy, wx)
yv = y.to_numpy()
t_end = t.max()
test_start = t_end - pd.Timedelta(days=TEST_DAYS)
train_end = test_start - pd.Timedelta(hours=168 + 24)
tr = np.asarray(t <= train_end)
te = np.asarray(t >= test_start)
print(f"stacked rows: {len(X)} - train {tr.sum()} - test {te.sum()} - features {X.shape[1]}")

Xte, yte, hte, tte = X[te], yv[te], h[te], t[te]
persist = Xte["WTMP"].to_numpy()

# subsample train rows for the linear LP (test stays full)
tr_idx = np.flatnonzero(tr)
rng = np.random.default_rng(SEED)
if len(tr_idx) > LIN_TRAIN_SUBSAMPLE:
    lin_idx = np.sort(rng.choice(tr_idx, size=LIN_TRAIN_SUBSAMPLE, replace=False))
else:
    lin_idx = tr_idx
print(f"linear trains on {len(lin_idx)} subsampled rows (alpha={LIN_ALPHA}); HGB on full {tr.sum()}")


def hgb(q):
    return HistGradientBoostingRegressor(loss="quantile", quantile=q, max_iter=MAX_ITER,
                                         learning_rate=0.07, random_state=SEED)


def lin(q):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         QuantileRegressor(quantile=q, alpha=LIN_ALPHA, solver="highs"))


def fit_predict(make, Xtr, ytr):
    """Fit one regressor per quantile, return dict q -> test preds (raw deg C)."""
    out = {}
    for q in featuresq.QUANTILES:
        m = make(q)
        m.fit(Xtr, ytr)
        out[q] = m.predict(Xte)
        print(f"  fit q{q:.2f}")
    return out


def anchor(pred):
    """train_q's anchor blend: shift trajectory by (obs_now - +1h median), decaying."""
    m1 = hte == 1
    delta_by_t = dict(zip(tte[m1].values, (persist[m1] - pred[0.5][m1])))
    delta = np.array([delta_by_t.get(v, 0.0) for v in tte.values])
    decay = np.exp(-(hte - 1) / TAU)
    return {q: pred[q] + delta * decay for q in featuresq.QUANTILES}


def monotonic(pred):
    """Sort the 5 quantile preds row-wise to kill any crossing. Returns (pred, raw_cross_rate)."""
    stack = np.column_stack([pred[q] for q in featuresq.QUANTILES])
    cross = float(np.mean(np.any(np.diff(stack, axis=1) < 0, axis=1)))
    stack.sort(axis=1)
    return {q: stack[:, i] for i, q in enumerate(featuresq.QUANTILES)}, cross


def pinball_F(pred, mask):
    """Mean pinball loss across the 5 quantiles for the masked rows, in deg F."""
    tot = 0.0
    for q in featuresq.QUANTILES:
        d = yte[mask] - pred[q][mask]
        tot += np.mean(np.maximum(q * d, (q - 1) * d))
    return float(tot / len(featuresq.QUANTILES)) * CF


def evaluate(name, make, Xtr, ytr):
    print(f"\nfitting {name} ...")
    pred = anchor(fit_predict(make, Xtr, ytr))
    pred, cross = monotonic(pred)
    rows = {}
    for hz in featuresq.HSET:
        m = hte == hz
        if m.sum() < 50:
            continue
        err = (pred[0.5][m] - yte[m]) * CF
        rows[hz] = {
            "mae": float(np.mean(np.abs(err))),
            "pinball": pinball_F(pred, m),
            "cover50": float(np.mean((yte[m] >= pred[0.25][m]) & (yte[m] <= pred[0.75][m]))),
            "cover90": float(np.mean((yte[m] >= pred[0.05][m]) & (yte[m] <= pred[0.95][m]))),
            "band90": float(np.mean(pred[0.95][m] - pred[0.05][m])) * CF,
            "n": int(m.sum()),
        }
    return rows, cross


hgb_rows, _ = evaluate("HGB (production)", hgb, X[tr], y[tr])
lin_rows, lin_cross = evaluate("LINEAR (L1 quantile)", lin, X.iloc[lin_idx], y.iloc[lin_idx])

print(f"\nlinear raw quantile-crossing rate before monotone sort: {lin_cross:.3%}\n")
print(f"{'h':>4} | {'MAE F  hgb / lin':>18} | {'pinball F  hgb / lin':>22} | "
      f"{'cover90 hgb/lin':>16} | {'band90 F hgb/lin':>17}")
print("-" * 92)
wmae = wpin = n = 0.0
for hz in featuresq.HSET:
    if hz not in hgb_rows or hz not in lin_rows:
        continue
    a, b = hgb_rows[hz], lin_rows[hz]
    w = a["n"]
    wmae += (b["mae"] - a["mae"]) * w
    wpin += (b["pinball"] - a["pinball"]) * w
    n += w
    print(f"{hz:>4} | {a['mae']:>7.3f} / {b['mae']:<7.3f} | "
          f"{a['pinball']:>9.3f} / {b['pinball']:<9.3f} | "
          f"{a['cover90']:>6.2f} / {b['cover90']:<6.2f} | "
          f"{a['band90']:>7.2f} / {b['band90']:<7.2f}")
print("-" * 92)
print(f"n-weighted MAE delta (lin - hgb): {wmae / n:+.4f} F   "
      f"(negative = linear tighter)")
print(f"n-weighted pinball delta (lin - hgb): {wpin / n:+.4f} F   "
      f"(negative = linear better-calibrated bands)")
