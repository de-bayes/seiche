"""Decisive test: does adding the streams tighten the BAND while holding coverage?
Full 5-quantile backtest, base vs +phys vs +both, scored on the proper
probabilistic metric (pinball loss) plus coverage (calibration) and 90%-band
width (sharpness) and median tail MAE, at long lead on stream-covered rows.

PRE-REGISTERED criterion (stated before the result): promote a stream set if, on
covered long-lead rows, it lowers pinball loss AND median tail MAE while keeping
cover90 within [0.86, 0.94]. Prefer +phys over +both unless +both is materially
better on pinball."""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

import featuresq
import streams

TAU, CF = 8.0, 1.8
Q = [0.05, 0.25, 0.5, 0.75, 0.95]
SAT = ["sat_basin", "sat_grad_near", "sat_grad_far", "sat_basin_d3", "sat_err"]
PHYS = ["lmhofs_now", "lmhofs_fut", "lmhofs_delta", "lmhofs_err"]
buoy = pd.read_csv("data/buoy.csv", index_col=0, parse_dates=True)
wx = pd.read_csv("data/weather.csv", index_col=0, parse_dates=True)
X, y, t, h = featuresq.stack(buoy, wx)
yv = y.to_numpy()
extra = streams.build_blocks(buoy); extra.index = X.index
VARIANTS = {"base": X, "+phys": pd.concat([X, extra[PHYS]], axis=1),
            "+both": pd.concat([X, extra[SAT + PHYS]], axis=1)}

folds = []
for year in [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]:
    iy = t[t.year == year]
    if len(iy):
        folds.append((str(year), iy.max() - pd.Timedelta(days=45), iy.max()))
folds.append(("2026", t.max() - pd.Timedelta(days=35), t.max()))


def quant_fit(Xset, tr, te, hte, tte):
    """5 anchored quantile preds, deg F, shape (5, n)."""
    raw = {}
    for q in Q:
        m = HistGradientBoostingRegressor(loss="quantile", quantile=q, max_iter=250,
                                          learning_rate=0.08, random_state=11)
        m.fit(Xset[tr], y[tr]); raw[q] = m.predict(Xset[te])
    persist = Xset[te]["WTMP"].to_numpy(); m1 = hte == 1
    dbt = dict(zip(tte[m1].values, persist[m1] - raw[0.5][m1]))
    delta = np.array([dbt.get(v, 0.0) for v in tte.values]) * np.exp(-(hte - 1) / TAU)
    return np.sort(np.vstack([raw[q] + delta for q in Q]), axis=0) * CF   # monotone, deg F


P = {k: [] for k in VARIANTS}
meta = {"h": [], "have": [], "y": []}
for name, wstart, wend in folds:
    tr = t <= (wstart - pd.Timedelta(days=8))
    te = (t >= wstart) & (t <= wend)
    if te.sum() < 1000:
        continue
    hte, tte = np.asarray(h[te]), t[te]
    for k, Xset in VARIANTS.items():
        P[k].append(quant_fit(Xset, tr, te, hte, tte))
    meta["h"].append(hte); meta["y"].append(yv[te] * CF)
    meta["have"].append(np.isfinite(extra.loc[np.asarray(te), "lmhofs_err"].to_numpy()))
    print(f"fold {name} done", flush=True)

P = {k: np.hstack(v) for k, v in P.items()}        # (5, N)
H = np.concatenate(meta["h"]); Y = np.concatenate(meta["y"]); have = np.concatenate(meta["have"])
longm = np.isin(H, [72, 120, 168]) & have
base_abs = np.abs(P["base"][2] - Y)
cut = np.quantile(base_abs[longm], 0.90)
tail = longm & (base_abs >= cut)


def pinball(pred, m):
    tot = 0.0
    for i, q in enumerate(Q):
        d = Y[m] - pred[i][m]
        tot += np.mean(np.maximum(q * d, (q - 1) * d))
    return tot / len(Q)


print(f"\n{have.sum()} covered rows · tail = base worst-10% long-lead (|err|>={cut:.2f}F)\n")
print(f"{'variant':<8} {'pinball':>8} {'cover90':>8} {'band90 w':>9} {'med MAE':>8} {'tail MAE':>9}")
for k, pred in P.items():
    pb = pinball(pred, longm)
    c90 = np.mean((Y[longm] >= pred[0][longm]) & (Y[longm] <= pred[4][longm]))
    w90 = np.mean(pred[4][longm] - pred[0][longm])
    med = np.mean(np.abs(pred[2][longm] - Y[longm]))
    tmae = np.mean(np.abs(pred[2][tail] - Y[tail]))
    print(f"{k:<8} {pb:>8.3f} {c90:>8.3f} {w90:>9.2f} {med:>8.3f} {tmae:>9.3f}")
print("\n(lower pinball/band/MAE better; cover90 should stay ~0.90)")
