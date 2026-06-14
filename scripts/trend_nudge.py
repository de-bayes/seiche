"""Test Ryan's idea: a Kalman-style trend nudge. The model regresses to the mean
during upwelling (too warm in the tail). So lean the median cooler in proportion
to the recent cooling rate: posterior = prior + beta * recent_trend * decay(h),
where recent_trend = the diurnal-filtered 24h change (wtmp_d24, already a feature
but under-used). beta is the Kalman gain on the live trend; decay lets the nudge
accumulate over a few days (upwelling persists) and saturate.

Reproduces the 9-season backtest, collects the anchored-median error and the
live trend per forecast, then sweeps (beta, tau) and reports whether it shrinks
the worst-decile tail at long lead WITHOUT hurting the calm cases or short leads.
Improving the median directly tightens the conformal band (smaller residuals)."""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

import featuresq

TAU, CF = 8.0, 1.8
buoy = pd.read_csv("data/buoy.csv", index_col=0, parse_dates=True)
wx = pd.read_csv("data/weather.csv", index_col=0, parse_dates=True)
X, y, t, h = featuresq.stack(buoy, wx)
yv = y.to_numpy()

folds = []
for year in [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]:
    iy = t[t.year == year]
    if len(iy):
        folds.append((str(year), iy.max() - pd.Timedelta(days=45), iy.max()))
folds.append(("2026", t.max() - pd.Timedelta(days=35), t.max()))

cols = {k: [] for k in ("err", "trend", "h", "fold", "base")}
for name, wstart, wend in folds:
    tr = t <= (wstart - pd.Timedelta(days=8))
    te = (t >= wstart) & (t <= wend)
    if te.sum() < 1000:
        continue
    m = HistGradientBoostingRegressor(loss="quantile", quantile=0.5, max_iter=250,
                                      learning_rate=0.08, random_state=11)
    m.fit(X[tr], y[tr])
    Xte, yte, hte, tte = X[te], yv[te], np.asarray(h[te]), t[te]
    p50 = m.predict(Xte); persist = Xte["WTMP"].to_numpy()
    m1 = hte == 1
    dbt = dict(zip(tte[m1].values, persist[m1] - p50[m1]))
    delta = np.array([dbt.get(v, 0.0) for v in tte.values])
    p50 = p50 + delta * np.exp(-(hte - 1) / TAU)
    cols["err"].append((p50 - yte) * CF)                 # signed deg F (+ = too warm)
    cols["trend"].append(Xte["wtmp_d24"].to_numpy())     # recent 24h change, deg C
    cols["h"].append(hte); cols["fold"].append(np.array([name] * len(yte)))
    cols["base"].append(np.asarray(tte))
    print(f"fold {name} done", flush=True)

err = np.concatenate(cols["err"]); trend = np.concatenate(cols["trend"])
H = np.concatenate(cols["h"]); FOLD = np.concatenate(cols["fold"]); BASE = np.concatenate(cols["base"])
good = np.isfinite(trend)
err, trend, H, FOLD, BASE = err[good], trend[good], H[good], FOLD[good], BASE[good]


def nudge(beta, tau):
    return err + beta * trend * (1 - np.exp(-H / tau)) * CF   # new signed error, deg F


def mae(e, mask=None):
    e = e if mask is None else e[mask]
    return float(np.mean(np.abs(e)))


# tail defined on the RAW long-lead error so we compare like-for-like
longm = np.isin(H, [72, 120, 168])
cut = np.quantile(np.abs(err[longm]), 0.90)
tailm = longm & (np.abs(err) >= cut)

print(f"\n{len(err)} forecasts · long-lead worst-10% tail |err|>= {cut:.2f}F\n")
print(f"{'beta':>5} {'tau':>4} | {'MAE all':>8} {'MAE+24':>7} {'MAE+168':>8} | "
      f"{'tail MAE':>8} {'tail bias':>9} | {'2020-09-20 +168h':>16}")
ev = (FOLD == "2020") & (H == 168) & (pd.to_datetime(BASE).date == pd.Timestamp("2020-09-20").date())
for beta, tau in [(0.0, 48), (0.25, 72), (0.5, 72), (0.75, 72), (1.0, 72), (0.5, 120), (1.0, 120), (1.5, 96)]:
    e = nudge(beta, tau)
    h24, h168 = H == 24, H == 168
    line = (f"{beta:>5.2f} {tau:>4} | {mae(e):>8.3f} {mae(e, h24):>7.3f} {mae(e, h168):>8.3f} | "
            f"{mae(e, tailm):>8.3f} {np.mean(e[tailm]):>+9.2f} | {np.mean(np.abs(e[ev])):>16.2f}")
    print(line)

print("\n=== the REACTIVE case: forecasts ISSUED while the water is already cooling fast ===")
print("(this is the live-update value: once a drop starts, do we react or regress back up?)")
for thr in [-0.7, -1.5]:
    cm = trend < thr
    print(f"\n  recent 24h change < {thr}°C  ({cm.sum()} forecasts, the active-cooling cohort):")
    print(f"    {'lead':>5} {'n':>5} | {'raw MAE':>8} {'raw bias':>9} | {'best beta':>9} {'MAE':>7} {'bias':>7}")
    for lead in [12, 24, 48, 72]:
        s = cm & (H == lead)
        if s.sum() < 30:
            continue
        raw, rbias = mae(err, s), float(np.mean(err[s]))
        best = None
        for beta in [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]:
            e = nudge(beta, 48); mm = mae(e, s)
            if best is None or mm < best[1]:
                best = (beta, mm, float(np.mean(e[s])))
        flag = "  <-- helps" if best[1] < raw - 0.02 else ""
        print(f"    {lead:>5} {s.sum():>5} | {raw:>8.3f} {rbias:>+9.2f} | {best[0]:>9.2f} {best[1]:>7.3f} {best[2]:>+7.2f}{flag}")
