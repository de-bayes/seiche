"""Can physics anticipation features shrink the upwelling tail? The worst day-7
misses are upwellings the model can't see coming a week out. An upwelling drop is
stratification x sustained upwelling-favorable wind, so give the model those
explicitly (the data shows the worst crashes followed v_m24 > 0, so upwelling-
favorable = positive alongshore v):

  strat        = WTMP - 4C            surface-to-deep available drop (stratification)
  up_now       = max(v_m24,0)*wspd_m24      recent sustained upwelling wind-stress
  up_fut       = max(fut_v,0)*fut_wspd      FORECAST upwelling wind over the window
  pot_now/fut  = strat * up_now / up_fut    THE interaction: ripe + forcing = drop
  ekman        = max(v,0)*WSPD              instantaneous upwelling stress

Backtests a median HGB with and without these, comparing overall MAE, the long-
lead worst-decile tail, the warm bias in the tail, and the 2020-09-20 event. Also
reports corr(|error|, pot_fut) to judge it as a band-conditioning signal."""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

import featuresq

TAU, CF, DEEP = 8.0, 1.8, 4.0
buoy = pd.read_csv("data/buoy.csv", index_col=0, parse_dates=True)
wx = pd.read_csv("data/weather.csv", index_col=0, parse_dates=True)
X, y, t, h = featuresq.stack(buoy, wx)
yv = y.to_numpy()

# physics features, all from existing columns
strat = (X["WTMP"] - DEEP).clip(lower=0)
A = X.copy()
A["strat"] = strat
A["up_now"] = np.maximum(X["v_m24"], 0) * X["wspd_m24"]
A["up_fut"] = np.maximum(X["fut_v"], 0) * X["fut_wspd"]
A["pot_now"] = strat * A["up_now"]
A["pot_fut"] = strat * A["up_fut"]
A["ekman"] = np.maximum(X["v"], 0) * X["WSPD"]
NEW = ["strat", "up_now", "up_fut", "pot_now", "pot_fut", "ekman"]

folds = []
for year in [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]:
    iy = t[t.year == year]
    if len(iy):
        folds.append((str(year), iy.max() - pd.Timedelta(days=45), iy.max()))
folds.append(("2026", t.max() - pd.Timedelta(days=35), t.max()))


def fit_predict(Xset, tr, te, hte, tte):
    m = HistGradientBoostingRegressor(loss="quantile", quantile=0.5, max_iter=250,
                                      learning_rate=0.08, random_state=11)
    m.fit(Xset[tr], y[tr])
    p = m.predict(Xset[te]); persist = Xset[te]["WTMP"].to_numpy()
    m1 = hte == 1
    dbt = dict(zip(tte[m1].values, persist[m1] - p[m1]))
    delta = np.array([dbt.get(v, 0.0) for v in tte.values])
    return p + delta * np.exp(-(hte - 1) / TAU)


rows = {k: [] for k in ("eb", "ea", "h", "fold", "base", "potfut")}
for name, wstart, wend in folds:
    tr = t <= (wstart - pd.Timedelta(days=8))
    te = (t >= wstart) & (t <= wend)
    if te.sum() < 1000:
        continue
    hte, tte, yte = np.asarray(h[te]), t[te], yv[te]
    pb = fit_predict(X, tr, te, hte, tte)
    pa = fit_predict(A, tr, te, hte, tte)
    rows["eb"].append((pb - yte) * CF); rows["ea"].append((pa - yte) * CF)
    rows["h"].append(hte); rows["fold"].append(np.array([name] * len(yte)))
    rows["base"].append(np.asarray(tte)); rows["potfut"].append(A[te]["pot_fut"].to_numpy())
    print(f"fold {name} done", flush=True)

eb = np.concatenate(rows["eb"]); ea = np.concatenate(rows["ea"])
H = np.concatenate(rows["h"]); FOLD = np.concatenate(rows["fold"])
BASE = np.concatenate(rows["base"]); POT = np.concatenate(rows["potfut"])
mae = lambda e, m=None: float(np.mean(np.abs(e if m is None else e[m])))

longm = np.isin(H, [72, 120, 168])
cut = np.quantile(np.abs(eb[longm]), 0.90)
tail = longm & (np.abs(eb) >= cut)
ev = (FOLD == "2020") & (H == 168) & (pd.to_datetime(BASE).date == pd.Timestamp("2020-09-20").date())

print(f"\n{len(eb)} forecasts · long-lead worst-10% tail |err|>= {cut:.2f}F\n")
print(f"{'metric':<26} {'base':>9} {'+physics':>9} {'change':>8}")
def line(lbl, fb, fa):
    print(f"{lbl:<26} {fb:>9.3f} {fa:>9.3f} {fa-fb:>+8.3f}")
line("MAE all", mae(eb), mae(ea))
line("MAE +24h", mae(eb, H == 24), mae(ea, H == 24))
line("MAE +168h", mae(eb, H == 168), mae(ea, H == 168))
line("tail MAE (long, worst10%)", mae(eb, tail), mae(ea, tail))
line("tail warm-bias", float(np.mean(eb[tail])), float(np.mean(ea[tail])))
line("2020-09-20 +168h |err|", mae(eb, ev), mae(ea, ev))

g = np.isfinite(POT) & longm
print(f"\nas a BAND-conditioning signal: corr(|base err|, pot_fut) on long-lead = "
      f"{np.corrcoef(np.abs(eb[g]), POT[g])[0,1]:+.3f}")
hi = g & (POT >= np.nanquantile(POT[g], 0.90))
print(f"  forecasts in the top-10% upwelling-potential: |err| {mae(eb,hi):.2f}F vs "
      f"{mae(eb, g & ~hi):.2f}F elsewhere  ({(np.mean(np.abs(eb[hi])>=cut))*100:.0f}% land in the tail "
      f"vs {(np.mean(np.abs(eb[g & ~hi])>=cut))*100:.0f}%)")
