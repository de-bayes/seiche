"""Which stream carries the tail win -- satellite, lake-physics, or both? Operations
favor satellite (100% coverage, reliable fetch) over LMHOFS (~50%, flaky). Backtest
base vs +sat vs +phys vs +both (median HGB), all judged on the SAME stream-covered
long-lead rows so it is apples-to-apples."""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

import featuresq
import streams

TAU, CF = 8.0, 1.8
SAT = ["sat_basin", "sat_grad_near", "sat_grad_far", "sat_basin_d3", "sat_err"]
PHYS = ["lmhofs_now", "lmhofs_fut", "lmhofs_delta", "lmhofs_err"]
buoy = pd.read_csv("data/buoy.csv", index_col=0, parse_dates=True)
wx = pd.read_csv("data/weather.csv", index_col=0, parse_dates=True)
X, y, t, h = featuresq.stack(buoy, wx)
yv = y.to_numpy()
extra = streams.build_blocks(buoy); extra.index = X.index
VARIANTS = {"base": X, "+sat": pd.concat([X, extra[SAT]], axis=1),
            "+phys": pd.concat([X, extra[PHYS]], axis=1),
            "+both": pd.concat([X, extra[SAT + PHYS]], axis=1)}

folds = []
for year in [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]:
    iy = t[t.year == year]
    if len(iy):
        folds.append((str(year), iy.max() - pd.Timedelta(days=45), iy.max()))
folds.append(("2026", t.max() - pd.Timedelta(days=35), t.max()))


def fp(Xset, tr, te, hte, tte):
    m = HistGradientBoostingRegressor(loss="quantile", quantile=0.5, max_iter=250,
                                      learning_rate=0.08, random_state=11)
    m.fit(Xset[tr], y[tr])
    p = m.predict(Xset[te]); persist = Xset[te]["WTMP"].to_numpy()
    m1 = hte == 1
    dbt = dict(zip(tte[m1].values, persist[m1] - p[m1]))
    delta = np.array([dbt.get(v, 0.0) for v in tte.values])
    return p + delta * np.exp(-(hte - 1) / TAU)


errs = {k: [] for k in VARIANTS}; meta = {"h": [], "have": [], "fold": []}
for name, wstart, wend in folds:
    tr = t <= (wstart - pd.Timedelta(days=8))
    te = (t >= wstart) & (t <= wend)
    if te.sum() < 1000:
        continue
    hte, tte, yte = np.asarray(h[te]), t[te], yv[te]
    for k, Xset in VARIANTS.items():
        errs[k].append((fp(Xset, tr, te, hte, tte) - yte) * CF)
    meta["h"].append(hte); meta["fold"].append(np.array([name] * len(yte)))
    meta["have"].append(np.isfinite(extra.loc[np.asarray(te), "lmhofs_err"].to_numpy()))
    print(f"fold {name} done", flush=True)

E = {k: np.concatenate(v) for k, v in errs.items()}
H = np.concatenate(meta["h"]); have = np.concatenate(meta["have"])
mae = lambda e, m: float(np.mean(np.abs(e[m])))
longm = np.isin(H, [72, 120, 168]) & have
cut = np.quantile(np.abs(E["base"][longm]), 0.90)
tail = longm & (np.abs(E["base"]) >= cut)

print(f"\njudged on {have.sum()} stream-covered rows · tail = base worst-10% (|err|>={cut:.2f}F)\n")
print(f"{'variant':<8} {'MAE all':>9} {'MAE+168':>9} {'tail MAE':>9} {'tail bias':>10}")
for k in VARIANTS:
    e = E[k]
    print(f"{k:<8} {mae(e, have):>9.3f} {mae(e, (H==168)&have):>9.3f} {mae(e, tail):>9.3f} "
          f"{float(np.mean(e[tail])):>+10.2f}")
