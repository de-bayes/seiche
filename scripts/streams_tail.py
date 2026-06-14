"""The one untested lever: do the SUBSURFACE/BASIN streams (satellite basin SST +
NOAA 3D lake-physics model) shrink the upwelling tail? They were rejected on
MEDIAN accuracy (validate_streams2) but never scored on the tail, and they are
the only inputs that see below the surface / across the basin -- exactly what
the surface buoy and cheap physics proxies couldn't (both failed: trend_nudge,
upwelling_features).

Backtests a median HGB with and without the stream features and reports the
long-lead worst-decile tail, the 2020-09-20 event, and each stream feature's
correlation with the error (its value as a band-conditioning early-warning)."""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

import featuresq
import streams

TAU, CF = 8.0, 1.8
buoy = pd.read_csv("data/buoy.csv", index_col=0, parse_dates=True)
wx = pd.read_csv("data/weather.csv", index_col=0, parse_dates=True)
X, y, t, h = featuresq.stack(buoy, wx)
yv = y.to_numpy()
extra = streams.build_blocks(buoy); extra.index = X.index
SCOLS = ["sat_basin", "sat_grad_near", "sat_grad_far", "sat_basin_d3", "sat_err",
         "lmhofs_now", "lmhofs_fut", "lmhofs_delta", "lmhofs_err"]
A = pd.concat([X, extra[SCOLS]], axis=1)

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


R = {k: [] for k in ("eb", "ea", "h", "fold", "base")}
SF = {c: [] for c in SCOLS}
for name, wstart, wend in folds:
    tr = t <= (wstart - pd.Timedelta(days=8))
    te = (t >= wstart) & (t <= wend)
    if te.sum() < 1000:
        continue
    hte, tte, yte = np.asarray(h[te]), t[te], yv[te]
    pb = fit_predict(X, tr, te, hte, tte)
    pa = fit_predict(A, tr, te, hte, tte)
    R["eb"].append((pb - yte) * CF); R["ea"].append((pa - yte) * CF)
    R["h"].append(hte); R["fold"].append(np.array([name] * len(yte))); R["base"].append(np.asarray(tte))
    for c in SCOLS:
        SF[c].append(A[te][c].to_numpy())
    print(f"fold {name} done", flush=True)

eb = np.concatenate(R["eb"]); ea = np.concatenate(R["ea"])
H = np.concatenate(R["h"]); FOLD = np.concatenate(R["fold"]); BASE = np.concatenate(R["base"])
sf = {c: np.concatenate(SF[c]) for c in SCOLS}
mae = lambda e, m=None: float(np.mean(np.abs(e if m is None else e[m])))

# restrict the comparison to rows where the streams actually exist (LMHOFS ~2019+)
have = np.isfinite(sf["lmhofs_err"])
longm = np.isin(H, [72, 120, 168]) & have
cut = np.quantile(np.abs(eb[longm]), 0.90)
tail = longm & (np.abs(eb) >= cut)
ev = (FOLD == "2020") & (H == 168) & (pd.to_datetime(BASE).date == pd.Timestamp("2020-09-20").date())

print(f"\n{have.sum()} forecasts with stream coverage · long-lead tail |err|>= {cut:.2f}F\n")
print(f"{'metric (stream-covered rows)':<30} {'base':>9} {'+streams':>9} {'change':>8}")
def line(lbl, fb, fa):
    print(f"{lbl:<30} {fb:>9.3f} {fa:>9.3f} {fa-fb:>+8.3f}")
line("MAE all", mae(eb, have), mae(ea, have))
line("MAE +168h", mae(eb, (H == 168) & have), mae(ea, (H == 168) & have))
line("tail MAE (long, worst10%)", mae(eb, tail), mae(ea, tail))
line("tail warm-bias", float(np.mean(eb[tail])), float(np.mean(ea[tail])))
line("2020-09-20 +168h |err|", mae(eb, ev), mae(ea, ev))

print("\nband-conditioning value: corr(|base err|, stream feature) on covered long-lead:")
g = longm
for c in SCOLS:
    v = sf[c][g]; e = np.abs(eb[g]); ok = np.isfinite(v) & np.isfinite(e)
    if ok.sum() > 100:
        print(f"  {c:16} {np.corrcoef(v[ok], e[ok])[0,1]:+.3f}")
