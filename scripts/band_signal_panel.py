"""Which signal should the conformal band condition on?

Volatility (std of the WTMP lag ladder) correlates only +0.22 with |error| and
does NOT rescue the 2019 fold, because 2019 was a systematic regime miss, not a
choppy-water year. This panel scores candidate conditioners on the same nine
backtest folds (median HGB only, anchored) by:
  - corr(signal, |error|) pooled  -- does it track error at all?
  - per-fold mean signal           -- is it ELEVATED in the bad folds (2019/2020/2026)?

Candidates:
  vol            std of the last-24h WTMP lag ladder (current production-candidate)
  absd24         |24h WTMP change|
  airwater       |air - water| (forcing magnitude)
  presd3         |3h pressure tendency| (synoptic activity)
  wspd           wind speed (mixing)
  modpersist     |median - persistence| (the model's own disagreement w/ persistence)
  recenterr      ADAPTIVE: trailing mean of the realized +24h |error| known by t0
                 (the ACI-style signal; the standard fix for distribution shift)

Prints a ranked table. Nothing is written."""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

import featuresq

TAU = 8.0
CF = 1.8
BAD = {"2019", "2020", "2026"}   # the under-covered folds we must catch

buoy = pd.read_csv("data/buoy.csv", index_col=0, parse_dates=True)
wx = pd.read_csv("data/weather.csv", index_col=0, parse_dates=True)
X, y, t, h = featuresq.stack(buoy, wx)
yv = y.to_numpy()

folds = []
for year in [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]:
    in_year = t[(t.year == year)]
    if len(in_year):
        wend = in_year.max()
        folds.append((str(year), wend - pd.Timedelta(days=45), wend))
folds.append(("2026", t.max() - pd.Timedelta(days=35), t.max()))

SIGNALS = ["vol", "absd24", "airwater", "presd3", "wspd", "modpersist", "recenterr"]
pooled = {s: [] for s in SIGNALS}
pooled_abserr = []
by_fold = {}

for name, wstart, wend in folds:
    tr = t <= (wstart - pd.Timedelta(days=8))
    te = (t >= wstart) & (t <= wend)
    if te.sum() < 1000:
        continue
    m = HistGradientBoostingRegressor(loss="quantile", quantile=0.5, max_iter=250,
                                      learning_rate=0.08, random_state=11)
    m.fit(X[tr], y[tr])
    Xte, yte, hte, tte = X[te], yv[te], np.asarray(h[te]), t[te]
    p50 = m.predict(Xte)
    persist = Xte["WTMP"].to_numpy()
    # anchor blend (same as production)
    m1 = hte == 1
    delta_by_t = dict(zip(tte[m1].values, persist[m1] - p50[m1]))
    delta = np.array([delta_by_t.get(v, 0.0) for v in tte.values])
    p50 = p50 + delta * np.exp(-(hte - 1) / TAU)
    abserr = np.abs(p50 - yte)

    sig = {
        "vol": featuresq.regime_signal(Xte),
        "absd24": np.abs(Xte["wtmp_d24"].to_numpy()),
        "airwater": np.abs(Xte["atmp_minus_wtmp"].to_numpy()),
        "presd3": np.abs(Xte["pres_d3"].to_numpy()),
        "wspd": Xte["WSPD"].to_numpy(),
        "modpersist": np.abs(p50 - persist),
    }

    # recenterr (ACI-style): for each base time t0, the trailing-48h mean of the
    # realized +24h error, counting only forecasts that have resolved by t0 (the
    # +24h call made at b resolves at b+24h, so it is known at t0 iff b+24h <= t0).
    base24 = tte[hte == 24]
    err24 = abserr[hte == 24]
    s24 = pd.Series(err24.values if hasattr(err24, "values") else err24,
                    index=base24).sort_index()
    s24 = s24[~s24.index.duplicated(keep="last")]
    resolved = s24.copy()
    resolved.index = resolved.index + pd.Timedelta(hours=24)   # time the error becomes known
    # trailing 48h mean of resolved errors, as of each base time in the fold
    rolled = resolved.rolling("48h").mean()
    recent = rolled.reindex(tte, method="ffill").to_numpy()
    recent = np.where(np.isfinite(recent), recent, np.nanmedian(rolled.to_numpy()))
    sig["recenterr"] = recent

    by_fold[name] = {s: float(np.nanmean(sig[s])) for s in SIGNALS}
    by_fold[name]["_mae_F"] = round(float(np.mean(abserr)) * CF, 3)
    for s in SIGNALS:
        pooled[s].append(sig[s])
    pooled_abserr.append(abserr)
    print(f"fold {name} done (mae {by_fold[name]['_mae_F']}F)", flush=True)

ae = np.concatenate(pooled_abserr)
print("\n=== corr(signal, |error|) pooled, and z-score of BAD-fold mean vs good folds ===")
rows = []
for s in SIGNALS:
    v = np.concatenate(pooled[s])
    g = np.isfinite(v) & np.isfinite(ae)
    corr = float(np.corrcoef(v[g], ae[g])[0, 1])
    bad = np.mean([by_fold[f][s] for f in by_fold if f in BAD])
    good = np.mean([by_fold[f][s] for f in by_fold if f not in BAD])
    sd = np.std([by_fold[f][s] for f in by_fold])
    sep = (bad - good) / sd if sd else 0.0   # how much higher the signal is in bad folds
    rows.append((s, corr, sep))
rows.sort(key=lambda r: -abs(r[1]))
print(f"  {'signal':>11} {'corr|err|':>10} {'bad-fold separation (sd)':>26}")
for s, corr, sep in rows:
    print(f"  {s:>11} {corr:>+10.3f} {sep:>+26.2f}")

print("\n=== per-fold mean signal (watch 2019/2020/2026) ===")
hdr = "  " + f"{'fold':>6} {'mae_F':>6} " + " ".join(f"{s:>10}" for s in SIGNALS)
print(hdr)
for f in by_fold:
    line = f"  {f:>6} {by_fold[f]['_mae_F']:>6} " + " ".join(f"{by_fold[f][s]:>10.3f}" for s in SIGNALS)
    print(line)
