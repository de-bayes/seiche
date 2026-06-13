"""Compare band-calibration methods on the nine backtest folds by what actually
matters: how uniformly the 90% band covers across folds (esp. the under-covered
regime years), and how tight it is.

Methods (all split-conformal on the anchored median's error, finite-sample
corrected, made monotone in lead like publish):
  static       one width per horizon (current production)
  vol_norm     normalized conformal, difficulty = lag-ladder volatility
  recent_norm  normalized conformal, difficulty = trailing realized +24h error (ACI-style)
  mp_norm      normalized conformal, difficulty = |median - persistence|
  blend_norm   difficulty = geometric mean of recent-error and |median-persistence|

Normalized conformal: scale(t)=clip(sigma(t)/median(sigma),[0.5,2.5]); score=err/scale;
per-horizon conformal quantiles of score; band = pred +/- q(h)*scale(t). Marginal
coverage holds by construction; a good scale buys CONDITIONAL coverage too.

Prints a comparison. Nothing is written."""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

import featuresq

TAU = 8.0
CF = 1.8
CLIP = (0.5, 2.5)
HSET = featuresq.HSET

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

rows = []   # per test row across all folds: dict of arrays
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
    m1 = hte == 1
    dbt = dict(zip(tte[m1].values, persist[m1] - p50[m1]))
    delta = np.array([dbt.get(v, 0.0) for v in tte.values])
    p50 = p50 + delta * np.exp(-(hte - 1) / TAU)
    err = p50 - yte

    # recenterr: trailing-48h mean realized +24h error known by base time t0
    b24, e24 = tte[hte == 24], np.abs(err[hte == 24])
    s24 = pd.Series(e24, index=b24).sort_index()
    s24 = s24[~s24.index.duplicated(keep="last")]
    res = s24.copy(); res.index = res.index + pd.Timedelta(hours=24)
    rolled = res.rolling("48h").mean()
    recent = rolled.reindex(tte, method="ffill").to_numpy()
    recent = np.where(np.isfinite(recent), recent, np.nanmedian(rolled.to_numpy()))

    rows.append({"name": name, "h": hte, "err": err, "p50": p50, "y": yte,
                 "vol": featuresq.regime_signal(Xte),
                 "recent": recent, "mp": np.abs(p50 - persist)})
    print(f"fit {name}", flush=True)

H = np.concatenate([r["h"] for r in rows])
ERR = np.concatenate([r["err"] for r in rows])
FOLD = np.concatenate([[r["name"]] * len(r["h"]) for r in rows])
SIG = {k: np.concatenate([r[k] for r in rows]) for k in ("vol", "recent", "mp")}
SIG["blend"] = np.sqrt(np.clip(SIG["recent"], 1e-6, None) * np.clip(SIG["mp"], 1e-6, None))


def conf_lo(a, q):
    n = a.size; return float(np.quantile(a, max(0.0, np.floor((n + 1) * q) / n), method="lower"))


def conf_hi(a, q):
    n = a.size; return float(np.quantile(a, min(1.0, np.ceil((n + 1) * q) / n), method="higher"))


def scale_of(sigkey):
    if sigkey is None:
        return np.ones_like(ERR)
    s = SIG[sigkey].astype(float)
    s = np.where(np.isfinite(s), s, np.nanmedian(s))
    return np.clip(s / np.nanmedian(s), *CLIP)


def evaluate(sigkey):
    scale = scale_of(sigkey)
    score = ERR / scale
    lo_mag, hi_mag = {}, {}
    for hz in HSET:
        sc = score[H == hz]
        if sc.size < 50:
            continue
        lo_mag[hz] = abs(conf_lo(sc, 0.05)); hi_mag[hz] = conf_hi(sc, 0.95)
    order = sorted(lo_mag)
    la = np.maximum.accumulate([lo_mag[k] for k in order])
    ha = np.maximum.accumulate([hi_mag[k] for k in order])
    lo_mag = {k: la[i] for i, k in enumerate(order)}; hi_mag = {k: ha[i] for i, k in enumerate(order)}
    covered = np.empty(ERR.size, bool); width = np.empty(ERR.size)
    for i in range(ERR.size):
        hz = int(H[i]); covered[i] = (-hi_mag[hz] * scale[i] <= ERR[i] <= lo_mag[hz] * scale[i])
        width[i] = (lo_mag[hz] + hi_mag[hz]) * scale[i] * CF
    per_fold = {f: float(covered[FOLD == f].mean()) for f in dict.fromkeys(FOLD)}
    return per_fold, float(np.mean(width)), float(covered.mean())


print(f"\n{'method':>12} {'marg.cov':>9} {'fold-spread':>12} {'2019':>6} {'min-fold':>9} {'meanband_F':>11}")
for key, label in [(None, "static"), ("vol", "vol_norm"), ("recent", "recent_norm"),
                   ("mp", "mp_norm"), ("blend", "blend_norm")]:
    pf, mw, marg = evaluate(key)
    vals = list(pf.values())
    print(f"{label:>12} {marg:>9.3f} {np.std(vals):>12.3f} {pf['2019']:>6.3f} "
          f"{min(vals):>9.3f} {mw:>11.2f}")

print("\nper-fold cover90 by method:")
methods = [(None, "static"), ("vol", "vol_norm"), ("recent", "recent_norm"),
           ("mp", "mp_norm"), ("blend", "blend_norm")]
pfs = {label: evaluate(key)[0] for key, label in methods}
print(f"  {'fold':>6} " + " ".join(f"{l:>12}" for _, l in methods))
for f in dict.fromkeys(FOLD):
    print(f"  {f:>6} " + " ".join(f"{pfs[l][f]:>12.3f}" for _, l in methods))
