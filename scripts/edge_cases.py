"""What are the worst-case forecasts? The 90% band is ±4°F at day 7 because the
worst ~10% of forecasts miss big; this characterizes those tail events to ask
whether they are reducible (a leading signal predicts them) or the lake's
intrinsic chaos.

Reproduces the 9-season backtest (median HGB per fold, anchored, ERA5 = perfect
weather, so these are the model's INTRINSIC errors, not weather-forecast error)
and records, for every (base time, lead) forecast, the signed error plus the
conditions at forecast time and how far the water actually moved. Then dissects
the worst decile at +72/+120/+168 h. Prints a report."""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

import featuresq

TAU, CF = 8.0, 1.8
LEADS = [72, 120, 168]
buoy = pd.read_csv("data/buoy.csv", index_col=0, parse_dates=True)
wx = pd.read_csv("data/weather.csv", index_col=0, parse_dates=True)
X, y, t, h = featuresq.stack(buoy, wx)
yv = y.to_numpy()
obs = buoy["WTMP"]

folds = []
for year in [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]:
    iy = t[t.year == year]
    if len(iy):
        folds.append((str(year), iy.max() - pd.Timedelta(days=45), iy.max()))
folds.append(("2026", t.max() - pd.Timedelta(days=35), t.max()))

rows = []
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
    err = (p50 - yte) * CF                                  # signed error, deg F (+ = model too warm)
    move = (yte - persist) * CF                             # how far the water actually moved, deg F
    pe = (persist - yte) * CF                               # persistence error, deg F
    for i in range(len(yte)):
        if hte[i] not in LEADS:
            continue
        rows.append({
            "fold": name, "base": tte[i], "h": int(hte[i]), "err": err[i], "abserr": abs(err[i]),
            "move": move[i], "absmove": abs(move[i]), "perr_abs": abs(pe[i]),
            "wspd": Xte["WSPD"].iloc[i], "u": Xte["u"].iloc[i], "v": Xte["v"].iloc[i],
            "wspd_m24": Xte["wspd_m24"].iloc[i], "v_m24": Xte["v_m24"].iloc[i],
            "airwater": Xte["atmp_minus_wtmp"].iloc[i], "wtmp_d24": Xte["wtmp_d24"].iloc[i],
            "pres_d3": Xte["pres_d3"].iloc[i], "wvht": Xte["WVHT"].iloc[i],
            "doy": int(pd.Timestamp(tte[i]).dayofyear), "month": int(pd.Timestamp(tte[i]).month),
        })
    print(f"fold {name} done", flush=True)

df = pd.DataFrame(rows)
print(f"\n{len(df)} long-lead forecasts ({'/'.join(map(str,LEADS))} h)\n")

DRIVERS = ["absmove", "wspd", "wspd_m24", "v", "v_m24", "u", "airwater", "wtmp_d24", "pres_d3", "wvht"]
for lead in LEADS:
    s = df[df.h == lead]
    cut = s["abserr"].quantile(0.90)
    tail = s[s.abserr >= cut]; rest = s[s.abserr < cut]
    print(f"=== +{lead}h  ({len(s)} fc) · worst-10% threshold |err| >= {cut:.2f}F ===")
    print(f"  median |err|: all {s.abserr.median():.2f}  tail {tail.abserr.median():.2f}  "
          f"(tail mean {tail.abserr.mean():.2f}F)")
    warm = (tail.err > 0).mean()
    print(f"  tail sign: {warm*100:.0f}% model-too-warm (positive err) · "
          f"mean signed {tail.err.mean():+.2f}F")
    print(f"  actual move |Δwater| over the window: rest {rest.absmove.mean():.2f}  tail {tail.absmove.mean():.2f}F")
    print(f"  persistence |err| on the SAME tail rows: {tail.perr_abs.mean():.2f}F "
          f"(model {tail.abserr.mean():.2f}F) -> model {'still beats' if tail.abserr.mean()<tail.perr_abs.mean() else 'loses to'} persistence in the tail")
    print(f"  corr(|err|, driver) over all +{lead}h fc:")
    cors = {d: s["abserr"].corr(s[d].abs() if d in ("u", "v", "v_m24", "airwater", "wtmp_d24", "pres_d3") else s[d]) for d in DRIVERS}
    for d, c in sorted(cors.items(), key=lambda kv: -abs(kv[1])):
        print(f"      {d:10} {c:+.3f}")
    by_month = s.groupby("month").abserr.mean()
    print("  mean |err| by month: " + " ".join(f"{m}:{v:.2f}" for m, v in by_month.items()))
    by_fold = s.groupby("fold").apply(lambda g: (g.abserr >= cut).mean(), include_groups=False)
    print("  share of tail by fold: " + " ".join(f"{f}:{v*100:.0f}%" for f, v in by_fold.items()))
    print()

# biggest single events at +168h
big = df[df.h == 168].nlargest(12, "abserr")
print("=== 12 worst +168h misses ===")
print(f"  {'base date':12} {'err':>6} {'Δwater':>7} {'wspd':>5} {'v_m24':>6} {'air-h2o':>7} {'month':>5}")
for _, r in big.iterrows():
    print(f"  {pd.Timestamp(r.base):%Y-%m-%d} {r.err:>+6.1f} {r.move:>+7.1f} {r.wspd:>5.1f} "
          f"{r.v_m24:>+6.2f} {r.airwater:>+7.1f} {r.month:>5}")
