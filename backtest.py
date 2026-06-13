"""Rolling-origin backtest across up to nine seasons. For each fold, the test
window is the final 45 in-season days of a year (35 for the current one);
training uses only data ending 8 days before the window, then P5/P50/P95
models are fit fresh and scored with the same anchor blending production
uses. Nothing from any test window ever touches its fold's training.

Writes models/backtest.json and reports/backtest.png."""

import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor

import featuresq

TAU = 8.0
CF = 1.8
DIFF_CLIP = (0.5, 2.5)   # how far the adaptive band may shrink / stretch vs typical


def trailing_recent_error(tte, hte, err):
    """ACI-style difficulty signal: for each base time, the trailing-48h mean of
    the realized +24h |error| that has already resolved by then (the +24h call
    launched at base b is known at b+24h). One value per row, constant across
    horizons for a given base time. Elevated through sustained-miss regimes
    (the 2019 fold), which is exactly where one static band width under-covers.
    publish.py rebuilds the same quantity live from its recent +24h forecasts."""
    b24, e24 = tte[hte == 24], np.abs(np.asarray(err)[hte == 24])
    s = pd.Series(e24, index=b24).sort_index()
    s = s[~s.index.duplicated(keep="last")]
    resolved = s.copy()
    resolved.index = resolved.index + pd.Timedelta(hours=24)   # when each error becomes known
    rolled = resolved.rolling("48h").mean()
    recent = rolled.reindex(tte, method="ffill").to_numpy()
    return np.where(np.isfinite(recent), recent, np.nanmedian(rolled.to_numpy()))


buoy = pd.read_csv("data/buoy.csv", index_col=0, parse_dates=True)
wx = pd.read_csv("data/weather.csv", index_col=0, parse_dates=True)
X, y, t, h = featuresq.stack(buoy, wx)
yv = y.to_numpy()

folds = []
for year in [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]:
    in_year = t[(t.year == year)]
    if len(in_year) == 0:
        continue
    wend = in_year.max()
    folds.append((str(year), wend - pd.Timedelta(days=45), wend))
folds.append(("2026", t.max() - pd.Timedelta(days=35), t.max()))

results = {"folds": [], "horizons": featuresq.HSET}
total_pairs = 0
resid_by_h = {hz: [] for hz in featuresq.HSET}  # anchored-median residuals (deg C)
band_by_h = {hz: [] for hz in featuresq.HSET}   # 90% quantile-band span (deg C)
fold_raw = []                                   # per-fold (resid, recent, h) for calib + diagnostic
for name, wstart, wend in folds:
    tr = t <= (wstart - pd.Timedelta(days=8))
    te = (t >= wstart) & (t <= wend)
    if te.sum() < 1000:
        print(f"fold {name}: too small, skipped")
        continue
    models = {}
    for q in [0.05, 0.5, 0.95]:
        m = HistGradientBoostingRegressor(loss="quantile", quantile=q, max_iter=250,
                                          learning_rate=0.08, random_state=11)
        m.fit(X[tr], y[tr])
        models[q] = m
    Xte, yte, hte, tte = X[te], yv[te], h[te], t[te]
    pred = {q: models[q].predict(Xte) for q in models}
    persist = Xte["WTMP"].to_numpy()

    m1 = hte == 1
    delta_by_t = dict(zip(tte[m1].values, persist[m1] - pred[0.5][m1]))
    delta = np.array([delta_by_t.get(v, 0.0) for v in tte.values])
    decay = np.exp(-(hte - 1) / TAU)
    for q in pred:
        pred[q] = pred[q] + delta * decay

    resid_fold = pred[0.5] - yte
    recent = trailing_recent_error(tte, np.asarray(hte), resid_fold)
    fold_raw.append({"name": name, "resid": resid_fold,
                     "recent": recent, "h": np.asarray(hte)})

    fold = {"name": name, "n": int(te.sum()), "train_n": int(tr.sum()),
            "window": f"{wstart:%Y-%m-%d} to {wend:%Y-%m-%d}", "mae": [], "mae_persist": [], "cover90": []}
    for hz in featuresq.HSET:
        m = hte == hz
        if m.sum() < 30:
            fold["mae"].append(None)
            fold["mae_persist"].append(None)
            fold["cover90"].append(None)
            continue
        fold["mae"].append(round(float(np.mean(np.abs(pred[0.5][m] - yte[m]))) * CF, 3))
        fold["mae_persist"].append(round(float(np.mean(np.abs(persist[m] - yte[m]))) * CF, 3))
        fold["cover90"].append(round(float(np.mean((yte[m] >= pred[0.05][m]) & (yte[m] <= pred[0.95][m]))), 3))
        resid_by_h[hz].append(pred[0.5][m] - yte[m])
        band_by_h[hz].append(pred[0.95][m] - pred[0.05][m])
    results["folds"].append(fold)
    total_pairs += int(te.sum())
    at24 = fold["mae"][featuresq.HSET.index(24)]
    p24 = fold["mae_persist"][featuresq.HSET.index(24)]
    print(f"fold {name} ({fold['window']}): {fold['n']} pairs · +24h mae {at24}F vs persist {p24}F")

results["total_pairs"] = total_pairs
mean = lambda key, i: round(float(np.mean([f[key][i] for f in results["folds"] if f[key][i] is not None])), 3)
results["mean_mae"] = [mean("mae", i) for i in range(len(featuresq.HSET))]
results["mean_mae_persist"] = [mean("mae_persist", i) for i in range(len(featuresq.HSET))]
results["mean_cover90"] = [mean("cover90", i) for i in range(len(featuresq.HSET))]

# band calibration from the pooled out-of-sample residuals across all folds:
# the empirical 5/25/75/95 quantiles of the anchored median's error per horizon
# (deg C). publish.py turns these into the displayed bands. Split-conformal with
# the finite-sample rank correction guarantees >= nominal coverage on
# exchangeable errors: upper fences round their rank up, lower fences round
# down, per the standard ceil((n+1)q)/n construction.
def conformal(r):
    n = r.size
    hi = lambda q: float(np.quantile(r, min(1.0, np.ceil((n + 1) * q) / n), method="higher"))
    lo = lambda q: float(np.quantile(r, max(0.0, np.floor((n + 1) * q) / n), method="lower"))
    return {"e05": round(lo(0.05), 4), "e25": round(lo(0.25), 4),
            "e75": round(hi(0.75), 4), "e95": round(hi(0.95), 4), "n": int(n)}


calib = {}
for hz in featuresq.HSET:
    if not resid_by_h[hz]:
        continue
    r = np.concatenate(resid_by_h[hz])
    c = conformal(r)
    c["band90_mean_c"] = round(float(np.mean(np.concatenate(band_by_h[hz]))), 4)
    c["conformal"] = True
    calib[str(hz)] = c
results["calib"] = calib

# Adaptive normalized conformal. Pooled conformal is only MARGINALLY valid: one
# static width per horizon under-covers sustained-miss regimes (the 2019 fold
# sat at 0.75) and over-covers calm ones. So normalize each error by an adaptive
# difficulty scale -- the trailing realized +24h error vs its typical level --
# conformalize the SCALED residuals per horizon, and let publish.py re-inflate
# by the live scale. Coverage stays marginally valid by construction and becomes
# far more uniform across folds (a method bake-off over vol / |median-persist| /
# recent-error picked recent-error: fold spread 0.082 -> 0.028, 2019 0.75 ->
# 0.93). scripts/band_method_compare.py has the comparison.
RESID = np.concatenate([fr["resid"] for fr in fold_raw])
RECENT = np.concatenate([fr["recent"] for fr in fold_raw])
HARR = np.concatenate([fr["h"] for fr in fold_raw])
ref = float(np.nanmedian(RECENT))
scale_all = np.clip(np.where(np.isfinite(RECENT), RECENT, ref) / ref, *DIFF_CLIP)
score = RESID / scale_all

calib_norm = {"ref": round(ref, 4), "clip": list(DIFF_CLIP), "window_h": 48, "horizons": {}}
for hz in featuresq.HSET:
    sc = score[HARR == hz]
    if sc.size < 50:
        continue
    calib_norm["horizons"][str(hz)] = conformal(sc)   # quantiles in SCALE units
results["calib_norm"] = calib_norm

# --- coverage diagnostic: per-fold cover90 of the band publish.py draws
# (magnitude fences made monotone in lead; weather term is zero here since the
# backtest runs under reanalysis weather), static vs adaptive. ---
def fences(table):
    hs = sorted(int(k) for k in table if k.isdigit())
    lo = np.maximum.accumulate([abs(table[str(k)]["e05"]) for k in hs])
    hi = np.maximum.accumulate([abs(table[str(k)]["e95"]) for k in hs])
    return {k: (lo[i], hi[i]) for i, k in enumerate(hs)}


static_f = fences(calib)
norm_f = fences(calib_norm["horizons"])
cover_diag = {"by_fold": [], "corr_recent_abserr": None, "method": "recent_norm"}
for fr in fold_raw:
    resid, hh = fr["resid"], fr["h"]
    sc = np.clip(np.where(np.isfinite(fr["recent"]), fr["recent"], ref) / ref, *DIFF_CLIP)
    ok_s = ok_a = 0
    wid_s = wid_a = 0.0
    for r, hz, s in zip(resid, hh, sc):
        lo_s, hi_s = static_f[int(hz)]
        lo_a, hi_a = norm_f[int(hz)]
        ok_s += (-hi_s <= r <= lo_s)
        ok_a += (-hi_a * s <= r <= lo_a * s)
        wid_s += (lo_s + hi_s) * CF
        wid_a += (lo_a + hi_a) * s * CF
    n = len(resid)
    cover_diag["by_fold"].append({
        "name": fr["name"], "n": int(n),
        "cover90_static": round(ok_s / n, 3), "cover90_adaptive": round(ok_a / n, 3),
        "band_F_static": round(wid_s / n, 2), "band_F_adaptive": round(wid_a / n, 2),
        "mean_recent": round(float(np.nanmean(fr["recent"])), 3)})
absE = np.abs(RESID)
g = np.isfinite(RECENT) & np.isfinite(absE)
cover_diag["corr_recent_abserr"] = round(float(np.corrcoef(RECENT[g], absE[g])[0, 1]), 3)
results["cover_diag"] = cover_diag

sd = lambda k: float(np.std([f[k] for f in cover_diag["by_fold"]]))
print("\nadaptive (recent-error normalized) conformal bands:")
print(f"  difficulty ref (median trailing +24h err, deg C): {ref:.3f} · clip {DIFF_CLIP}")
print(f"  corr(recent error, |error|) pooled: {cover_diag['corr_recent_abserr']:+.3f}")
print(f"  per-fold cover90 spread (lower=more uniform): static {sd('cover90_static'):.3f}"
      f" -> adaptive {sd('cover90_adaptive'):.3f}")
print(f"  {'fold':>6} {'recent':>7} {'cov static':>11} {'cov adapt':>10} {'band_F st':>10} {'band_F ad':>10}")
for f in cover_diag["by_fold"]:
    print(f"  {f['name']:>6} {f['mean_recent']:>7.3f} {f['cover90_static']:>11.3f} "
          f"{f['cover90_adaptive']:>10.3f} {f['band_F_static']:>10.2f} {f['band_F_adaptive']:>10.2f}")

with open("models/backtest.json", "w") as fh:
    json.dump(results, fh)

plt.style.use("dark_background")
BG, PANEL, INK, CYAN, FAINT = "#0a0c0e", "#101418", "#cfdce8", "#39c2ff", "#5d7283"
fig, ax = plt.subplots(figsize=(11, 5.6), facecolor=BG)
ax.set_facecolor(PANEL)
hs = featuresq.HSET
for f in results["folds"]:
    ax.plot(hs, f["mae"], color=CYAN, alpha=0.30, lw=1.1)
    ax.plot(hs, f["mae_persist"], color=FAINT, alpha=0.30, lw=1.1, ls="--")
ax.plot(hs, results["mean_mae"], color=CYAN, lw=2.6, marker="o", ms=4, label="model, all-season mean")
ax.plot(hs, results["mean_mae_persist"], color=FAINT, lw=2, ls="--", label="persistence, all-season mean")
ax.set_xlabel("lead time (hours)", color=INK)
ax.set_ylabel("MAE (deg F)", color=INK)
ax.set_title(f"{len(results['folds'])}-season rolling backtest · {total_pairs:,} forecast/outcome pairs · thin lines are individual seasons",
             color="w", loc="left")
ax.set_xticks([1, 24, 48, 72, 96, 120, 144, 168])
ax.grid(alpha=0.15)
ax.tick_params(colors=INK)
ax.legend(frameon=False, labelcolor=INK, fontsize=10)
fig.tight_layout()
fig.savefig("reports/backtest.png", dpi=150, facecolor=BG)
print(f"\ntotal: {total_pairs:,} verified forecast pairs across {len(results['folds'])} seasons")
print("wrote models/backtest.json and reports/backtest.png")
