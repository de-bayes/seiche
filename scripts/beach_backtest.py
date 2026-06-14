"""Throwaway: backtest the nearshore model honestly. Predict beach_ohio from
buoy WTMP + diurnal + season + recent gap + solar/air, held out by YEAR (so a
whole season is unseen), vs the naive 'beach = buoy' baseline. Report MAE and
band coverage. This decides the model used in beach.py.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

beach = pd.read_csv("data/chibeach.csv", index_col=0, parse_dates=True)
buoy = pd.read_csv("data/buoy.csv", index_col=0, parse_dates=True)
wx = pd.read_csv("data/weather.csv", index_col=0, parse_dates=True)
BEACH = "beach_ohio"


def build(df):
    """Feature frame. Live-available at forecast time: buoy WTMP forecast, the
    valid hour/month, recent solar+air, and the recent beach-buoy gap (the last
    measured offset, which the live pipeline carries as a scalar)."""
    h = df.index.hour
    out = pd.DataFrame(index=df.index)
    out["wtmp"] = df["WTMP"]
    out["hour_sin"] = np.sin(2 * np.pi * h / 24)
    out["hour_cos"] = np.cos(2 * np.pi * h / 24)
    out["month_sin"] = np.sin(2 * np.pi * df.index.month / 12)
    out["month_cos"] = np.cos(2 * np.pi * df.index.month / 12)
    out["solar"] = df["shortwave_radiation"]
    out["airwater"] = df["temperature_2m"] - df["WTMP"]
    # recent gap: trailing 24h mean of (beach-buoy), shifted 1h so it never peeks
    out["gap_recent"] = df["gap"].rolling(24, min_periods=3).mean().shift(1)
    return out


df = beach[[BEACH]].join(buoy[["WTMP", "ATMP"]], how="inner")
df = df.join(wx[["shortwave_radiation", "temperature_2m"]], how="left")
df = df.dropna(subset=[BEACH, "WTMP"])
df["gap"] = df[BEACH] - df["WTMP"]
df = df[df.index.month.isin(range(5, 11))]   # season the model serves
X = build(df)
y = df[BEACH]
FEATS = list(X.columns)

years = sorted(df.index.year.unique())
print(f"{len(df)} hours, years {years}")
print(f"features: {FEATS}\n")

# leave-one-year-out: each season is the held-out test set once
F = lambda c: c * 1.8
rows = []
for ty in years:
    tr = df.index.year != ty
    te = df.index.year == ty
    if te.sum() < 50 or tr.sum() < 500:
        continue
    Xtr, Xte = X[tr].copy(), X[te].copy()
    ytr, yte = y[tr], y[te]
    # gap_recent leaks the within-test-year gap only from its own past 24h, which
    # the live pipeline genuinely has; that is fair. But to test the COLD-START
    # case (no recent beach data) we also score with gap_recent blanked.
    m = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05,
                                      max_iter=400, min_samples_leaf=40,
                                      l2_regularization=1.0, random_state=0)
    m.fit(Xtr, ytr)
    pred = m.predict(Xte)
    Xte_cold = Xte.copy(); Xte_cold["gap_recent"] = np.nan
    pred_cold = m.predict(Xte_cold)

    naive = df.loc[te, "WTMP"].to_numpy()      # baseline: beach = buoy
    yt = yte.to_numpy()
    mae_model = np.mean(np.abs(pred - yt))
    mae_cold = np.mean(np.abs(pred_cold - yt))
    mae_naive = np.mean(np.abs(naive - yt))
    rows.append({"year": ty, "n": int(te.sum()),
                 "mae_naive_F": F(mae_naive), "mae_model_F": F(mae_model),
                 "mae_cold_F": F(mae_cold)})

res = pd.DataFrame(rows)
print("=== Leave-one-year-out (MAE deg F) ===")
print(res.to_string(index=False,
      formatters={"mae_naive_F": "{:.2f}".format, "mae_model_F": "{:.2f}".format,
                  "mae_cold_F": "{:.2f}".format}))
w = res["n"]
mn = np.average(res["mae_naive_F"], weights=w)
mm = np.average(res["mae_model_F"], weights=w)
mc = np.average(res["mae_cold_F"], weights=w)
print(f"\nweighted  naive {mn:.2f}F  |  model {mm:.2f}F ({(1-mm/mn)*100:.0f}% better)  "
      f"|  cold-start {mc:.2f}F ({(1-mc/mn)*100:.0f}% better)")

# also a transparent Ridge for comparison (sanity: HGB worth it?)
mr_list = []
for ty in years:
    tr = df.index.year != ty; te = df.index.year == ty
    if te.sum() < 50 or tr.sum() < 500:
        continue
    Xtr = X[tr].fillna(X[tr].mean()); Xte = X[te].fillna(X[tr].mean())
    r = Ridge(alpha=1.0).fit(Xtr, y[tr])
    mr_list.append((np.mean(np.abs(r.predict(Xte) - y[te].to_numpy())) * 1.8, te.sum()))
mr = np.average([a for a, _ in mr_list], weights=[n for _, n in mr_list])
print(f"ridge (reference)  {mr:.2f}F")

# ---- band sizing: residual quantiles from the held-out errors -----------------
# Build pooled out-of-fold residuals to size symmetric-ish bands.
oof = []
for ty in years:
    tr = df.index.year != ty; te = df.index.year == ty
    if te.sum() < 50 or tr.sum() < 500:
        continue
    m = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=400,
                                      min_samples_leaf=40, l2_regularization=1.0, random_state=0)
    m.fit(X[tr], y[tr])
    e = y[te].to_numpy() - m.predict(X[te])
    oof.extend(e.tolist())
oof = np.array(oof)
print(f"\n=== out-of-fold residuals (beach actual - model), deg C, n={len(oof)} ===")
for q in [5, 25, 50, 75, 95]:
    print(f"  p{q:02d}  {np.percentile(oof, q):+.2f} C  ({np.percentile(oof, q)*1.8:+.2f} F)")
print(f"  90% half-width ~ {(np.percentile(oof,95)-np.percentile(oof,5))/2:.2f} C "
      f"({(np.percentile(oof,95)-np.percentile(oof,5))/2*1.8:.2f} F)")
lo, hi = np.percentile(oof, 5), np.percentile(oof, 95)
cov = np.mean((oof >= lo) & (oof <= hi))
print(f"  empirical 90% coverage of [p05,p95] band: {cov:.2f}")
