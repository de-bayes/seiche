"""Throwaway: pick the production model. Compare Ridge vs HGB, with and without
the recent-gap anchor, leave-one-year-out. The cold-start column is the honest
worst case (sensor down, no recent beach reading). We want a model that beats
naive WITH the anchor and at least does not BLOW UP without it.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

beach = pd.read_csv("data/chibeach.csv", index_col=0, parse_dates=True)
buoy = pd.read_csv("data/buoy.csv", index_col=0, parse_dates=True)
wx = pd.read_csv("data/weather.csv", index_col=0, parse_dates=True)
BEACH = "beach_ohio"
F = lambda c: c * 1.8


def build(df):
    h = df.index.hour
    out = pd.DataFrame(index=df.index)
    out["wtmp"] = df["WTMP"]
    out["hour_sin"] = np.sin(2 * np.pi * h / 24)
    out["hour_cos"] = np.cos(2 * np.pi * h / 24)
    out["month_sin"] = np.sin(2 * np.pi * df.index.month / 12)
    out["month_cos"] = np.cos(2 * np.pi * df.index.month / 12)
    out["solar"] = df["shortwave_radiation"]
    out["airwater"] = df["temperature_2m"] - df["WTMP"]
    out["gap_recent"] = df["gap"].rolling(24, min_periods=3).mean().shift(1)
    return out


df = beach[[BEACH]].join(buoy[["WTMP"]], how="inner")
df = df.join(wx[["shortwave_radiation", "temperature_2m"]], how="left")
df = df.dropna(subset=[BEACH, "WTMP"])
df["gap"] = df[BEACH] - df["WTMP"]
df = df[df.index.month.isin(range(5, 11))]
X = build(df); y = df[BEACH]
years = sorted(df.index.year.unique())

# gap_recent fallback value used at cold start: the SEASONAL climatology gap,
# i.e. mean gap by month from training only. This is what beach.py will fall
# back to when no live beach reading exists, so test it honestly.
def make_models():
    return {
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "hgb": HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05,
                max_iter=400, min_samples_leaf=40, l2_regularization=1.0, random_state=0),
    }


def score(fill_cold):
    """fill_cold: how to fill gap_recent for the cold-start test.
       'nan' -> leave NaN (HGB handles; ridge needs impute)
       'clim' -> training monthly-climatology gap (what production falls back to)."""
    out = {k: [] for k in ["ridge", "hgb"]}
    naive_all = []
    for ty in years:
        tr = df.index.year != ty; te = df.index.year == ty
        if te.sum() < 50 or tr.sum() < 500:
            continue
        Xtr, Xte = X[tr], X[te]
        ytr, yte = y[tr], y[te].to_numpy()
        # monthly climatology gap from training
        clim = df.loc[tr].groupby(df.loc[tr].index.month)["gap"].mean()
        te_clim = df.loc[te].index.month.map(clim).to_numpy()
        for name, m in make_models().items():
            if name == "ridge":
                tr_imp = Xtr.fillna(Xtr.mean()); te_imp = Xte.fillna(Xtr.mean())
                m.fit(tr_imp, ytr)
                if fill_cold == "clim":
                    tc = te_imp.copy(); tc["gap_recent"] = te_clim
                else:
                    tc = te_imp.copy(); tc["gap_recent"] = Xtr["gap_recent"].mean()
                pred = m.predict(tc)
            else:
                m.fit(Xtr, ytr)
                tc = Xte.copy()
                tc["gap_recent"] = te_clim if fill_cold == "clim" else np.nan
                pred = m.predict(tc)
            out[name].append((F(np.mean(np.abs(pred - yte))), te.sum()))
        naive_all.append((F(np.mean(np.abs(df.loc[te, "WTMP"].to_numpy() - yte))), te.sum()))
    wavg = lambda L: np.average([a for a, _ in L], weights=[n for _, n in L])
    return wavg(naive_all), {k: wavg(v) for k, v in out.items()}


# warm-start (has recent gap) reference
def score_warm():
    out = {k: [] for k in ["ridge", "hgb"]}; naive = []
    for ty in years:
        tr = df.index.year != ty; te = df.index.year == ty
        if te.sum() < 50 or tr.sum() < 500: continue
        for name, m in make_models().items():
            if name == "ridge":
                m.fit(X[tr].fillna(X[tr].mean()), y[tr])
                pred = m.predict(X[te].fillna(X[tr].mean()))
            else:
                m.fit(X[tr], y[tr]); pred = m.predict(X[te])
            out[name].append((F(np.mean(np.abs(pred - y[te].to_numpy()))), te.sum()))
        naive.append((F(np.mean(np.abs(df.loc[te,"WTMP"].to_numpy()-y[te].to_numpy()))), te.sum()))
    wavg = lambda L: np.average([a for a,_ in L], weights=[n for _,n in L])
    return wavg(naive), {k: wavg(v) for k,v in out.items()}

nw, warm = score_warm()
nc, cold = score("clim")
print(f"naive baseline MAE        {nw:.2f} F\n")
print(f"{'model':8s}  {'warm (recent gap)':>18s}  {'cold (clim gap)':>16s}")
for k in ["ridge", "hgb"]:
    print(f"{k:8s}  {warm[k]:>15.2f} F   {cold[k]:>13.2f} F")
print(f"\nimprovement vs naive: ridge warm {(1-warm['ridge']/nw)*100:.0f}%, "
      f"cold {(1-cold['ridge']/nw)*100:.0f}%")
