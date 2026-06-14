"""Nearshore "beach" water-temperature estimate, alongside the offshore buoy
forecast. The buoy (45174) sits in deeper, cooler water and lags; the nearshore
beach where people actually swim warms more in the sun and swings more day to
night. This maps the published buoy forecast onto a Chicago-shelf nearshore
estimate with honest uncertainty.

WHAT THE DATA SAYS (data/chibeach.csv, Ohio St beach ~25 km south of Wilmette,
9.9k summer hours overlapping the buoy, 2017-2026):
  - the beach runs WARMER than the buoy, but strongly seasonally: +2.7C in May,
    +1.9C in June, falling to ~+0.5C by Jul-Sep as the lake stratifies/warms;
  - the gap shrinks as the buoy warms (corr gap vs WTMP = -0.53);
  - a clear diurnal cycle: gap peaks ~+1.5C mid-afternoon, troughs ~+0.5C
    pre-dawn, and the beach's daily swing (~1.8C) is twice the buoy's (~0.9C);
  - the recent measured gap is sticky (daily lag-1 autocorr 0.64), the single
    most useful predictor when a fresh beach reading exists.

MODEL: a transparent Ridge on [buoy WTMP, hour-of-day (sin/cos), month (sin/cos),
recent solar, air-water diff, recent beach-buoy gap]. Leave-one-year-out backtest
(each season held out once), MAE deg F, vs the naive "beach = buoy" baseline:
    naive 2.19  |  with a fresh recent-gap anchor 1.04 (53% better)
                |  cold start, climatology gap     1.61 (26% better)

HONEST LIMITATION: this is the Ohio Street beach, a SOUTHERN Chicago-shelf
station, used as a proxy. It is NOT a Wilmette measurement, and the sensor is
intermittent (~34% hourly coverage, May-Oct only). The output is therefore
labelled a nearshore/shelf adjustment, not a Wilmette beach reading. When no
recent beach reading exists the anchor falls back to monthly climatology and the
bands widen accordingly.

Public entry point:
    beach_forecast(buoy_hourly_df, trajectory) -> [ {date, label,
        beach_p50, beach_lo, beach_hi} ... ]  per day, deg F.
Graceful: returns [] if the beach data is missing/unusable so publish.py simply
omits the beach block.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

BEACH_PATH = "data/chibeach.csv"
WX_PATH = "data/weather.csv"
BUOY_PATH = "data/buoy.csv"        # full buoy history for fitting (live path passes only ~45d realtime)
COL = "beach_ohio"                 # nearest to Wilmette and best-covered sensor
SEASON = range(5, 11)             # May-Oct: the sensor deploys, the lake matters
GAP_FRESH_DAYS = 12              # how recent a beach reading must be to anchor on
MIN_TRAIN = 800                 # refuse to train on too-thin overlap
F = lambda c: c * 1.8 + 32

FEATS = ["wtmp", "hour_sin", "hour_cos", "month_sin", "month_cos",
         "solar", "airwater", "gap_recent"]


def _feature_frame(wtmp, hour, month, solar, airwater, gap_recent):
    """Assemble the model's feature columns from already-aligned arrays/scalars.
    Diurnal and seasonal terms are cyclic so the model sees the afternoon bump
    and the May-to-September gap decay smoothly."""
    arr = lambda x: np.asarray(x, dtype=float)   # drop any index so columns align by position
    out = pd.DataFrame({
        "wtmp": arr(wtmp),
        "hour_sin": np.sin(2 * np.pi * arr(hour) / 24),
        "hour_cos": np.cos(2 * np.pi * arr(hour) / 24),
        "month_sin": np.sin(2 * np.pi * arr(month) / 12),
        "month_cos": np.cos(2 * np.pi * arr(month) / 12),
        "solar": arr(solar),
        "airwater": arr(airwater),
        "gap_recent": arr(gap_recent),
    })
    return out[FEATS]


def _train():
    """Fit Ridge on the historical beach/buoy/weather overlap (summer only) and
    return the fitted pipeline, the residual band quantiles (out of sample shape,
    deg C), and the monthly-climatology gap used as the cold-start anchor.
    Reads the FULL buoy history from data/buoy.csv (the live path only carries
    ~45 days of realtime, too thin to fit). Returns None if too little data."""
    try:
        beach = pd.read_csv(BEACH_PATH, index_col=0, parse_dates=True)
        wx = pd.read_csv(WX_PATH, index_col=0, parse_dates=True)
        buoy_df = pd.read_csv(BUOY_PATH, index_col=0, parse_dates=True)
    except FileNotFoundError:
        return None
    if COL not in beach.columns:
        return None

    df = beach[[COL]].join(buoy_df[["WTMP"]], how="inner")
    df = df.join(wx[["shortwave_radiation", "temperature_2m"]], how="left")
    df = df.dropna(subset=[COL, "WTMP"])
    df = df[df.index.month.isin(SEASON)]
    if len(df) < MIN_TRAIN:
        return None

    df["gap"] = df[COL] - df["WTMP"]
    # recent gap: trailing-24h mean offset, shifted 1h so it never uses the value
    # it is predicting -- exactly the sticky anchor the live path reconstructs.
    df["gap_recent"] = df["gap"].rolling(24, min_periods=3).mean().shift(1)
    X = _feature_frame(df["WTMP"], df.index.hour, df.index.month,
                       df["shortwave_radiation"], df["temperature_2m"] - df["WTMP"],
                       df["gap_recent"])
    X.index = df.index
    clim_gap = df.groupby(df.index.month)["gap"].mean()   # cold-start fallback
    Xf = X.fillna({"solar": X["solar"].mean(),
                   "airwater": X["airwater"].mean(),
                   "gap_recent": clim_gap.mean()})
    y = df[COL]

    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(Xf, y)
    # band shape from out-of-fold (leave-one-year-out) residuals so the spread
    # reflects genuine held-out error, not in-sample optimism.
    resid = []
    for ty in df.index.year.unique():
        tr, te = df.index.year != ty, df.index.year == ty
        if te.sum() < 50 or tr.sum() < 500:
            continue
        m = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(Xf[tr], y[tr])
        resid.extend((y[te].to_numpy() - m.predict(Xf[te])).tolist())
    resid = np.array(resid) if resid else (y.to_numpy() - model.predict(Xf))
    qband = {q: float(np.percentile(resid, q)) for q in (5, 50, 95)}
    return model, qband, clim_gap


def _recent_gap(buoy_df, t0):
    """The latest measured beach-buoy offset (deg C), if a beach reading exists
    within GAP_FRESH_DAYS of t0; else None so the caller falls back to climatology.
    Uses the same trailing-24h mean the model was trained on."""
    try:
        beach = pd.read_csv(BEACH_PATH, index_col=0, parse_dates=True)
    except FileNotFoundError:
        return None
    s = beach[COL].dropna() if COL in beach.columns else pd.Series(dtype=float)
    s = s[s.index <= t0]
    if s.empty or (t0 - s.index.max()) > pd.Timedelta(days=GAP_FRESH_DAYS):
        return None
    gap = (s - buoy_df["WTMP"].reindex(s.index)).dropna()
    gap = gap[gap.index > gap.index.max() - pd.Timedelta(hours=48)]
    return float(gap.tail(24).mean()) if len(gap) >= 3 else None


def beach_forecast(buoy_hourly_df, trajectory):
    """Map publish.py's buoy forecast `trajectory` to a per-day nearshore beach
    estimate. `trajectory` is a list of {h, t (ISO valid time), p05..p95} in deg F
    (we use the p50 line as the buoy temperature). Returns a list of per-day dicts
    {date, label, beach_p50, beach_lo, beach_hi} in deg F. Returns [] gracefully
    if the beach data is unavailable so publish.py can omit the block."""
    if not trajectory:
        return []
    trained = _train()
    if trained is None:
        return []
    model, qband, clim_gap = trained

    t0 = buoy_hourly_df.index[-1]
    gap_now = _recent_gap(buoy_hourly_df, t0)        # None -> use monthly climatology
    fresh = gap_now is not None
    # extra band inflation when we have no fresh anchor: the cold-start backtest
    # ran ~55% wider error than the warm case (1.61F vs 1.04F MAE), so widen the
    # residual bands by that ratio rather than pretend the climatology is as sharp.
    cold_widen = 1.0 if fresh else 1.55

    # pull weather for the forecast window (solar/air drive the diurnal bump); if
    # the forecast horizon outruns the file it simply ffills, then the cyclic hour
    # term still carries the diurnal shape.
    try:
        wx = pd.read_csv(WX_PATH, index_col=0, parse_dates=True)
    except FileNotFoundError:
        wx = None

    # per-hour nearshore estimate over the trajectory
    times = pd.to_datetime([p["t"] for p in trajectory])
    wtmp_c = np.array([(p["p50"] - 32) / 1.8 for p in trajectory])     # buoy p50 -> C
    months = times.month.to_numpy()
    gap_recent = np.full(len(times), gap_now if fresh else np.nan)
    if not fresh:
        gap_recent = months_to_clim(months, clim_gap)
    if wx is not None:
        w = wx.reindex(times, method="nearest", tolerance=pd.Timedelta("3h"))
        solar = w["shortwave_radiation"].to_numpy()
        air = w["temperature_2m"].to_numpy()
    else:
        solar = np.full(len(times), np.nan)
        air = np.full(len(times), np.nan)
    # fill any weather gaps with the model's training-mean stand-ins (the cyclic
    # hour/month terms still encode the diurnal/seasonal shape without them)
    solar = np.where(np.isfinite(solar), solar, np.nanmean(solar) if np.isfinite(solar).any() else 200.0)
    airwater = np.where(np.isfinite(air), air - wtmp_c, 0.0)

    X = _feature_frame(wtmp_c, times.hour.to_numpy(), months, solar, airwater, gap_recent)
    beach_c = model.predict(X)
    band_c = {q: beach_c + qband[q] * (cold_widen if q != 50 else 1.0) for q in (5, 50, 95)}

    # aggregate to per-day p50 / lo / hi in deg F, matching publish.py's daily digest
    out, byday = [], {}
    for i, ts in enumerate(times):
        byday.setdefault(ts.date().isoformat(), []).append(i)
    for date, idx in sorted(byday.items()):
        if len(idx) < 12:                              # need most of a day to average
            continue
        idx = np.array(idx)
        out.append({
            "date": date,
            "label": pd.Timestamp(date).strftime("%a"),
            "beach_p50": round(float(F(np.mean(beach_c[idx]))), 1),
            "beach_lo": round(float(F(np.mean(band_c[5][idx]))), 1),
            "beach_hi": round(float(F(np.mean(band_c[95][idx]))), 1),
        })
    return out


def months_to_clim(months, clim_gap):
    """Map each forecast month to its training-climatology beach-buoy gap (deg C),
    falling back to the overall mean for any month with no history."""
    overall = float(clim_gap.mean())
    return np.array([float(clim_gap.get(m, overall)) for m in months])
