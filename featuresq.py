"""Features for the horizon-conditioned quantile model. One row = (time t,
horizon h): buoy state at t, weather aggregated over (t, t+h] under GENERIC
column names, and h itself as a feature, so a single model serves every
horizon from +1 h to +168 h.

assemble() is the single source of truth for the feature columns; train.py,
backtest.py, and publish.py all go through it (or inference_rows for live),
so the column set can never drift between fit and predict.

History note: an across-lake neighbor buoy (45026), season degree-day, and
cumulative wind-stress features were trialled here. On the 131k-pair nine-season
backtest they carried near-zero permutation importance and slightly raised MAE
(over-reaction variance), so they were dropped. Only dewpoint depression
(evaporative cooling) survived; the neighbor is still fetched for possible
later use. See DOCS / git history."""

import pathlib

import numpy as np
import pandas as pd

import features
import features7

HSET = [1, 2, 3, 6, 9, 12, 18, 24, 36, 48, 60, 72, 96, 120, 144, 168]
QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]

FUT_COLS = ["fut_u", "fut_v", "fut_wspd", "fut_t2m", "fut_solar", "fut_gust",
            "fut_dewdep", "fut_airwater"]

# columns (all present in both stack() and inference_rows(), so no model retrain)
# used to gauge how turbulent the water is right now -- the std of the last-24 h
# WTMP lag ladder. See regime_signal().
VOL_COLS = ["WTMP", "wtmp_l1", "wtmp_l2", "wtmp_l3", "wtmp_l6", "wtmp_l12", "wtmp_l24"]


def regime_signal(X):
    """Recent water-temp volatility at each base time (deg C): the dispersion of
    the last-24 h WTMP lag ladder. A candidate difficulty signal for conditional
    conformal bands. It was tested (scripts/band_method_compare.py) but lost to
    the trailing realized-error scale used in backtest.py: volatility tracks
    |error| only +0.22 and is NOT elevated in the 2019 regime-miss fold, so it
    could not fix that fold's under-coverage. Kept for the analysis scripts."""
    ladder = X[VOL_COLS].to_numpy(dtype=float)
    return np.nanstd(ladder, axis=1)

# open-lake buoy still fetched (data/buoy_neighbor.csv) but not used as a
# feature; it added variance without skill on the backtest.
NEIGHBOR_STATION = "45026"
NEIGHBOR_PATH = "data/buoy_neighbor.csv"

# subsurface/basin streams PROMOTED into production 2026-06-14: NOAA LMHOFS 3D
# lake-physics model surface temp at the station + MUR satellite basin SST. They
# give the surface buoy eyes below the surface and cut the long-lead upwelling
# tail (worst-decile MAE 4.2->3.4F, pinball -10%) without hurting calm leads. The
# old validate_streams2 rejected them on MEDIAN MAE; on the tail / 90% band they
# win. Built by streams.py; missing data -> NaN (HGB handles). See DOCS.md.
STREAM_SETS = ("SAT", "PHYS")
STREAM_COLS = ["sat_basin", "sat_grad_near", "sat_grad_far", "sat_basin_d3", "sat_err",
               "lmhofs_now", "lmhofs_fut", "lmhofs_delta", "lmhofs_err"]


def _attach_streams(buoy_df, Xs, horizons):
    """Concat the promoted stream features onto a stack()-ordered frame. Lazy
    import (streams imports featuresq). Any failure -> all-NaN columns so the
    pipeline never breaks when a stream is unavailable."""
    n = len(Xs)
    try:
        import streams
        blk = streams.build_blocks(buoy_df, horizons=horizons, sets=STREAM_SETS)
        blk = blk.reindex(columns=STREAM_COLS).reset_index(drop=True)
    except Exception as e:
        print(f"  streams unavailable ({e}); filling NaN")
        blk = pd.DataFrame(np.nan, index=range(n), columns=STREAM_COLS)
    return pd.concat([Xs.reset_index(drop=True), blk], axis=1)


def load_neighbor():
    p = pathlib.Path(NEIGHBOR_PATH)
    if not p.exists():
        return None
    return pd.read_csv(p, index_col=0, parse_dates=True)


def future_generic(wx, h, wtmp):
    f = pd.DataFrame(index=wx.index)
    f["fut_u"] = wx["u"].rolling(h).mean().shift(-h)
    f["fut_v"] = wx["v"].rolling(h).mean().shift(-h)
    f["fut_wspd"] = wx["wind_speed_10m"].rolling(h).mean().shift(-h)
    f["fut_t2m"] = wx["temperature_2m"].rolling(h).mean().shift(-h)
    f["fut_solar"] = wx["shortwave_radiation"].rolling(h).mean().shift(-h)
    f["fut_gust"] = wx["wind_gusts_10m"].rolling(h).max().shift(-h)
    # mean dewpoint depression over the window: how hard evaporation pulls heat
    # off the surface (the one weather covariate the trial features kept)
    f["fut_dewdep"] = (wx["temperature_2m"] - wx["dew_point_2m"]).rolling(h).mean().shift(-h)
    f["fut_airwater"] = f["fut_t2m"] - wtmp
    return f


def assemble(buoy_df, wx, h):
    """Full feature frame for a fixed horizon h, indexed by base time. Same
    column set for every horizon and for live inference."""
    base = features.build(buoy_df)
    wxp = features7.prep_weather(wx)
    X = base.join(future_generic(wxp, h, buoy_df["WTMP"]), how="left")
    X["h"] = float(h)
    return X


def stack(buoy_df, wx, horizons=HSET):
    """Stacked training matrix across horizons. Returns X, y, base_time, h."""
    blocks, ys, ts, hs = [], [], [], []
    for h in horizons:
        X = assemble(buoy_df, wx, h)
        y = buoy_df["WTMP"].shift(-h)
        ok = y.notna() & buoy_df["WTMP"].notna()
        blocks.append(X[ok])
        ys.append(y[ok])
        ts.append(X.index[ok])
        hs.append(np.full(int(ok.sum()), h))
    Xs = _attach_streams(buoy_df, pd.concat(blocks, ignore_index=True), horizons)
    return (Xs, pd.concat(ys, ignore_index=True),
            pd.DatetimeIndex(np.concatenate([t.values for t in ts])), np.concatenate(hs))


def inference_rows(buoy_df, wx, t0, horizons):
    """Feature rows at a single base time t0 for each horizon (live forecast)."""
    base = features.build(buoy_df).loc[[t0]]
    wxp = features7.prep_weather(wx)
    after = wxp.loc[wxp.index > t0]
    wtmp0 = buoy_df["WTMP"].loc[t0]
    rows = []
    for h in horizons:
        win = after.iloc[:h]
        row = base.copy()
        row["fut_u"] = win["u"].mean()
        row["fut_v"] = win["v"].mean()
        row["fut_wspd"] = win["wind_speed_10m"].mean()
        row["fut_t2m"] = win["temperature_2m"].mean()
        row["fut_solar"] = win["shortwave_radiation"].mean()
        row["fut_gust"] = win["wind_gusts_10m"].max()
        row["fut_dewdep"] = (win["temperature_2m"] - win["dew_point_2m"]).mean()
        row["fut_airwater"] = row["fut_t2m"].iloc[0] - wtmp0
        row["h"] = float(h)
        rows.append(row)
    out = pd.concat(rows, ignore_index=True)
    try:
        import streams
        blk = streams.inference_block(buoy_df, t0, horizons, sets=STREAM_SETS)
        blk = blk.reindex(columns=STREAM_COLS).reset_index(drop=True)
    except Exception as e:
        print(f"  streams unavailable at inference ({e}); filling NaN")
        blk = pd.DataFrame(np.nan, index=range(len(out)), columns=STREAM_COLS)
    return pd.concat([out, blk], axis=1)
