"""Throwaway: characterize the Chicago nearshore beach sensors vs the offshore
buoy (45174), so beach.py is built on measured structure, not assumption.

Questions:
  1. typical beach-buoy offset and its seasonal/diurnal shape
  2. diurnal swing amplitude at the beach vs the buoy
  3. dependence on solar / air / wind
  4. correlation and coverage
  5. which beach column to lean on
"""

import numpy as np
import pandas as pd

beach = pd.read_csv("data/chibeach.csv", index_col=0, parse_dates=True)
buoy = pd.read_csv("data/buoy.csv", index_col=0, parse_dates=True)
wx = pd.read_csv("data/weather.csv", index_col=0, parse_dates=True)

BEACH = "beach_ohio"  # nearest to Wilmette, best covered

# ---- coverage, per beach, summer only (the season the model serves) -----------
print("=== COVERAGE (May-Oct hours with both beach + buoy WTMP) ===")
summer = beach.index.month.isin(range(5, 11))
for c in beach.columns:
    j = beach.loc[summer, [c]].join(buoy[["WTMP"]], how="inner").dropna()
    if len(j):
        print(f"  {c:16s} {len(j):5d} hrs  {j.index.min().date()} -> {j.index.max().date()}")

# ---- core overlap frame: beach_ohio + buoy + weather --------------------------
df = beach[[BEACH]].join(buoy[["WTMP", "ATMP", "WSPD"]], how="inner")
df = df.join(wx[["shortwave_radiation", "temperature_2m", "wind_speed_10m"]], how="left")
df = df.dropna(subset=[BEACH, "WTMP"])
df["gap"] = df[BEACH] - df["WTMP"]
df["hour"] = df.index.hour
df["month"] = df.index.month
print(f"\n=== {BEACH} vs buoy: {len(df)} overlapping hours, "
      f"{df.index.min().date()} -> {df.index.max().date()} ===")

# ---- offset / correlation -----------------------------------------------------
print(f"\ngap (beach - buoy), deg C:  mean {df['gap'].mean():+.2f}  "
      f"median {df['gap'].median():+.2f}  std {df['gap'].std():.2f}")
print(f"  pctiles  p05 {df['gap'].quantile(.05):+.2f}  p25 {df['gap'].quantile(.25):+.2f}  "
      f"p75 {df['gap'].quantile(.75):+.2f}  p95 {df['gap'].quantile(.95):+.2f}")
print(f"corr(beach, buoy) = {df[BEACH].corr(df['WTMP']):.3f}")
print(f"naive 'beach=buoy' MAE = {df['gap'].abs().mean():.2f} C "
      f"({df['gap'].abs().mean()*1.8:.2f} F)")

# ---- diurnal shape of the gap -------------------------------------------------
print("\n=== gap by hour-of-day (UTC; local CDT = UTC-5) ===")
by_h = df.groupby("hour")["gap"].agg(["mean", "std", "count"])
for h, r in by_h.iterrows():
    bar = "#" * int(max(0, r["mean"]) * 8)
    print(f"  {h:02d}UTC ({(h-5)%24:02d} CDT)  {r['mean']:+.2f} +/- {r['std']:.2f}  n={int(r['count']):4d} {bar}")

# diurnal swing amplitude: per calendar day, max-min of each series
def daily_swing(s):
    g = s.dropna().groupby(s.dropna().index.normalize())
    amp = g.agg(lambda x: x.max() - x.min())[g.size() >= 12]
    return amp
beach_amp = daily_swing(df[BEACH])
buoy_amp = daily_swing(df["WTMP"])
print(f"\ndiurnal swing (daily max-min, days with >=12 hrs):")
print(f"  beach  median {beach_amp.median():.2f} C  mean {beach_amp.mean():.2f} C  (n={len(beach_amp)} days)")
print(f"  buoy   median {buoy_amp.median():.2f} C  mean {buoy_amp.mean():.2f} C  (n={len(buoy_amp)} days)")

# ---- seasonal shape of the gap ------------------------------------------------
print("\n=== gap by month ===")
for m, r in df.groupby("month")["gap"].agg(["mean", "std", "count"]).iterrows():
    print(f"  month {m:2d}  {r['mean']:+.2f} +/- {r['std']:.2f}  n={int(r['count'])}")

# ---- dependence on solar / air / wind -----------------------------------------
print("\n=== gap correlation with drivers ===")
for c in ["shortwave_radiation", "temperature_2m", "wind_speed_10m", "ATMP", "WSPD", "WTMP"]:
    sub = df[["gap", c]].dropna()
    if len(sub) > 50:
        print(f"  corr(gap, {c:20s}) = {sub['gap'].corr(sub[c]):+.3f}  (n={len(sub)})")

# air-water at buoy as a gap predictor (warm air over cool water -> beach warms more)
df["airwater"] = df["temperature_2m"] - df["WTMP"]
sub = df[["gap", "airwater"]].dropna()
print(f"  corr(gap, air-water diff)        = {sub['gap'].corr(sub['airwater']):+.3f}  (n={len(sub)})")

# ---- gap persistence: does yesterday's mean gap predict today's? ---------------
daily_gap = df["gap"].resample("1D").mean().dropna()
lag1 = daily_gap.autocorr(lag=1)
print(f"\ndaily-mean gap lag-1 autocorr = {lag1:.3f}  (gap memory for the recent-gap feature)")
