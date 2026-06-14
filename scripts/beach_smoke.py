"""Throwaway: smoke-test beach.beach_forecast end to end on live data, the way
publish.py would call it. Builds a minimal 168h trajectory from the buoy file
(p50 = a gentle persistence-ish line; we only need the structure), then checks
the per-day output is sane and prints the implied nearshore-vs-buoy lift."""

import numpy as np
import pandas as pd

import beach

# load buoy history the way publish does (here from the cached file)
b = pd.read_csv("data/buoy.csv", index_col=0, parse_dates=True)
b = b[~b.index.duplicated(keep="last")].sort_index()
t0 = b.index[-1]
obs_now_c = float(b["WTMP"].ffill().iloc[-1])
print(f"t0 = {t0}, buoy now = {obs_now_c:.1f}C / {beach.F(obs_now_c):.1f}F")

# fake but structured buoy trajectory: flat p50 near obs with a tiny warming drift
# and +/- bands, in deg F, matching publish.py's dict shape.
traj = []
for h in range(1, 169):
    valid = t0 + pd.Timedelta(hours=h)
    p50_c = obs_now_c + 0.01 * h          # ~+1.7C drift over the week
    p50 = beach.F(p50_c)
    traj.append({"h": h, "t": valid.isoformat(),
                 "p05": round(p50 - 4, 2), "p25": round(p50 - 1.5, 2),
                 "p50": round(p50, 2), "p75": round(p50 + 1.5, 2),
                 "p95": round(p50 + 4, 2)})

days = beach.beach_forecast(b, traj)
print(f"\nbeach_forecast returned {len(days)} days:")
buoy_daily = {}
for pt in traj:
    buoy_daily.setdefault(pt["t"][:10], []).append(pt["p50"])
for d in days:
    bmean = np.mean(buoy_daily[d["date"]])
    lift = d["beach_p50"] - bmean
    print(f"  {d['date']} {d['label']}  beach {d['beach_p50']:.1f}F "
          f"[{d['beach_lo']:.1f}-{d['beach_hi']:.1f}]   buoy {bmean:.1f}F   "
          f"lift {lift:+.1f}F   band {d['beach_hi']-d['beach_lo']:.1f}F")

# check graceful failure
import os, shutil
print("\n--- graceful: missing beach file returns [] ---")
shutil.move("data/chibeach.csv", "data/chibeach.csv.bak")
try:
    print("  result:", beach.beach_forecast(b, traj))
finally:
    shutil.move("data/chibeach.csv.bak", "data/chibeach.csv")

print("\n--- graceful: empty trajectory returns [] ---")
print("  result:", beach.beach_forecast(b, []))
