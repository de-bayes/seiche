"""Turn the calibrated forecast distribution into decisions: swim guidance,
threshold-exceedance probabilities, and alert conditions. Pure functions over
the data.json the model already produces (temps in deg F); publish.py wires them
in and handles delivery (ntfy push). Thresholds are explicit and transparent --
this is judgment from the numbers, labeled as such, not a model output."""

import numpy as np
import pandas as pd

QUANTILE_P = np.array([0.05, 0.25, 0.50, 0.75, 0.95])
SWIM_THRESHOLDS = [60, 65, 70]            # cold / cool / pleasant boundaries, deg F
CT = pd.Timedelta(hours=-5)               # CDT, for day grouping/labels


# ---------- swim guidance ----------
def swim_comfort(water_f, air_f=None, wind_kt=None):
    """Plain-language swim read from water temp, with an air/wind exit caveat and
    a rough cold-exposure guideline (general guidance, not medical advice)."""
    w = water_f
    if w < 55:
        cat, label, expo = "frigid", "Frigid. Cold-shock risk; serious cold-water skill or a wetsuit only.", "minutes"
    elif w < 60:
        cat, label, expo = "very cold", "Very cold. A wetsuit for anything but a quick acclimation dip.", "~10-20 min"
    elif w < 65:
        cat, label, expo = "cold", "Cold but swimmable for the acclimated; brisk and short.", "~20-45 min"
    elif w < 70:
        cat, label, expo = "cool", "Cool and refreshing once you're in; comfortable open-water swimming.", "45+ min"
    elif w < 75:
        cat, label, expo = "pleasant", "Pleasant. Easy, comfortable swimming.", "no cold limit"
    else:
        cat, label, expo = "warm", "Warm. Lingering, lake-day swimming.", "no cold limit"
    note = ""
    if air_f is not None and wind_kt is not None and air_f < w - 2 and wind_kt > 12:
        note = "A cold wind off the water will bite on the way out; bring a warm layer."
    return {"category": cat, "label": label, "exposure": expo, "exit_note": note,
            "water_f": round(float(w), 1)}


# ---------- threshold-exceedance probabilities ----------
def _p_at_least(quantiles, T):
    """P(water >= T) from 5 calibrated quantiles, linear-interp CDF, clamped to
    [0.05, 0.95] (honest given five points)."""
    q = np.sort(np.asarray(quantiles, dtype=float))
    p_le = float(np.interp(T, q, QUANTILE_P))     # clamps to 0.05 / 0.95 at the tails
    return round(1.0 - p_le, 2)


def threshold_probs(trajectory, thresholds=SWIM_THRESHOLDS):
    """Per-day P(water >= T) (averaged over the day's hours) plus, per threshold,
    the first day it's more-likely-than-not (>=50%)."""
    rows = []
    for pt in trajectory:
        day = (pd.Timestamp(pt["t"]) + CT).strftime("%Y-%m-%d")
        rows.append((day, [pt["p05"], pt["p25"], pt["p50"], pt["p75"], pt["p95"]]))
    by_day_q = {}
    for day, q in rows:
        by_day_q.setdefault(day, []).append(q)

    by_day = []
    for day in sorted(by_day_q):
        qs = by_day_q[day]
        p_ge = {str(T): round(float(np.mean([_p_at_least(q, T) for q in qs])), 2) for T in thresholds}
        label = pd.Timestamp(day).strftime("%a")
        by_day.append({"date": day, "label": label, "p_ge": p_ge})

    crossings = []
    for T in thresholds:
        hit = next((d for d in by_day if d["p_ge"][str(T)] >= 0.5), None)
        crossings.append({
            "temp": T,
            "first_day": hit["label"] if hit else None,
            "first_date": hit["date"] if hit else None,
            "prob": hit["p_ge"][str(T)] if hit else (by_day[-1]["p_ge"][str(T)] if by_day else None),
            "already": by_day[0]["p_ge"][str(T)] >= 0.5 if by_day else False,
        })
    return {"thresholds": thresholds, "by_day": by_day, "crossings": crossings}


# ---------- alerts ----------
def alerts(now_f, daily, thresh, swim_now):
    """Actionable conditions from the week ahead. Returns a list of
    {level, kind, title, detail}. level: 'good' | 'watch'."""
    out = []
    # 1. a swim threshold likely reached this week (and not already there)
    for c in thresh["crossings"]:
        if c["first_day"] and not c["already"] and c["temp"] >= 65:
            out.append({"level": "good", "kind": "threshold",
                        "title": f"Water likely to reach {c['temp']}°F by {c['first_day']}",
                        "detail": f"{int(c['prob']*100)}% chance the water tops {c['temp']}°F by {c['first_day']}."})
            break  # only the most relevant threshold
    # 2. sharp cooling / possible upwelling: a big median drop within the next ~4 days
    if daily:
        wk = [d["p50"] for d in daily[:5]]
        drop = now_f - min(wk)
        if drop >= 4.0:
            di = int(np.argmin(wk))
            out.append({"level": "watch", "kind": "cold_snap",
                        "title": f"Sharp cooling ahead: about {drop:.0f}°F by {daily[di]['label']}",
                        "detail": f"The water may fall from {now_f:.0f}°F to ~{min(wk):.0f}°F by "
                                  f"{daily[di]['label']} (a possible upwelling). Plan colder swims."})
    # 3. a swim window opening (cold now, pleasant later)
    if swim_now["category"] in ("frigid", "very cold", "cold") and daily:
        good = next((d for d in daily if d["p50"] >= 68), None)
        if good:
            out.append({"level": "good", "kind": "window",
                        "title": f"Swim window opening: ~{good['p50']:.0f}°F by {good['label']}",
                        "detail": f"Water warms toward a comfortable ~{good['p50']:.0f}°F by {good['label']}."})
    return out
