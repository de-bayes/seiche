"""Seiche one-page outlook: a single-page, consulting-style brief regenerated
every six hours from the live forecast. Times New Roman, clean exhibits, the
halftone buoy as the only brand mark. Every sentence is a deterministic function
of the numbers (reproducible; no model in the loop). Pure matplotlib + numpy + PIL.

Writes site/briefs/seiche_report_<UTCstamp>.pdf + site/briefs/latest.pdf, pruned."""

import json
import pathlib
import shutil
import textwrap
from datetime import datetime, timezone, timedelta

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.image as mpimg
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

INK, GREY, FAINT = "#1a1a1a", "#5b6470", "#9aa1ab"
NAVY, BAND1, BAND2 = "#1257a0", (0.07, 0.34, 0.63, 0.12), (0.07, 0.34, 0.63, 0.22)
AMBER, GRID, RULE = "#b45309", "#e8e8e8", "#cdd2d8"
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "Liberation Serif", "Times", "DejaVu Serif"],
    "axes.edgecolor": RULE, "axes.linewidth": 0.8, "text.color": INK,
    "axes.labelcolor": GREY, "xtick.color": GREY, "ytick.color": GREY,
    "xtick.labelsize": 6, "ytick.labelsize": 6, "mathtext.fontset": "dejavuserif",
})
LETTER = (8.5, 11.0)
M = 0.07
ASSETS, BRIEFS_DIR, BRIEFS_KEEP = pathlib.Path("site/assets"), pathlib.Path("site/briefs"), 28
CENTRAL = timezone(timedelta(hours=-5))
ct = lambda iso: datetime.fromisoformat(iso).astimezone(CENTRAL).replace(tzinfo=None)


# ---------- data ----------
def load(name, default=None):
    p = pathlib.Path(name)
    return json.loads(p.read_text()) if p.exists() else default


def derive(d):
    traj, now_f = d["trajectory"], d["now"]["wtmp_f"]
    day = traj[:24]; p50d = [p["p50"] for p in day]
    hi, lo = int(np.argmax(p50d)), int(np.argmin(p50d))
    wk = [p["p50"] for p in traj]; hist = [p["f"] for p in d["history"]]
    d48 = now_f - hist[0] if hist else 0.0
    return {
        "now_f": now_f, "at24": traj[23]["p50"], "chg24": traj[23]["p50"] - now_f,
        "p05_24": traj[23]["p05"], "p95_24": traj[23]["p95"],
        "hi": (day[hi]["p50"], ct(day[hi]["t"])), "lo": (day[lo]["p50"], ct(day[lo]["t"])),
        "wk_med": float(np.mean(wk)), "wk_hi": max(wk), "wk_lo": min(wk),
        "band168": (traj[167]["p95"] - traj[167]["p05"]) / 2, "d48": d48,
        "day6": d["daily"][-1]["p50"], "band_scale": d.get("band_scale"),
    }


def _dir(x, a, b, c, eps=0.3):
    return c if abs(x) < eps else (a if x > 0 else b)


def lede(d, dv):
    cmp = ("a touch below" if dv["now_f"] < dv["wk_med"] - 0.3 else
           "a touch above" if dv["now_f"] > dv["wk_med"] + 0.3 else "in line with")
    move = _dir(dv["chg24"], "warm", "cool", "hold near")
    return (f"Water stands at {dv['now_f']:.1f}°F, {cmp} the week's {dv['wk_med']:.1f}°F median, "
            f"and is set to {move} to {dv['at24']:.1f}°F by tomorrow ({dv['chg24']:+.1f}°F). "
            f"Expect a {dv['wk_lo']:.0f}–{dv['wk_hi']:.0f}°F range across the seven days; confidence "
            f"is firm inside two days and the band widens to about ±{dv['band168']:.1f}°F by day seven.")


def note(d, dv, v):
    trend = (f"risen {abs(dv['d48']):.1f}°F over two days" if dv["d48"] > 0.4 else
             f"fallen {abs(dv['d48']):.1f}°F over two days" if dv["d48"] < -0.4 else
             "held roughly flat over two days")
    h = (v or {}).get("headline", {}) or {}
    if h.get("mae_f"):
        conf = (f" Over the last thirty days the typical one-day miss was {h['mae_f']:.2f}°F and the "
                f"90% band held {h['cover90']*100:.0f}% of the time.")
    else:
        conf = " Skill is shown on a nine-season replay until live forecasts accumulate."
    return f"The water has {trend}; the forecast is anchored to the live buoy, so the next hours are near-certain and the week carries the doubt.{conf}"


# ---------- chrome ----------
def clean(ax, ygrid=True):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(RULE)
    ax.tick_params(length=0)
    if ygrid:
        ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)


def ex(fig, rect, n, title, sub=None):
    left, bottom, w, h = rect
    fig.text(left, bottom + h + 0.022, f"EXHIBIT {n}", fontsize=6.5, color=NAVY, fontweight="bold")
    fig.text(left + 0.072, bottom + h + 0.022, title, fontsize=8.5, color=INK, fontweight="bold")
    if sub:
        fig.text(left, bottom + h + 0.009, sub, fontsize=6.6, color=GREY, style="italic")
    return fig.add_axes(rect)


def datefmt(ax, fmt="%a"):
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter(fmt))


# ---------- the page ----------
def build(fig, d, dv, v, qs, issue_ct, valid_ct):
    # masthead
    mh = fig.add_axes([M, 0.915, 1 - 2 * M, 0.06]); mh.axis("off"); mh.set_xlim(0, 1); mh.set_ylim(0, 1)
    mh.text(0, 0.62, "Seiche", fontsize=21, fontweight="bold", color=INK, va="center")
    mh.text(0.005, 0.18, "Lake Michigan Water-Temperature Outlook", fontsize=9.5, style="italic",
            color=GREY, va="center")
    try:
        im = mpimg.imread(str(ASSETS / "buoy-halftone.png"))
        iax = fig.add_axes([0.752, 0.913, 0.178, 0.062]); iax.axis("off")
        iax.imshow(im, aspect="auto", interpolation="bilinear")
    except Exception:
        pass
    mh.axhline(0.0, color=INK, lw=1.6); mh.axhline(-0.10, color=INK, lw=0.6)
    fig.text(1 - M, 0.904, f"Issued {issue_ct}  ·  NDBC 45174, Wilmette, Illinois",
             fontsize=6.6, color=GREY, ha="right")

    # key takeaway
    tk = fig.add_axes([M, 0.838, 1 - 2 * M, 0.06]); tk.axis("off"); tk.set_xlim(0, 1); tk.set_ylim(0, 1)
    tk.text(0, 0.96, "KEY TAKEAWAY", fontsize=6.5, color=NAVY, fontweight="bold", va="top")
    tk.text(0, 0.74, textwrap.fill(lede(d, dv), width=118), fontsize=9.6, color=INK, va="top",
            fontweight="bold", linespacing=1.3)
    tk.axhline(0.0, color=RULE, lw=0.8)

    # exhibit 1: seven-day cone
    ax = ex(fig, [M, 0.585, 1 - 2 * M, 0.20], 1, "Seven-day forecast", "Observed (black) into the 90% forecast band, °F")
    tj = d["trajectory"]; tjt = [ct(p["t"]) for p in tj]
    ax.fill_between(tjt, [p["p05"] for p in tj], [p["p95"] for p in tj], color=BAND1, lw=0)
    ax.fill_between(tjt, [p["p25"] for p in tj], [p["p75"] for p in tj], color=BAND2, lw=0)
    ax.plot(tjt, [p["p50"] for p in tj], color=NAVY, lw=2.0)
    ax.plot([ct(p["t"]) for p in d["history"]], [p["f"] for p in d["history"]], color=INK, lw=1.3)
    ax.axvline(ct(d["valid_utc"]), color=FAINT, lw=0.8, ls=(0, (2, 2)))
    datefmt(ax, "%a %-d"); clean(ax)

    # exhibit 2: daily outlook
    days = d["daily"]
    ax2 = ex(fig, [M, 0.38, 0.40, 0.135], 2, "Daily outlook", "Median dot, 90% band, °F")
    ax2.set_xlim(-0.5, len(days) - 0.5)
    for i, day in enumerate(days):
        ax2.plot([i, i], [day["p05"], day["p95"]], color="#b9c6da", lw=4, solid_capstyle="round", zorder=1)
        ax2.scatter([i], [day["p50"]], s=22, color=NAVY, zorder=3)
        ax2.text(i, day["p95"] + 0.25, f"{day['p50']:.0f}", fontsize=6.5, color=INK, ha="center", fontweight="bold")
    ax2.axhline(dv["now_f"], color=FAINT, lw=0.7, ls=(0, (3, 3)))
    ax2.set_xticks(range(len(days))); ax2.set_xticklabels([x["label"] for x in days])
    clean(ax2, ygrid=False)

    # exhibit 3: the day ahead
    day = d["trajectory"][:24]; t = [ct(p["t"]) for p in day]
    ax3 = ex(fig, [M + 0.46, 0.38, 1 - 2 * M - 0.46, 0.135], 3, "The day ahead", "Next 24 hours, hourly, °F")
    ax3.fill_between(t, [p["p05"] for p in day], [p["p95"] for p in day], color=BAND1, lw=0)
    ax3.fill_between(t, [p["p25"] for p in day], [p["p75"] for p in day], color=BAND2, lw=0)
    ax3.plot(t, [p["p50"] for p in day], color=NAVY, lw=2.0)
    ax3.scatter([ct(d["valid_utc"])], [dv["now_f"]], s=18, color=INK, zorder=5)
    ax3.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 6)))
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%-I%p")); clean(ax3)

    # exhibit 4: accuracy vs persistence
    by = (v or {}).get("by_lead", [])
    ax4 = ex(fig, [M, 0.175, 0.40, 0.135], 4, "Accuracy vs. “no change”", "Mean error by lead, °F (lower is better)")
    if by:
        hs = [b["h"] for b in by]
        ax4.plot(hs, [b["mae_persist_f"] for b in by], color=GREY, lw=1.3, ls=(0, (4, 3)))
        ax4.plot(hs, [b["mae_f"] for b in by], color=NAVY, lw=2.0)
        ax4.text(hs[-1], by[-1]["mae_persist_f"], " no change", fontsize=6, color=GREY, va="center")
        ax4.text(hs[-1], by[-1]["mae_f"], " Seiche", fontsize=6, color=NAVY, va="center", fontweight="bold")
        ax4.set_xlabel("lead time (hours)", fontsize=6.5)
    clean(ax4)

    # exhibit 5: leading factors
    ax5 = ex(fig, [M + 0.46, 0.175, 1 - 2 * M - 0.46, 0.135], 5, "Leading factors", "Permutation importance, °F")
    imp = (qs or {}).get("importance", [])[:6][::-1]
    if imp:
        LBL = {"WTMP": "water now", "h": "lead time", "fut_t2m": "air ahead", "fut_v": "alongshore wind",
               "fut_wspd": "wind ahead", "fut_solar": "sun ahead", "fut_gust": "gusts ahead",
               "doy_cos": "season", "doy_sin": "season", "fut_airwater": "air–water gap",
               "fut_dewdep": "evaporation", "WVHT": "waves now", "atmp_minus_wtmp": "air–water gap"}
        names = [LBL.get(x["name"], x["name"]) for x in imp]
        vals = [x["value"] for x in imp]
        ax5.barh(range(len(imp)), vals, color=NAVY, height=0.62)
        for i, val in enumerate(vals):
            ax5.text(val, i, f" {val:.2f}", fontsize=6, color=GREY, va="center")
        ax5.set_yticks(range(len(imp))); ax5.set_yticklabels(names, fontsize=6.5)
        ax5.set_xlim(0, max(vals) * 1.18)
        for s in ("top", "right", "bottom"):
            ax5.spines[s].set_visible(False)
        ax5.set_xticks([]); ax5.tick_params(length=0)

    # analysis + source note
    fn = fig.add_axes([M, 0.045, 1 - 2 * M, 0.085]); fn.axis("off"); fn.set_xlim(0, 1); fn.set_ylim(0, 1)
    fn.axhline(0.96, color=RULE, lw=0.8)
    fn.text(0, 0.82, textwrap.fill(note(d, dv, v), width=132), fontsize=7.6, color=INK, va="top", linespacing=1.3)
    fn.text(0, 0.06, "Source: NDBC 45174; ERA5; GFS/ECMWF/ICON/GEM ensemble (Open-Meteo). Five quantile "
            "models, adaptive-conformal bands, nine-season backtest. Regenerated every six hours.",
            fontsize=5.8, color=FAINT, va="bottom")


# ---------- main ----------
def main():
    d = load("site/data.json")
    if d is None:
        raise SystemExit("site/data.json not found; run publish.py first")
    v = load("site/verify.json", {}); qs = load("models/qstats.json", {})
    dv = derive(d)
    issued = datetime.now(timezone.utc)
    issue_ct = issued.astimezone(CENTRAL).strftime("%-d %B %Y, %-I:%M %p CT")
    valid_ct = datetime.fromisoformat(d["valid_utc"]).astimezone(CENTRAL).strftime("%-d %b %-I:%M %p")

    BRIEFS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = issued.strftime("%Y%m%d_%H%MZ")
    out = BRIEFS_DIR / f"seiche_report_{stamp}.pdf"
    with PdfPages(out) as pdf:
        fig = plt.figure(figsize=LETTER)
        build(fig, d, dv, v, qs, issue_ct, valid_ct)
        pdf.savefig(fig); plt.close(fig)
        meta = pdf.infodict(); meta["Title"] = f"Seiche outlook {stamp}"; meta["Author"] = "Seiche"

    shutil.copy(out, BRIEFS_DIR / "latest.pdf")
    kept = sorted(BRIEFS_DIR.glob("seiche_report_*.pdf"))
    for old in kept[:-BRIEFS_KEEP]:
        old.unlink()
    for old in BRIEFS_DIR.glob("seiche_brief_*.pdf"):
        old.unlink()
    print(f"wrote {out} (+ latest.pdf); {len(kept[-BRIEFS_KEEP:])} reports kept")


if __name__ == "__main__":
    main()
