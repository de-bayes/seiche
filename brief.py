"""Seiche brief: an auto-generated, timestamped PDF outlook regenerated every few
hours from the live forecast (site/data.json + site/verify.json + the validation
stats). Page 1 is the seven-day outlook, page 2 the day ahead in detail, page 3
the recent (backtested) track record. Paper-white "Making Software" aesthetic,
one accent; the header is a dot-stipple standing wave (a seiche) behind the
wordmark with a buoy mark drawn natively. Pure matplotlib + numpy, no SVG.

Writes site/briefs/seiche_brief_<UTCstamp>.pdf plus a stable site/briefs/latest.pdf,
and prunes to the most recent BRIEFS_KEEP files."""

import json
import pathlib
import shutil
from datetime import datetime, timezone, timedelta

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

# ---- palette (site design tokens) ----
INK, MUTED, FAINT = "#16181d", "#5b6470", "#8a929c"
ACCENT, AMBER, GREEN = "#1257a0", "#b45309", "#2f7d5b"
RULE, WASH, RED = "#ececec", "#fdf3c7", "#e0312e"
MEMBER = (0.47, 0.51, 0.56)
DOT = (20 / 255, 24 / 255, 30 / 255)
plt.rcParams.update({
    "font.serif": ["Lora", "Georgia", "DejaVu Serif"], "font.family": "serif",
    "font.monospace": ["IBM Plex Mono", "Menlo", "DejaVu Sans Mono"],
    "axes.edgecolor": RULE, "axes.linewidth": 0.8, "text.color": INK,
    "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
})
MONO = {"family": "monospace"}
A4 = (8.27, 11.69)
M = 0.085
BRIEFS_DIR = pathlib.Path("site/briefs")
BRIEFS_KEEP = 28
CENTRAL = timezone(timedelta(hours=-5))   # CDT, display only (season is May-Nov)

ct = lambda iso: datetime.fromisoformat(iso).astimezone(CENTRAL).replace(tzinfo=None)


# ---------- data + derived ----------
def load(name, default=None):
    p = pathlib.Path(name)
    return json.loads(p.read_text()) if p.exists() else default


def derive(d):
    traj = d["trajectory"]
    now_f = d["now"]["wtmp_f"]
    day = traj[:24]
    p50d = [pt["p50"] for pt in day]
    hi, lo = int(np.argmax(p50d)), int(np.argmin(p50d))
    wk = [pt["p50"] for pt in traj]
    bs = d.get("band_scale")
    if bs is None or 0.95 <= bs <= 1.05:
        state = "NORMAL"
    elif bs > 1.05:
        state = f"{bs:.2f}x WIDE"
    else:
        state = f"{bs:.2f}x TIGHT"
    return {
        "now_f": now_f, "at24": traj[23]["p50"], "chg24": traj[23]["p50"] - now_f,
        "p05_24": traj[23]["p05"], "p95_24": traj[23]["p95"],
        "hi": (day[hi]["p50"], ct(day[hi]["t"])), "lo": (day[lo]["p50"], ct(day[lo]["t"])),
        "wk_med": float(np.mean(wk)), "band168": (traj[167]["p95"] - traj[167]["p05"]) / 2,
        "band_state": state,
    }


def swim_word(wt, air, wind):
    if wt < 60: s = "Very cold. Numbing fast; wetsuit weather."
    elif wt < 65: s = "Cold. A brisk, short dip for the acclimated."
    elif wt < 70: s = "Cool but swimmable. Refreshing once you are in."
    elif wt < 75: s = "Pleasant. Comfortable open-water swimming."
    else: s = "Warm. Easy, lingering swims."
    if air < wt - 2 and wind > 12:
        s += " Wind off the water will bite on the way out."
    return s


# ---------- shared chrome ----------
def stipple_wave(ax, n=1300, seed=7, span=0.60):
    """Faint dot-stipple standing wave (a seiche), confined to the left `span`
    so the right-aligned timestamps stay clean."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, n)
    env = np.cos(np.pi * x) + 0.4 * (0.35 * np.cos(2 * np.pi * x + 0.6))
    y = 0.5 + 0.16 * env + rng.normal(0, 0.05, n)
    edge = np.clip(1 - np.abs(y - 0.5) / 0.30, 0, 1)
    fade = np.clip(1 - x, 0, 1)                     # taper toward the right edge
    rgba = np.zeros((n, 4)); rgba[:, 0], rgba[:, 1], rgba[:, 2] = DOT
    rgba[:, 3] = (0.05 + 0.05 * edge) * (0.4 + 0.6 * fade)
    ax.scatter(x * span, y, s=rng.uniform(1.2, 3.0, n) ** 2, c=rgba, linewidths=0,
               transform=ax.transAxes, zorder=0)
    xs = np.linspace(0, 1, 240)
    ys = 0.5 + 0.16 * (np.cos(np.pi * xs) + 0.4 * (0.35 * np.cos(2 * np.pi * xs + 0.6)))
    ax.plot(xs * span, ys, color=ACCENT, lw=1.0, alpha=0.12, transform=ax.transAxes, zorder=0)


def buoy_glyph(ax, cx=0.012, base=0.34):
    xs = np.linspace(-1, 1, 80)
    bell = np.exp(-(xs ** 2) / 0.28)
    ax.fill_between(xs * 0.014 + cx, base, base + bell * 0.30, color=ACCENT, alpha=0.9,
                    transform=ax.transAxes, zorder=2, lw=0)
    ax.plot([cx, cx], [base + 0.30, base + 0.50], color=INK, lw=1.5, transform=ax.transAxes, zorder=2)
    ax.scatter([cx], [base + 0.52], s=20, color=RED, transform=ax.transAxes, zorder=3)


def header(fig, issued_ct, valid_ct, kicker=None):
    ax = fig.add_axes([M, 0.885, 1 - 2 * M, 0.085]); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    stipple_wave(ax)
    buoy_glyph(ax)
    ax.text(0.055, 0.60, "S E I C H E", fontsize=18, fontweight="bold", color=INK,
            transform=ax.transAxes, va="center")
    sub = kicker or "LAKE MICHIGAN WATER-TEMPERATURE OUTLOOK · NDBC 45174, EVANSTON–WILMETTE"
    ax.text(0.057, 0.28, sub, fontsize=6.6, color=ACCENT if kicker else MUTED,
            transform=ax.transAxes, va="center", **MONO)
    ax.text(0.995, 0.62, f"ISSUED {issued_ct}", fontsize=7.2, color=MUTED, ha="right",
            va="center", transform=ax.transAxes, **MONO)
    ax.text(0.995, 0.40, f"DATA VALID {valid_ct} · REFRESH ~6 H", fontsize=7.2, color=FAINT,
            ha="right", va="center", transform=ax.transAxes, **MONO)
    ax.axhline(0.04, color=RULE, lw=0.8)
    ax.plot([0.0, 0.05], [0.04, 0.04], color=ACCENT, lw=2.4, transform=ax.transAxes)


def strip(fig, y, cells, h=0.05):
    """Bracketed mono label/value row."""
    ax = fig.add_axes([M, y, 1 - 2 * M, h]); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axhline(0.96, color=RULE, lw=0.8); ax.axhline(0.04, color=RULE, lw=0.8)
    n = len(cells)
    for i, (lab, val, col) in enumerate(cells):
        x = (i + 0.5) / n
        ax.text(x, 0.66, lab, fontsize=6, color=MUTED, ha="center", va="center", **MONO)
        ax.text(x, 0.30, val, fontsize=12.5, color=col, ha="center", va="center",
                fontweight="bold", **MONO)


def kick(fig, y, text):
    ax = fig.add_axes([M, y, 1 - 2 * M, 0.03]); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.text(0, 0.3, text, fontsize=7.5, color=ACCENT, fontweight="bold", **MONO)
    ax.axhline(0.0, color=RULE, lw=0.8)


def foot(fig, page, total):
    fig.text(M, 0.035, "Seiche outlook", fontsize=7, color=FAINT)
    fig.text(1 - M, 0.035, f"page {page} of {total}", fontsize=7, color=FAINT, ha="right")


def datefmt(ax):
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %-d"))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    ax.grid(axis="y", color=RULE, lw=0.6)


# ---------- page 1: seven-day outlook ----------
def page_seven_day(fig, d, dv, issued_ct, valid_ct):
    header(fig, issued_ct, valid_ct)
    now = d["now"]
    strip(fig, 0.835, [
        ("WATER", f"{now['wtmp_f']:.1f}°F", INK), ("AIR", f"{now['atmp_f']:.1f}°F", INK),
        ("WAVES", f"{now['wvht_ft']:.1f} ft", INK), ("WIND", f"{now['wspd_kt']:.1f} kt", INK),
        ("GUST", f"{now['gst_kt']:.1f} kt", INK)])

    # hero 168h fan
    ax = fig.add_axes([M, 0.45, 1 - 2 * M, 0.345])
    tj_t = [ct(p["t"]) for p in d["trajectory"]]
    for m in d.get("members", []):
        ax.plot(tj_t, m["traj"], color=MEMBER, alpha=0.04, lw=0.6, zorder=1)
    p05 = [p["p05"] for p in d["trajectory"]]; p95 = [p["p95"] for p in d["trajectory"]]
    p25 = [p["p25"] for p in d["trajectory"]]; p75 = [p["p75"] for p in d["trajectory"]]
    p50 = [p["p50"] for p in d["trajectory"]]
    ax.fill_between(tj_t, p05, p95, color=ACCENT, alpha=0.10, lw=0, zorder=2)
    ax.fill_between(tj_t, p25, p75, color=ACCENT, alpha=0.20, lw=0, zorder=2)
    ax.plot(tj_t, p50, color=ACCENT, lw=2.2, zorder=4)
    hist_t = [ct(p["t"]) for p in d["history"]]
    ax.plot(hist_t, [p["f"] for p in d["history"]], color=INK, lw=1.4, zorder=4)
    now_ct = ct(d["valid_utc"])
    ax.axvline(now_ct, color=FAINT, lw=0.8, ls=(0, (2, 2)), zorder=3)
    ax.text(now_ct, ax.get_ylim()[1], " now", fontsize=6.5, color=FAINT, va="top", **MONO)
    datefmt(ax)
    ax.set_ylabel("water temperature (°F)", fontsize=8)
    ax.set_title("Seven days ahead · observed (black) into the forecast cone (P5–P95)",
                 fontsize=9, loc="left", color=INK, pad=6)

    # 7-day calendar gauges
    days = d["daily"]
    ax2 = fig.add_axes([M, 0.205, 1 - 2 * M, 0.165]); ax2.axis("off")
    ax2.set_xlim(0, len(days)); ax2.set_ylim(0, 1)
    lo = min(x["p05"] for x in days); hi = max(x["p95"] for x in days)
    pad = (hi - lo) * 0.12 + 0.5
    ymap = lambda v: 0.18 + 0.62 * (v - (lo - pad)) / ((hi + pad) - (lo - pad))
    nowy = ymap(now["wtmp_f"])
    ax2.axhline(nowy, color=FAINT, lw=0.7, ls=(0, (3, 3)))
    ax2.text(len(days), nowy, f" now {now['wtmp_f']:.0f}°", fontsize=6, color=FAINT, va="center", **MONO)
    for i, day in enumerate(days):
        cx = i + 0.5
        ax2.plot([cx, cx], [ymap(day["p05"]), ymap(day["p95"])], color=AMBER, lw=3, alpha=0.55,
                 solid_capstyle="round")
        ax2.scatter([cx], [ymap(day["p50"])], s=34, color=ACCENT, zorder=3)
        ax2.text(cx, 0.95, day["label"], fontsize=8, color=INK, ha="center", va="center")
        ax2.text(cx, 0.05, f"{day['p50']:.0f}°", fontsize=9, color=INK, ha="center",
                 va="center", fontweight="bold", **MONO)
    ax2.set_title("Day by day · median dot, 90% band (amber)", fontsize=9, loc="left", pad=4)

    strip(fig, 0.10, [
        ("NOW", f"{dv['now_f']:.1f}°F", INK),
        ("TOMORROW", f"{dv['at24']:.1f}°F", INK),
        ("CHANGE", f"{dv['chg24']:+.1f}°F", ACCENT if dv["chg24"] >= 0 else AMBER),
        ("7-DAY MEDIAN", f"{dv['wk_med']:.1f}°F", INK),
        ("+168H BAND", f"±{dv['band168']:.1f}°F", INK),
        ("BAND STATE", dv["band_state"], MUTED)])
    foot(fig, 1, 3)


# ---------- page 2: the day ahead ----------
def page_day_ahead(fig, d, dv, issued_ct, valid_ct):
    header(fig, issued_ct, valid_ct, kicker="THE DAY AHEAD · NEXT 24 HOURS")
    now = d["now"]
    day = d["trajectory"][:24]
    t = [ct(p["t"]) for p in day]

    ax = fig.add_axes([M, 0.55, 1 - 2 * M, 0.30])
    p05 = [p["p05"] for p in day]; p95 = [p["p95"] for p in day]
    p25 = [p["p25"] for p in day]; p75 = [p["p75"] for p in day]
    p50 = [p["p50"] for p in day]
    ax.fill_between(t, p05, p95, color=ACCENT, alpha=0.12, lw=0)
    ax.fill_between(t, p25, p75, color=ACCENT, alpha=0.22, lw=0)
    ax.plot(t, p50, color=ACCENT, lw=2.4, marker="o", ms=2.5)
    ax.scatter([ct(d["valid_utc"])], [now["wtmp_f"]], s=26, color=INK, zorder=5)
    ax.annotate(f"{dv['hi'][0]:.1f}°", dv["hi"], textcoords="offset points", xytext=(0, 8),
                fontsize=7, color=AMBER, ha="center", **MONO)
    ax.annotate("▲", dv["hi"], textcoords="offset points", xytext=(0, 1), fontsize=6,
                color=AMBER, ha="center")
    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 3)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%-I%p"))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0); ax.grid(axis="y", color=RULE, lw=0.6)
    ax.set_ylabel("water temperature (°F)", fontsize=8)
    ax.set_title("Hour by hour · the black dot is the live buoy reading the forecast is pinned to",
                 fontsize=9, loc="left", color=INK, pad=6)

    # callout cards
    axc = fig.add_axes([M, 0.40, 0.50, 0.12]); axc.axis("off"); axc.set_xlim(0, 1); axc.set_ylim(0, 1)
    cards = [
        ("TOMORROW'S HIGH", f"{dv['hi'][0]:.1f}°F  @ {dv['hi'][1]:%-I %p}", INK),
        ("TOMORROW'S LOW", f"{dv['lo'][0]:.1f}°F  @ {dv['lo'][1]:%-I %p}", INK),
        ("CHANGE VS NOW", f"{dv['chg24']:+.1f}°F {'warmer' if dv['chg24'] >= 0 else 'cooler'}",
         ACCENT if dv["chg24"] >= 0 else AMBER),
        ("90% RANGE AT +24H", f"{dv['p05_24']:.1f} – {dv['p95_24']:.1f}°F", INK)]
    for i, (lab, val, col) in enumerate(cards):
        yy = 0.86 - i * 0.27
        axc.text(0, yy, lab, fontsize=6.5, color=MUTED, **MONO)
        axc.text(0, yy - 0.10, val, fontsize=11, color=col, fontweight="bold", **MONO)

    # wind / wave / air
    axw = fig.add_axes([M + 0.55, 0.40, 1 - 2 * M - 0.55, 0.12]); axw.axis("off")
    axw.set_xlim(0, 1); axw.set_ylim(0, 1)
    axw.text(0, 0.92, "CONDITIONS NOW", fontsize=6.5, color=MUTED, **MONO)
    for i, (lab, val) in enumerate([("Air", f"{now['atmp_f']:.1f}°F"),
                                    ("Waves", f"{now['wvht_ft']:.1f} ft"),
                                    ("Wind", f"{now['wspd_kt']:.0f} kt (gust {now['gst_kt']:.0f})")]):
        yy = 0.66 - i * 0.22
        axw.text(0, yy, lab, fontsize=9, color=MUTED)
        axw.text(1, yy, val, fontsize=9, color=INK, ha="right", fontweight="bold", **MONO)
    axw.text(0, 0.0, "North winds are what crash this shoreline.", fontsize=6.5, color=FAINT, style="italic")

    # uncertainty decomposition over next 24 h
    axu = fig.add_axes([M, 0.17, 1 - 2 * M, 0.17])
    unc = d["uncertainty"][:24]
    irr = np.array([u["irreducible"] for u in unc]); wx = np.array([u["weather"] for u in unc])
    ax_t = t
    axu.fill_between(ax_t, 0, irr, color=ACCENT, alpha=0.35, lw=0, label="irreducible (the lake)")
    axu.fill_between(ax_t, irr, irr + wx, color=AMBER, alpha=0.45, lw=0, label="weather-model spread")
    axu.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 6)))
    axu.xaxis.set_major_formatter(mdates.DateFormatter("%-I%p"))
    for s in ("top", "right"):
        axu.spines[s].set_visible(False)
    axu.tick_params(length=0); axu.set_ylabel("band half-width (°F)", fontsize=8)
    axu.legend(loc="upper left", fontsize=6.5, frameon=False)
    axu.set_title("Where the uncertainty comes from", fontsize=9, loc="left", pad=4)

    # swim comfort chip
    axs = fig.add_axes([M, 0.075, 1 - 2 * M, 0.05]); axs.axis("off"); axs.set_xlim(0, 1); axs.set_ylim(0, 1)
    axs.axhline(0.95, color=RULE, lw=0.8)
    axs.text(0, 0.62, "SWIM READ", fontsize=6.5, color=MUTED, **MONO)
    axs.text(0, 0.28, swim_word(now["wtmp_f"], now["atmp_f"], now["wspd_kt"]), fontsize=10, color=INK)
    axs.text(1, 0.62, "judgment, from water+air+wind", fontsize=6, color=FAINT, ha="right", style="italic")
    foot(fig, 2, 3)


# ---------- page 3: backtested track record ----------
def page_track_record(fig, v, issued_ct, valid_ct):
    live = v.get("n_live", 0) > 0
    tag = "TRACK RECORD · LAST 30 DAYS" if live else "BACKTESTED · LAST 30 DAYS (REPLAYED)"
    header(fig, issued_ct, valid_ct, kicker=tag)
    h = v.get("headline", {}) or {}

    rec = v.get("recent24", [])
    ax = fig.add_axes([M, 0.55, 1 - 2 * M, 0.30])
    if rec:
        t = [ct(p["valid"]) for p in rec]
        ax.fill_between(t, [p["p05"] for p in rec], [p["p95"] for p in rec], color=ACCENT,
                        alpha=0.13, lw=0)
        ax.plot(t, [p["p50"] for p in rec], color=ACCENT, lw=1.6, label="published median")
        live_t = [tt for tt, p in zip(t, rec) if p["origin"] != "hindcast"]
        live_a = [p["actual"] for p in rec if p["origin"] != "hindcast"]
        hind_t = [tt for tt, p in zip(t, rec) if p["origin"] == "hindcast"]
        hind_a = [p["actual"] for p in rec if p["origin"] == "hindcast"]
        ax.plot(hind_t, hind_a, color=INK, lw=1.0, ls=(0, (3, 2)), alpha=0.7, label="actual (replayed)")
        if live_a:
            ax.plot(live_t, live_a, color=INK, lw=1.4, label="actual (live)")
        ax.legend(loc="upper left", fontsize=6.5, frameon=False)
        datefmt(ax)
        ax.set_ylabel("water temperature (°F)", fontsize=8)
    ax.set_title("Recent +24h forecasts vs what the water actually did", fontsize=9, loc="left", pad=6)

    by = v.get("by_lead", [])
    axL = fig.add_axes([M, 0.28, 0.46, 0.18])
    if by:
        hs = [b["h"] for b in by]
        axL.plot(hs, [b["mae_persist_f"] for b in by], color=MUTED, lw=1.4, ls=(0, (4, 3)),
                 marker="o", ms=3, label="persistence")
        axL.plot(hs, [b["mae_f"] for b in by], color=ACCENT, lw=2.0, marker="o", ms=3, label="Seiche")
        axL.legend(fontsize=6.5, frameon=False, loc="upper left")
        for s in ("top", "right"):
            axL.spines[s].set_visible(False)
        axL.tick_params(length=0); axL.grid(axis="y", color=RULE, lw=0.6)
        axL.set_xlabel("lead time (h)", fontsize=7.5); axL.set_ylabel("MAE (°F)", fontsize=7.5)
    axL.set_title("Skill vs persistence", fontsize=9, loc="left", pad=4)

    axR = fig.add_axes([M + 0.54, 0.28, 1 - 2 * M - 0.54, 0.18])
    if by:
        hs = [b["h"] for b in by]
        axR.axhline(90, color=MUTED, lw=1.0, ls=(0, (4, 3)))
        axR.plot(hs, [b["cover90"] * 100 for b in by], color=ACCENT, lw=2.0, marker="o", ms=3)
        axR.set_ylim(60, 100)
        for s in ("top", "right"):
            axR.spines[s].set_visible(False)
        axR.tick_params(length=0); axR.grid(axis="y", color=RULE, lw=0.6)
        axR.set_xlabel("lead time (h)", fontsize=7.5); axR.set_ylabel("coverage (%)", fontsize=7.5)
    axR.set_title("90% band coverage", fontsize=9, loc="left", pad=4)

    axn = fig.add_axes([M, 0.10, 1 - 2 * M, 0.14]); axn.axis("off"); axn.set_xlim(0, 1); axn.set_ylim(0, 1)
    mae, skill, cov, bias = h.get("mae_f"), h.get("skill_pct"), h.get("cover90"), h.get("bias_f")
    if mae is not None:
        warm = "warm" if (bias or 0) >= 0 else "cool"
        note = (f"Over the window the typical one-day miss was {mae:.2f}°F, the 90% band held "
                f"the truth {cov*100:.0f}% of the time, and the forecast ran {abs(bias):.2f}°F {warm}. "
                f"Within a day or two trust the median to about a degree; across a week trust the "
                f"direction and the band, not the exact number.")
        axn.text(0, 0.85, note, fontsize=9, color=INK, va="top", wrap=True)
    if not live:
        axn.text(0, 0.04, "Scored on replayed history; live scoring begins as published forecasts resolve.",
                 fontsize=6.5, color=FAINT, style="italic", **MONO)
    foot(fig, 3, 3)


# ---------- main ----------
def main():
    d = load("site/data.json")
    if d is None:
        raise SystemExit("site/data.json not found; run publish.py first")
    v = load("site/verify.json", {})
    dv = derive(d)

    issued = datetime.now(timezone.utc)
    fc = lambda t: t.astimezone(CENTRAL).strftime("%b %-d  %-I:%M %p CT")
    issued_ct = fc(issued)
    valid_ct = fc(datetime.fromisoformat(d["valid_utc"]))

    BRIEFS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = issued.strftime("%Y%m%d_%H%MZ")
    out = BRIEFS_DIR / f"seiche_brief_{stamp}.pdf"

    with PdfPages(out) as pdf:
        for page in (lambda f: page_seven_day(f, d, dv, issued_ct, valid_ct),
                     lambda f: page_day_ahead(f, d, dv, issued_ct, valid_ct),
                     lambda f: page_track_record(f, v, issued_ct, valid_ct)):
            fig = plt.figure(figsize=A4)
            page(fig)
            pdf.savefig(fig); plt.close(fig)
        meta = pdf.infodict()
        meta["Title"] = f"Seiche 7-day outlook {stamp}"
        meta["Author"] = "Seiche"

    shutil.copy(out, BRIEFS_DIR / "latest.pdf")
    kept = sorted(BRIEFS_DIR.glob("seiche_brief_*.pdf"))
    for old in kept[:-BRIEFS_KEEP]:
        old.unlink()
    print(f"wrote {out} (+ latest.pdf); {len(kept[-BRIEFS_KEEP:])} briefs kept")


if __name__ == "__main__":
    main()
