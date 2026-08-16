"""
make_fig1_overview.py — DRAFT redesign of Figure 1 (four-panel method overview)
================================================================================
Nature-style overview figure, fully code-drawn:
  (a) the disposition problem   (b) the four-layer pipeline
  (c) the deferral mechanism    (d) blocked temporal validation
Okabe-Ito palette; Arial + DejaVu mathtext; 180 mm width. DRAFT ONLY.
Output: results/revision/figures/fig1_overview_draft.{png,svg}
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import (FancyBboxPatch, FancyArrowPatch, Circle,
                                Wedge, Rectangle)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "results", "revision", "figures")
os.makedirs(OUT, exist_ok=True)

BLUE, SKY, ORANGE, GREEN = "#0072B2", "#56B4E9", "#E69F00", "#009E73"
VERM, PURPLE, YELLOW, GRAY = "#D55E00", "#CC79A7", "#F0E442", "#8C8C8C"
INK = "#1A1A2E"

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 6.5,
    "axes.linewidth": 0.6,
    "svg.fonttype": "none",
    "mathtext.fontset": "dejavusans",
})

FW, FH = 7.09, 5.6
fig = plt.figure(figsize=(FW, FH), facecolor="white")


def rbox(ax, x, y, w, h, fc, ec=INK, lw=0.8, r=0.02, alpha=1.0):
    b = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                       fc=fc, ec=ec, lw=lw, alpha=alpha, mutation_aspect=1)
    ax.add_patch(b)
    return b


def arrow(ax, x1, y1, x2, y2, color=INK, lw=1.4, ms=8):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                        mutation_scale=ms, color=color, lw=lw,
                        shrinkA=1, shrinkB=1)
    ax.add_patch(a)
    return a


def person(ax, x, y, s, color):
    ax.add_patch(Circle((x, y + 0.62 * s), 0.34 * s, fc=color, ec="none"))
    ax.add_patch(Wedge((x, y - 0.75 * s), 0.72 * s, 15, 165, fc=color, ec="none"))


VALS = [[0, 2, 1], [2, 0, 2], [5, 5, 0]]
CMAP = {0: "#FFFFFF", 1: "#FFE8CC", 2: "#FFD199", 5: VERM}

# ══════════════════════════════════════════════════════════════════
# PANEL (a)
# ══════════════════════════════════════════════════════════════════
ax = fig.add_axes([0.012, 0.535, 0.295, 0.40])
ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
ax.text(0.55, 10.35, "The disposition problem", fontsize=7.5,
        fontweight="bold", color=INK)

mx, my, mw, mh = 0.5, 7.6, 3.4, 1.1
ax.add_patch(Rectangle((mx, my), mw, mh, fc="#2E6E4E", ec=INK, lw=0.7))
for i in range(6):
    ax.add_patch(Rectangle((mx + 0.22 + i * 0.53, my + 0.4), 0.4, 0.5,
                           fc="#111111", ec="none"))
for i in range(14):
    ax.add_patch(Rectangle((mx + 0.12 + i * 0.232, my - 0.16), 0.13, 0.17,
                           fc="#C9A227", ec="none"))
ax.scatter([mx + mw - 0.3], [my + mh + 0.18], marker="*", s=110, c=VERM,
           zorder=5, edgecolors="none")
ax.text(0.5, 6.95, "defective memory\nmodule, features $x$", fontsize=5.6,
        color=INK, va="top")

actions = [("Repair", SKY, 8.6), ("Return to\nvendor", YELLOW, 6.6),
           ("Scrap", VERM, 4.6)]
for label, color, yy in actions:
    rbox(ax, 6.7, yy - 0.72, 2.85, 1.44, fc=color, r=0.12, lw=0.7)
    ax.text(8.12, yy, label, ha="center", va="center", fontsize=6.2,
            fontweight="bold", color="white" if color == VERM else INK)
    arrow(ax, 4.2, 8.15, 6.6, yy, lw=1.0, ms=6)
ax.text(5.35, 9.35, "disposition\ndecision", fontsize=5.4, color=GRAY,
        ha="center", style="italic")

gx, gy, cs = 1.75, 0.55, 0.8
rows = ["Rep", "RTV", "Scr"]
for i in range(3):
    for j in range(3):
        v = VALS[i][j]
        ax.add_patch(Rectangle((gx + j * cs, gy + (2 - i) * cs), cs, cs,
                               fc=CMAP[v], ec=INK, lw=0.5))
        ax.text(gx + j * cs + cs / 2, gy + (2 - i) * cs + cs / 2, str(v),
                ha="center", va="center", fontsize=6.2,
                color="white" if v == 5 else INK,
                fontweight="bold" if v == 5 else "normal")
    ax.text(gx - 0.22, gy + (2 - i) * cs + cs / 2, rows[i], ha="right",
            va="center", fontsize=5.3, color=INK)
    ax.text(gx + i * cs + cs / 2, gy + 3 * cs + 0.14, rows[i], ha="center",
            va="bottom", fontsize=5.3, color=INK)
ax.text(gx - 1.25, gy + 1.5 * cs, "true", fontsize=5.3, rotation=90,
        va="center", ha="center", color=GRAY)
ax.text(gx + 1.5 * cs, gy + 3 * cs + 0.68, "action", fontsize=5.3,
        ha="center", color=GRAY)
ax.text(4.75, gy + 1.35,
        "asymmetric costs:\nmissing a true Scrap costs\n5× an unnecessary Scrap",
        fontsize=5.6, va="center", color=INK)

# ══════════════════════════════════════════════════════════════════
# PANEL (b)
# ══════════════════════════════════════════════════════════════════
ax = fig.add_axes([0.322, 0.535, 0.672, 0.40])
ax.set_xlim(0, 22); ax.set_ylim(0, 10); ax.axis("off")
ax.text(0.3, 10.35, "Cost-calibrated decision-support framework", fontsize=7.5,
        fontweight="bold", color=INK)

LAYERS = [
    ("Cost-sensitive\ntraining", "XGB + LGBM ensemble\nweights from cost matrix", BLUE),
    ("Probability\ncalibration", "isotonic / Platt / temp.\nauto-selected by inner CV", ORANGE),
    ("Bayes-risk\ndecision", r"$a^{*}\!=\arg\min_a \sum_c C(c,a)\,p(c|x)$", GREEN),
    ("Risk-based\ndeferral", "defer if $\\min_a R(a|x) > r_c$\ncapacity cap $B$", PURPLE),
]
bw, bh, y0 = 3.9, 5.2, 2.6
xs = [0.3, 4.75, 9.2, 13.65]
for k, ((title, sub, color), x) in enumerate(zip(LAYERS, xs)):
    rbox(ax, x, y0, bw, bh, fc="white", ec=color, lw=1.4, r=0.10)
    ax.add_patch(Rectangle((x + 0.07, y0 + bh - 1.22), bw - 0.14, 1.15,
                           fc=color, ec="none", alpha=0.16))
    ax.add_patch(Circle((x + 0.52, y0 + bh - 0.64), 0.30, fc=color, ec="none"))
    ax.text(x + 0.52, y0 + bh - 0.64, str(k + 1), ha="center", va="center",
            fontsize=6.4, fontweight="bold", color="white")
    ax.text(x + 1.0, y0 + bh - 0.64, title, ha="left", va="center",
            fontsize=5.9, fontweight="bold", color=INK, linespacing=1.05)
    ax.text(x + bw / 2, y0 + 0.66, sub, ha="center", va="center",
            fontsize=4.5 if k == 2 else 4.7, color=INK, linespacing=1.25)

# mini-visuals
bx = xs[0] + 0.95
for i, h in enumerate([0.75, 1.3, 1.85]):
    ax.add_patch(Rectangle((bx + i * 0.72, y0 + 1.45), 0.46, h, fc=BLUE,
                           ec="none", alpha=0.45 + 0.27 * i))
ax.text(xs[0] + bw / 2, y0 + 1.24, "class weight $\\propto$ cost",
        fontsize=4.5, ha="center", color=GRAY)

ox = xs[1] + 0.95
t = np.linspace(0, 1, 50)
ax.plot(ox + 2.0 * t, y0 + 1.5 + 1.75 * t, ls="--", lw=0.8, color=GRAY)
ax.plot(ox + 2.0 * t, y0 + 1.5 + 1.75 * (t ** 1.9), lw=1.2, color=GRAY, alpha=0.55)
ax.plot(ox + 2.0 * t, y0 + 1.5 + 1.75 * (t ** 1.12), lw=1.5, color=ORANGE)
ax.text(xs[1] + bw / 2, y0 + 1.24, "miscalibrated → calibrated",
        fontsize=4.5, ha="center", color=GRAY)

gx3, gy3, c3 = xs[2] + 0.72, y0 + 1.62, 0.5
for i in range(3):
    for j in range(3):
        ax.add_patch(Rectangle((gx3 + j * c3, gy3 + (2 - i) * c3), c3, c3,
                               fc=CMAP[VALS[i][j]], ec=INK, lw=0.35))
ax.text(gx3 + 3 * c3 + 0.3, gy3 + 1.5 * c3, "×", fontsize=8, va="center")
for i in range(3):
    ax.add_patch(Rectangle((gx3 + 3 * c3 + 0.66, gy3 + i * c3), c3, c3,
                           fc=GREEN, ec=INK, lw=0.35, alpha=0.25 + i * 0.25))
ax.text(xs[2] + bw / 2, y0 + 1.24, "cost matrix × calibrated probabilities",
        fontsize=4.5, ha="center", color=GRAY)

hx = xs[3] + 0.75
hh = [1.75, 1.35, 1.05, 0.8, 0.6, 0.42, 0.28, 0.18]
for i, h in enumerate(hh):
    ax.add_patch(Rectangle((hx + i * 0.31, y0 + 1.45), 0.24, h,
                           fc=PURPLE if i < 2 else GRAY, ec="none",
                           alpha=0.85 if i < 2 else 0.35))
ax.plot([hx + 2 * 0.31 - 0.035] * 2, [y0 + 1.35, y0 + 3.35], color=VERM,
        lw=1.0, ls="--")
ax.text(xs[3] + bw / 2, y0 + 1.24, "highest-risk cases deferred",
        fontsize=4.5, ha="center", color=GRAY)

for x in xs[:-1]:
    arrow(ax, x + bw + 0.04, y0 + bh / 2, x + bw + 0.52, y0 + bh / 2, lw=1.5)

rbox(ax, 0.3, 8.75, 3.4, 0.95, fc="#EFEFEF", r=0.10, lw=0.6)
ax.text(2.0, 9.22, "unit $x$  (32 features)", ha="center", va="center",
        fontsize=5.4)
arrow(ax, 2.0, 8.73, 2.0, y0 + bh + 0.06, lw=1.0, ms=6)

outx = 18.35
rbox(ax, outx, 5.9, 3.35, 1.6, fc=GREEN, r=0.12, lw=0, alpha=0.92)
ax.add_patch(Circle((outx + 0.55, 6.7), 0.33, fc="white", ec="none"))
ax.plot([outx + 0.38, outx + 0.52, outx + 0.74],
        [6.70, 6.55, 6.86], color=GREEN, lw=1.5, solid_capstyle="round")
ax.text(outx + 1.02, 6.7, "automated\naction  $a^{*}$", ha="left", va="center",
        fontsize=5.6, fontweight="bold", color="white", linespacing=1.15)
rbox(ax, outx, 3.1, 3.35, 1.6, fc=PURPLE, r=0.12, lw=0, alpha=0.92)
person(ax, outx + 0.55, 3.9, 0.33, "white")
ax.text(outx + 1.02, 3.9, "expert\nreview", ha="left", va="center",
        fontsize=5.6, fontweight="bold", color="white", linespacing=1.15)
arrow(ax, xs[3] + bw + 0.04, y0 + bh / 2 + 0.7, outx - 0.05, 6.7,
      color=GREEN, lw=1.4)
arrow(ax, xs[3] + bw + 0.04, y0 + bh / 2 - 0.7, outx - 0.05, 3.9,
      color=PURPLE, lw=1.4)
ax.text(outx + 1.67, 7.72, "confident", fontsize=4.8, color=GREEN,
        ha="center", style="italic")
ax.text(outx + 1.67, 2.82, "uncertain", fontsize=4.8, color=PURPLE,
        ha="center", style="italic")

ax.text(11.0, 1.15,
        "modular: layers 2–4 are model-agnostic post-processing — "
        "calibration + Bayes-risk alone yield a 19.4% cost reduction "
        "with no retraining",
        fontsize=5.1, ha="center", color=GRAY, style="italic")

# ══════════════════════════════════════════════════════════════════
# PANEL (c)
# ══════════════════════════════════════════════════════════════════
ax = fig.add_axes([0.075, 0.075, 0.42, 0.345])
x = np.linspace(0, 100, 400)
risk = 2.3 * np.exp(-x / 16) + 0.14 + 0.10 * np.exp(-((x - 32) / 9) ** 2)
rc = 0.5
CAP = 10.0
ax.plot(x, risk, color=INK, lw=1.3)
ax.axhline(rc, color=VERM, lw=1.0, ls="--")
ax.fill_between(x, 0, risk, where=x <= CAP, color=PURPLE, alpha=0.30, lw=0)
ax.fill_between(x, 0, risk, where=x > CAP, color=GREEN, alpha=0.14, lw=0)
ax.axvline(CAP, color=PURPLE, lw=1.0)
ax.set_xlim(0, 100); ax.set_ylim(0, 2.6)
ax.set_xlabel("cases ranked by expected risk (%)", fontsize=6)
ax.set_ylabel(r"$\min_a R(a\,|\,x)$", fontsize=6)
ax.tick_params(labelsize=5.5, width=0.6)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.text(13.5, 2.42, "deferred to expert (top-ranked, ≤ capacity $B$)",
        fontsize=5.6, color=PURPLE, va="top")
ax.text(45, 1.15, "automated Bayes-risk action", fontsize=5.6, color=GREEN)
ax.text(97, rc + 0.09, "review cost  $r_c$", fontsize=5.6, color=VERM,
        ha="right")
ax.text(CAP + 1.6, 0.09, "$B$ = 10%", fontsize=5.4, color=PURPLE)
ax.set_title("Risk-based deferral under a review budget", fontsize=7.5,
             fontweight="bold", color=INK, loc="left", pad=5)

# ══════════════════════════════════════════════════════════════════
# PANEL (d)
# ══════════════════════════════════════════════════════════════════
ax = fig.add_axes([0.545, 0.055, 0.445, 0.365])
ax.set_xlim(0, 12.6); ax.set_ylim(-0.6, 8.6); ax.axis("off")
ax.text(0.55, 8.55, "Blocked forward-chaining temporal validation",
        fontsize=7.5, fontweight="bold", color=INK)

yy = 6.35
for f in [1, 2, 3, None, 7]:
    if f is None:
        ax.scatter([6.05] * 3, [yy + 0.55, yy + 0.38, yy + 0.21], s=1.6,
                   c=GRAY, edgecolors="none")
        yy -= 1.02
        continue
    ax.text(0.42, yy + 0.32, f"fold {f}", fontsize=5.6, ha="right",
            va="center", color=INK)
    for b in range(10):
        if b < f:
            c, a = BLUE, 0.75
        elif b == f:
            c, a = ORANGE, 0.9
        elif b == f + 1:
            c, a = GREEN, 0.9
        else:
            c, a = "#DDDDDD", 1.0
        ax.add_patch(Rectangle((0.55 + b * 1.05, yy), 0.95, 0.62, fc=c,
                               alpha=a, ec="white", lw=0.4))
    yy -= 1.02
ax.annotate("", xy=(11.3, 7.45), xytext=(0.55, 7.45),
            arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=0.9))
ax.text(5.9, 7.62, "production time  (10 chronological blocks, 2011→2013)",
        fontsize=5.4, ha="center", color=GRAY)
lx = 0.55
for lab, c, a in [("train", BLUE, 0.75), ("calibrate", ORANGE, 0.9),
                  ("test (strictly later)", GREEN, 0.9)]:
    ax.add_patch(Rectangle((lx, 0.62), 0.55, 0.5, fc=c, alpha=a, ec="none"))
    ax.text(lx + 0.68, 0.87, lab, fontsize=5.5, va="center", color=INK)
    lx += 0.68 + 0.155 * len(lab) + 0.7
ax.text(0.55, -0.25,
        "7 folds × 10 seeds = 70 runs;  fold = unit of statistical inference",
        fontsize=5.5, color=INK)

for letter, (px, py) in {"a": (0.006, 0.968), "b": (0.318, 0.968),
                         "c": (0.006, 0.462), "d": (0.523, 0.462)}.items():
    fig.text(px, py, letter, fontsize=11, fontweight="bold", family="Arial")

for ext in ("png", "svg"):
    fig.savefig(os.path.join(OUT, f"fig1_overview_draft.{ext}"), dpi=600,
                bbox_inches="tight", facecolor="white")
plt.close(fig)
print("fig1_overview_draft v2 saved")
