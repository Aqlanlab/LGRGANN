"""
make_fig2_summary.py — Figure 2: cross-dataset cost summary (forest plot)
=========================================================================
Replaces the former Table 1. One panel per dataset; each row shows mean
expected cost per 1000 decisions with its cluster-bootstrap 95% CI
(fold-level t interval for the deferral rows), annotated with the
reduction vs. baseline and the fold-level paired-t p-value.

Data: results/revision/table1_corrected.csv and policy_matched/deferral_foldCIs.csv
Output: results/revision/figures/fig2_summary.{png,svg} (600 DPI)
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "results", "revision", "figures")
os.makedirs(OUT, exist_ok=True)

GRAY, BLUE, SKY, PURPLE = "#8C8C8C", "#0072B2", "#56B4E9", "#CC79A7"
INK = "#1A1A2E"

plt.rcParams.update({"font.family": "Arial", "font.size": 7,
                     "axes.linewidth": 0.6, "svg.fonttype": "none"})

# rows: (label, color) top to bottom
ROWS = [("Baseline\n(XGB, argmax)", GRAY),
        ("CCDS\n(optimized)", BLUE),
        ("CCDS\n(default)", SKY),
        ("CCDS + deferral\n(10% capacity)", PURPLE)]

# per dataset: [(mean, lo, hi, annot)]
DATA = {
    "DIMM": {
        "config": ["", "CB_Ens_Iso_BR (7/7)", "CB_Ens_Auto_BR", ""],
        "vals": [(543.5, 400, 701, ""),
                 (410.5, 281, 558, "−24.5%,  p = 0.047"),
                 (427.8, 322, 548, "−21.3%,  p = 0.055"),
                 (364.4, 181, 545, "−33.0%")],
        "xlim": (150, 740),
    },
    "Steel Plates": {
        "config": ["", "cross-fitted (mixed)", "CB_Ens_Auto_BR", ""],
        "vals": [(244.3, 231, 267, ""),
                 (241.3, 221, 263, "−1.2%,  p = 0.75"),
                 (237.6, 223, 254, "−2.8%,  p = 0.44"),
                 (196.0, 153, 239, "−19.8%")],
        "xlim": (140, 285),
    },
    "SECOM": {
        "config": ["", "CS_BR (5/5)", "CB_Ens_Auto_BR", ""],
        "vals": [(326.5, 317, 334, ""),
                 (311.4, 297, 326, "−4.6%,  p = 0.15"),
                 (325.2, 319, 332, "−0.4%,  p = 0.84"),
                 (294.7, 282, 309, "−9.7%")],
        "xlim": (270, 345),
    },
}

fig, axes = plt.subplots(1, 3, figsize=(7.09, 2.75), sharey=True)
ys = np.arange(len(ROWS))[::-1]  # top row first

for ax, (ds, d) in zip(axes, DATA.items()):
    for (label, color), y, (m, lo, hi, annot), cfg in zip(ROWS, ys, d["vals"], d["config"]):
        ax.plot([lo, hi], [y, y], color=color, lw=1.6, solid_capstyle="butt",
                zorder=2)
        for xcap in (lo, hi):
            ax.plot([xcap, xcap], [y - 0.09, y + 0.09], color=color, lw=1.2)
        ax.plot(m, y, "o", ms=5.5, mfc=color, mec="white", mew=0.8, zorder=3)
        ax.annotate(f"{m:.1f}", (m, y), xytext=(0, 6), textcoords="offset points",
                    ha="center", fontsize=6.2, color=INK, fontweight="bold")
        if annot:
            ax.annotate(annot, (m, y), xytext=(0, -11.5),
                        textcoords="offset points", ha="center", fontsize=5.4,
                        color=color)
        if cfg:
            ax.annotate(cfg, (d["xlim"][0] + 0.01 * (d["xlim"][1] - d["xlim"][0]),
                              y + 0.30), fontsize=4.8, color=GRAY, va="bottom",
                        ha="left", style="italic")
    # baseline reference line
    ax.axvline(d["vals"][0][0], color=GRAY, lw=0.7, ls=":", alpha=0.7, zorder=1)
    ax.set_xlim(*d["xlim"])
    ax.set_ylim(-0.65, len(ROWS) - 0.3)
    ax.set_title(ds, fontsize=8, fontweight="bold", color=INK, pad=4)
    ax.set_xlabel("expected cost per 1000 decisions", fontsize=6.5)
    ax.tick_params(labelsize=6, width=0.6)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.set_yticks(ys)
    ax.tick_params(axis="y", length=0)

axes[0].set_yticklabels([r[0] for r in ROWS], fontsize=6.5)
for ax in axes[1:]:
    ax.tick_params(labelleft=False)

fig.tight_layout(w_pad=1.2)
for ext in ("png", "svg"):
    fig.savefig(os.path.join(OUT, f"fig2_summary.{ext}"), dpi=600,
                bbox_inches="tight", facecolor="white")
plt.close(fig)
print("fig2_summary saved")
