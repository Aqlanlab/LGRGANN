"""
make_revision_figures.py — New supplementary figures for the revision
=====================================================================
Figure S5 (replacement): rate-matched deferral policy comparison —
  total operational cost vs. exact deferral rate, 4 policies x 3 datasets,
  fold-level 95% CI bands. Review fees identical across policies at each
  rate, so curves differ only in ranking quality.

Figure S7 (new): cost-model robustness —
  (a) tornado plot of % cost reduction under one-at-a-time +/-25%/50%
      perturbation of each DIMM cost entry (full retraining);
  (b) histogram of % cost reduction over 1,000 Monte Carlo cost matrices
      (all off-diagonal entries jointly perturbed +/-50%).

Okabe-Ito colorblind-safe palette. 600-DPI PNG + SVG.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PM = os.path.join(BASE, "results", "revision", "policy_matched")
CU = os.path.join(BASE, "results", "revision", "cost_uncertainty")
OUT = os.path.join(BASE, "results", "revision", "figures")
os.makedirs(OUT, exist_ok=True)

# Okabe-Ito
C_RISK = "#0072B2"      # blue
C_CONF = "#E69F00"      # orange
C_ENT = "#009E73"       # green
C_MARG = "#CC79A7"      # pink
POLICY_STYLE = {
    "risk": (C_RISK, "-", "o", "Risk-based"),
    "confidence": (C_CONF, "--", "s", "Confidence"),
    "entropy": (C_ENT, "-.", "^", "Entropy"),
    "risk_margin": (C_MARG, ":", "D", "Risk margin"),
}
DS_LABEL = {"DIMM_enhanced": "DIMM", "Steel_Plates": "Steel Plates", "SECOM": "SECOM"}

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
                     "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8})


def fig_s5():
    df = pd.read_csv(os.path.join(PM, "matched_rate_perrun.csv"))
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6))
    for ax, ds in zip(axes, ["DIMM_enhanced", "Steel_Plates", "SECOM"]):
        sub = df[df.dataset == ds]
        for pol, (color, ls, marker, label) in POLICY_STYLE.items():
            g = (sub[sub.policy == pol]
                 .groupby(["k", "fold"])["total_cost"].mean()
                 .reset_index())
            ks = sorted(g["k"].unique())
            means, los, his = [], [], []
            for k in ks:
                fm = g[g.k == k]["total_cost"]
                m = fm.mean()
                sem = fm.std(ddof=1) / np.sqrt(len(fm))
                lo, hi = stats.t.interval(0.95, len(fm) - 1, loc=m, scale=sem)
                means.append(m); los.append(lo); his.append(hi)
            ks_pct = [k * 100 for k in ks]
            ax.plot(ks_pct, means, ls, color=color, marker=marker,
                    markersize=3.5, label=label, lw=1.4)
            ax.fill_between(ks_pct, los, his, color=color, alpha=0.12, lw=0)
        ax.set_title(DS_LABEL[ds])
        ax.set_xlabel("Deferral rate (%, exact top-k quota)")
        ax.grid(alpha=0.25, lw=0.5)
    axes[0].set_ylabel("Total operational cost per 1000 decisions")
    axes[0].legend(frameon=False, loc="lower left")
    fig.suptitle("Rate-matched deferral policy comparison "
                 "(identical review fees at every rate; $r_c$ = 0.5, perfect expert)",
                 y=1.02, fontsize=10)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(OUT, f"fig_s5_matched_rate.{ext}"),
                    dpi=600, bbox_inches="tight")
    plt.close(fig)
    print("fig_s5_matched_rate saved")


def fig_s7():
    oat = pd.read_csv(os.path.join(CU, "oat_summary.csv"))
    mc = pd.read_csv(os.path.join(CU, "mc_draws.csv"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.8),
                                   gridspec_kw={"width_ratios": [1.15, 1]})

    # (a) tornado: per entry, min/max reduction across the 4 perturbations
    LABELS = {
        "Rep_as_RTV": "C(Repair, RTV) = 2",
        "Rep_as_Scr": "C(Repair, Scrap) = 1",
        "RTV_as_Rep": "C(RTV, Repair) = 2",
        "RTV_as_Scr": "C(RTV, Scrap) = 2",
        "Scr_as_Rep": "C(Scrap, Repair) = 5",
        "Scr_as_RTV": "C(Scrap, RTV) = 5",
    }
    g = oat.groupby("entry")["pct_reduction"].agg(["min", "max"])
    base_red = 21.3  # default config base reduction
    order = g.sort_values("max").index
    y = np.arange(len(order))
    for i, e in enumerate(order):
        ax1.barh(i, g.loc[e, "max"] - g.loc[e, "min"], left=g.loc[e, "min"],
                 color="#56B4E9", edgecolor="#0072B2", height=0.6)
    ax1.axvline(base_red, color="#D55E00", lw=1.2, ls="--",
                label=f"Base matrix ({base_red}%)")
    ax1.axvline(0, color="black", lw=0.8)
    ax1.set_yticks(y)
    ax1.set_yticklabels([LABELS[e] for e in order], fontsize=8)
    ax1.set_xlabel("Cost reduction vs. baseline (%)")
    ax1.set_title("(a) One-at-a-time entry perturbation (±25%, ±50%; retrained)")
    ax1.legend(frameon=False, loc="lower right")
    ax1.grid(axis="x", alpha=0.25, lw=0.5)

    # (b) MC histogram (default config)
    s = mc[mc.config == "CB_Ens_Auto_BR"]["pct_reduction"]
    ax2.hist(s, bins=40, color="#56B4E9", edgecolor="#0072B2", lw=0.4)
    ax2.axvline(0, color="black", lw=1.0)
    ax2.axvline(s.median(), color="#D55E00", lw=1.2, ls="--",
                label=f"Median {s.median():.1f}%")
    ax2.set_xlabel("Cost reduction vs. baseline (%)")
    ax2.set_ylabel("Monte Carlo draws")
    ax2.set_title("(b) 1,000 joint draws, all entries ±50%")
    ax2.legend(frameon=False)
    ax2.grid(axis="y", alpha=0.25, lw=0.5)
    txt = f"{(s > 0).mean():.1%} of draws favor the framework"
    ax2.annotate(txt, xy=(0.03, 0.93), xycoords="axes fraction", fontsize=8)

    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(OUT, f"fig_s7_cost_uncertainty.{ext}"),
                    dpi=600, bbox_inches="tight")
    plt.close(fig)
    print("fig_s7_cost_uncertainty saved")


if __name__ == "__main__":
    fig_s5()
    fig_s7()
    print("DONE")
