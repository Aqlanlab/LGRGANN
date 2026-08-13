"""
analyze_cost_uncertainty.py — Monte Carlo cost-matrix uncertainty (decision layer)
==================================================================================
Addresses Reviewer 2: "How uncertainty in every cost entry affects the
conclusions". Complements run_cost_oat.py (which retrains under one-at-a-time
entry perturbations) by sampling ALL off-diagonal entries jointly and
re-scoring saved probabilities: each draw perturbs every off-diagonal entry
independently, Uniform(0.5x, 1.5x) around its base value; Bayes-risk actions
are recomputed under the perturbed matrix and both the Proposed configuration
and the XGB_Argmax baseline are evaluated under the SAME perturbed matrix.

This covers the decision- and evaluation-layer dependence on C. The
training-weight dependence is covered by the OAT retraining grid; Table 3's
small spread between weighting variants indicates it is second order.

Also post-processes the OAT grid: evaluates the argmax baseline under each
OAT-perturbed matrix (argmax decisions are matrix-independent) so each scenario's
reduction is like-for-like.

Input: results/revision/predictions/, results/revision/cost_oat/
Output: results/revision/cost_uncertainty/*.csv
"""
import os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRED = os.path.join(BASE, "results", "revision", "predictions")
OAT = os.path.join(BASE, "results", "revision", "cost_oat")
OUT = os.path.join(BASE, "results", "revision", "cost_uncertainty")
os.makedirs(OUT, exist_ok=True)

CMS = np.load(os.path.join(PRED, "cost_matrices.npz"))
N_DRAWS = 1000
RNG = np.random.RandomState(42)
DS = "DIMM_enhanced"
CONFIGS = ["CB_Ens_Auto_BR", "CB_Ens_Iso_BR"]


def load_runs(ds, config):
    d = os.path.join(PRED, ds, config)
    out = []
    for f in sorted(os.listdir(d)):
        if f.endswith(".npz"):
            z = np.load(os.path.join(d, f))
            out.append((z["probs"].astype(np.float64), z["y_true"].astype(int)))
    return out


def mc_analysis():
    base_cm = CMS[DS]
    K = base_cm.shape[0]
    off = [(r, c) for r in range(K) for c in range(K) if r != c]

    bl_runs = load_runs(DS, "XGB_Argmax")
    bl_actions = [p.argmax(axis=1) for p, _ in bl_runs]  # matrix-independent

    draws = []
    for _ in range(N_DRAWS):
        cm = base_cm.copy()
        for (r, c) in off:
            cm[r, c] = base_cm[r, c] * RNG.uniform(0.5, 1.5)
        draws.append(cm)

    rows = []
    for config in CONFIGS:
        runs = load_runs(DS, config)
        for di, cm in enumerate(draws):
            prop_cost, bl_cost = 0.0, 0.0
            n_tot = 0
            for (p, y), ba in zip(runs, bl_actions):
                a = (p @ cm).argmin(axis=1)
                prop_cost += cm[y, a].sum()
                bl_cost += cm[y, ba].sum()
                n_tot += len(y)
            prop = prop_cost / n_tot * 1000
            bl = bl_cost / n_tot * 1000
            rows.append({"config": config, "draw": di,
                         "proposed_cost": prop, "baseline_cost": bl,
                         "pct_reduction": (bl - prop) / bl * 100})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "mc_draws.csv"), index=False)

    print(f"Monte Carlo ({N_DRAWS} draws, all off-diagonals U(0.5x, 1.5x)):")
    for config in CONFIGS:
        s = df[df.config == config]["pct_reduction"]
        wins = (df[df.config == config]["proposed_cost"]
                < df[df.config == config]["baseline_cost"]).mean()
        print(f"  {config}: median reduction {s.median():.1f}%  "
              f"[5th, 95th pct: {s.quantile(0.05):.1f}%, {s.quantile(0.95):.1f}%]  "
              f"min {s.min():.1f}%  beats baseline in {wins:.1%} of draws")
    return df


def oat_postprocess():
    """Pair each OAT scenario with the argmax baseline under the same matrix."""
    if not os.path.isdir(OAT):
        print("OAT directory missing — run run_cost_oat.py first")
        return None
    base_cm = CMS[DS]
    bl_runs = load_runs(DS, "XGB_Argmax")

    ENTRIES = {"Rep_as_RTV": (0, 1), "Rep_as_Scr": (0, 2), "RTV_as_Rep": (1, 0),
               "RTV_as_Scr": (1, 2), "Scr_as_Rep": (2, 0), "Scr_as_RTV": (2, 1)}
    rows = []
    for f in sorted(os.listdir(OAT)):
        if not (f.startswith("oat_") and f.endswith(".csv")):
            continue
        df = pd.read_csv(os.path.join(OAT, f))
        entry, pct = df["entry"].iloc[0], int(df["pct"].iloc[0])
        r, c = ENTRIES[entry]
        cm = base_cm.copy()
        cm[r, c] = df["perturbed_value"].iloc[0]
        # baseline under same matrix (argmax actions fixed)
        bl_cost, n_tot = 0.0, 0
        for p, y in bl_runs:
            a = p.argmax(axis=1)
            bl_cost += cm[y, a].sum()
            n_tot += len(y)
        bl = bl_cost / n_tot * 1000
        prop = df["cost_per_1000"].mean()
        rows.append({"entry": entry, "pct": pct,
                     "perturbed_value": round(df["perturbed_value"].iloc[0], 2),
                     "proposed_cost": round(prop, 1),
                     "baseline_cost": round(bl, 1),
                     "pct_reduction": round((bl - prop) / bl * 100, 1),
                     "hc_recall": round(df["hc_recall"].mean(), 3),
                     "false_hc": round(df["false_hc"].mean(), 3),
                     "n_runs": len(df)})
    odf = pd.DataFrame(rows).sort_values(["entry", "pct"])
    odf.to_csv(os.path.join(OUT, "oat_summary.csv"), index=False)
    print("\nOAT grid (Proposed retrained vs argmax baseline, same matrix):")
    print(odf.to_string(index=False))
    print(f"\nReduction range across all OAT scenarios: "
          f"{odf['pct_reduction'].min():.1f}% to {odf['pct_reduction'].max():.1f}%")
    return odf


if __name__ == "__main__":
    mc_analysis()
    oat_postprocess()
    print("\nDONE")
