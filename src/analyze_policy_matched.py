"""
analyze_policy_matched.py — Rate-matched deferral policy comparison
===================================================================
Addresses Reviewer 2: the original policy comparison let each policy's
fixed threshold determine how many cases were deferred (risk 8.8%,
confidence 5.0%, entropy 1.8%, risk-margin 6.4% on DIMM), confounding
ranking quality with review rate.

Here every policy defers EXACTLY the top ceil(k*n) cases by its own
ranking score at each budget k, so review fees are identical across
policies and differences reflect ranking quality alone.

Also produces:
  - per-run threshold-based deferral outcomes (deployment rule:
    defer if min_a R(a|x) > r_c, 10% cap) for fold-level CIs (Table 1)
  - verification of whether the capacity cap ever bound for non-risk
    policies in the original runs (decision_layer.py truncation detail)
  - sanity check: Bayes-risk cost from saved probabilities vs run_final CSVs

Input: results/revision/predictions/  (from run_prob_dump.py)
Output: results/revision/policy_matched/*.csv
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import entropy as sp_entropy

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRED = os.path.join(BASE, "results", "revision", "predictions")
OUT = os.path.join(BASE, "results", "revision", "policy_matched")
os.makedirs(OUT, exist_ok=True)

BEST = {"DIMM_enhanced": "CB_Ens_Iso_BR",
        "Steel_Plates": "CS_Ens_Auto_BR",
        "SECOM": "CS_BR"}
DEFAULT = "CB_Ens_Auto_BR"
KS = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20]
RC = 0.5
POLICIES = ["risk", "confidence", "entropy", "risk_margin"]

CMS = np.load(os.path.join(PRED, "cost_matrices.npz"))


def load_runs(ds, config):
    d = os.path.join(PRED, ds, config)
    runs = []
    for f in sorted(os.listdir(d)):
        if f.endswith(".npz"):
            z = np.load(os.path.join(d, f))
            fold, seed = f[:-4].replace("f", "").split("_s")
            runs.append((int(fold), int(seed), z["probs"], z["y_true"]))
    return runs


def policy_scores(probs, cm):
    """Higher score = deferred first, for every policy."""
    risk = probs @ cm
    best_risk = risk.min(axis=1)
    sorted_risk = np.sort(risk, axis=1)
    margin = sorted_risk[:, 1] - sorted_risk[:, 0]
    return {
        "risk": best_risk,                      # defer highest risk
        "confidence": -probs.max(axis=1),       # defer lowest confidence
        "entropy": sp_entropy(probs.T),         # defer highest entropy
        "risk_margin": -margin,                 # defer smallest margin
    }


def matched_rate_costs(probs, y, cm, rc=RC):
    """Total operational cost per 1000 at each budget k, per policy."""
    n = len(y)
    risk = probs @ cm
    actions = risk.argmin(axis=1)
    case_cost = cm[y, actions]
    scores = policy_scores(probs, cm)
    rows = {}
    for pol, sc in scores.items():
        order = np.argsort(-sc)  # descending: defer first
        for k in KS:
            nd = int(np.ceil(k * n))
            defer_idx = order[:nd]
            retained_cost = case_cost.sum() - case_cost[defer_idx].sum()
            total = (retained_cost + nd * rc) / n * 1000
            rows[(pol, k)] = total
    return rows


def threshold_deferral(probs, y, cm, rc=RC, cap=0.10):
    """Deployment rule: defer if min risk > rc, capped at cap (risk-ranked)."""
    n = len(y)
    risk = probs @ cm
    actions = risk.argmin(axis=1)
    best_risk = risk.min(axis=1)
    defer = best_risk > rc
    if defer.sum() > int(cap * n):
        thr_idx = np.argsort(-best_risk)[:int(cap * n)]
        defer = np.zeros(n, bool)
        defer[thr_idx] = True
    case_cost = cm[y, actions]
    total = (case_cost[~defer].sum() + defer.sum() * rc) / n * 1000
    misclass_only = case_cost[~defer].sum() / n * 1000
    return total, misclass_only, defer.mean()


def original_threshold_rates(probs, cm):
    """Pre-cap deferral fraction under each policy's original fixed threshold."""
    risk = probs @ cm
    best_risk = risk.min(axis=1)
    sorted_risk = np.sort(risk, axis=1)
    margin = sorted_risk[:, 1] - sorted_risk[:, 0]
    ent = sp_entropy(probs.T)
    return {
        "risk": (best_risk > 0.5).mean(),
        "confidence": (probs.max(axis=1) < 0.6).mean(),
        "entropy": (ent > 0.8).mean(),
        "risk_margin": (margin < 0.5).mean(),
    }


def fold_ci(series_by_fold):
    n = len(series_by_fold)
    m = series_by_fold.mean()
    sem = series_by_fold.std(ddof=1) / np.sqrt(n)
    lo, hi = stats.t.interval(0.95, n - 1, loc=m, scale=sem)
    return m, lo, hi


def main():
    sanity, curve_rows, test_rows, capbound_rows, table1_rows = [], [], [], [], []

    for ds, best in BEST.items():
        cm = CMS[ds]
        for config in [best, DEFAULT, "XGB_Argmax"]:
            runs = load_runs(ds, config)

            # sanity: BR cost from probs (argmax cost for XGB_Argmax)
            costs = []
            for fold, seed, p, y in runs:
                a = (p @ cm).argmin(axis=1) if config != "XGB_Argmax" else p.argmax(axis=1)
                costs.append(cm[y, a].mean() * 1000)
            sanity.append({"dataset": ds, "config": config,
                           "dump_cost": round(np.mean(costs), 1), "n_runs": len(runs)})

            # threshold-based deferral per run (for Table 1 deferral rows)
            if config != "XGB_Argmax":
                rows = []
                for fold, seed, p, y in runs:
                    total, mis, rate = threshold_deferral(p, y, cm)
                    rows.append({"fold": fold, "seed": seed, "total": total,
                                 "misclass": mis, "rate": rate})
                rdf = pd.DataFrame(rows)
                rdf.to_csv(os.path.join(OUT, f"{ds}_{config}_threshold_deferral.csv"),
                           index=False)
                fm = rdf.groupby("fold")["total"].mean()
                m, lo, hi = fold_ci(fm)
                table1_rows.append({
                    "dataset": ds, "config": config,
                    "deferral_total_mean": round(m, 1),
                    "foldCI": f"[{lo:.0f}, {hi:.0f}]",
                    "review_rate": round(rdf["rate"].mean(), 3)})

        # ── matched-rate curves for the best config ──
        runs = load_runs(ds, best)
        for fold, seed, p, y in runs:
            mr = matched_rate_costs(p, y, cm)
            for (pol, k), v in mr.items():
                curve_rows.append({"dataset": ds, "config": best, "fold": fold,
                                   "seed": seed, "policy": pol, "k": k,
                                   "total_cost": v})
            # cap-bound verification on original thresholds
            otr = original_threshold_rates(p, cm)
            capbound_rows.append({"dataset": ds, "fold": fold, "seed": seed,
                                  **{f"{k}_precap_rate": round(v, 3)
                                     for k, v in otr.items()}})

    cdf = pd.DataFrame(curve_rows)
    cdf.to_csv(os.path.join(OUT, "matched_rate_perrun.csv"), index=False)

    # fold-level aggregation + tests: risk vs each policy at each k
    for ds in BEST:
        sub = cdf[cdf.dataset == ds]
        for k in KS:
            if k == 0.0:
                continue
            at_k = sub[sub.k == k]
            fold_means = at_k.groupby(["policy", "fold"])["total_cost"].mean().unstack(0)
            for pol in POLICIES:
                if pol == "risk":
                    continue
                d = fold_means[pol] - fold_means["risk"]  # >0: risk better
                t_p = stats.ttest_rel(fold_means[pol], fold_means["risk"]).pvalue
                try:
                    w_p = stats.wilcoxon(fold_means[pol], fold_means["risk"]).pvalue
                except Exception:
                    w_p = np.nan
                test_rows.append({
                    "dataset": ds, "k": k, "policy": pol,
                    "policy_cost": round(fold_means[pol].mean(), 1),
                    "risk_cost": round(fold_means["risk"].mean(), 1),
                    "diff": round(d.mean(), 1),
                    "folds_risk_wins": int((d > 0).sum()),
                    "fold_t_p": round(t_p, 4), "fold_w_p": round(w_p, 4)})

    pd.DataFrame(test_rows).to_csv(os.path.join(OUT, "matched_rate_tests.csv"),
                                   index=False)
    pd.DataFrame(capbound_rows).to_csv(os.path.join(OUT, "capbound_check.csv"),
                                       index=False)
    pd.DataFrame(sanity).to_csv(os.path.join(OUT, "sanity_check.csv"), index=False)
    pd.DataFrame(table1_rows).to_csv(os.path.join(OUT, "deferral_foldCIs.csv"),
                                     index=False)

    print("SANITY (dump-derived costs; compare with published means):")
    print(pd.DataFrame(sanity).to_string(index=False))
    print("\nTable 1 deferral rows (threshold rule, rc=0.5, 10% cap, fold CIs):")
    print(pd.DataFrame(table1_rows).to_string(index=False))
    print("\nMatched-rate comparison at k=10% (fold level):")
    t = pd.DataFrame(test_rows)
    print(t[t.k == 0.10].to_string(index=False))
    cb = pd.DataFrame(capbound_rows)
    print("\nCap-bound check (fraction of runs where pre-cap rate > 10%):")
    for col in ["risk_precap_rate", "confidence_precap_rate",
                "entropy_precap_rate", "risk_margin_precap_rate"]:
        print(f"  {col}: {(cb[col] > 0.10).mean():.3f}")
    print("\nDONE")


if __name__ == "__main__":
    main()
