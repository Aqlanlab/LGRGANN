"""
stats_corrected.py — Fold-clustered statistical re-analysis (revision)
======================================================================
Addresses Reviewer 2's dependence critique: seeds within a fold reuse the
same test records and are pseudo-replicates; folds share training data.
The fold is therefore the unit of inference.

For every configuration vs. the XGB_Argmax baseline, and for every external
baseline vs. the Proposed default, this script computes:
  - run-level Wilcoxon (the original analysis, kept for the transparency note)
  - fold-level paired t-test (primary), seeds averaged within fold
  - fold-level exact Wilcoxon signed-rank (robustness)
  - Nadeau-Bengio / Bouckaert-Frank corrected t (5-fold CV datasets only)
  - hierarchical (fold-then-seed) bootstrap 95% CIs
  - fold-level Cohen's d, folds-won count, Holm-adjusted p-values
  - variance decomposition (fold-level vs seed-level variance)

Also runs the unbiased leave-one-fold-out (LOFO) configuration selection
(addresses the test-set selection critique) and the pre-specified default
configuration analysis (CB_Ens_Auto_BR from run_supplement, valid pairing:
both runners use deterministic fold generators with identical settings).

Pure re-analysis: no model retraining. Outputs to results/revision/.
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy import stats

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FINAL = os.path.join(BASE, "results", "final")
SUPP = os.path.join(BASE, "results", "supplement")
OUT = os.path.join(BASE, "results", "revision")
os.makedirs(OUT, exist_ok=True)

DATASETS = {
    # name: (folder, n_folds, nb_rho or None, best_config)
    "DIMM":  ("DIMM_enhanced", 7, None, "CB_Ens_Iso_BR"),
    "Steel": ("Steel_Plates", 5, 0.25, "CS_Ens_Auto_BR"),
    "SECOM": ("SECOM", 5, 0.25, "CS_BR"),
}
BASELINE = "XGB_Argmax"
B_BOOT = 10000
RNG = np.random.RandomState(42)


def fold_means(df, col="cost_per_1000"):
    return df.groupby("fold")[col].mean()


def exact_wilcoxon(a, b):
    try:
        return stats.wilcoxon(a, b, mode="exact").pvalue
    except TypeError:
        try:
            return stats.wilcoxon(a, b, method="exact").pvalue
        except Exception:
            return stats.wilcoxon(a, b).pvalue
    except Exception:
        try:
            return stats.wilcoxon(a, b).pvalue
        except Exception:
            return np.nan


def nb_corrected_t(diff_folds, rho):
    """Nadeau-Bengio / Bouckaert-Frank corrected paired t for k-fold CV.
    Variance inflated by (1/k + rho), rho = n_test/n_train."""
    k = len(diff_folds)
    md = diff_folds.mean()
    vd = diff_folds.var(ddof=1)
    if vd == 0:
        return np.nan, np.nan
    t = md / np.sqrt((1.0 / k + rho) * vd)
    p = 2 * stats.t.sf(abs(t), df=k - 1)
    return t, p


def hier_boot_ci(bl_df, m_df, B=B_BOOT, rng=RNG):
    """Hierarchical bootstrap: resample folds, then seeds within fold.
    Returns 95% CI on the mean paired diff and on the method's mean cost."""
    folds = sorted(bl_df["fold"].unique())
    # per (fold, seed) paired diffs and method costs
    bl_p = bl_df.set_index(["fold", "seed"])["cost_per_1000"]
    m_p = m_df.set_index(["fold", "seed"])["cost_per_1000"]
    common = bl_p.index.intersection(m_p.index)
    diffs = {f: (bl_p.loc[f] - m_p.loc[f]).values for f in folds}
    costs = {f: m_p.loc[f].values for f in folds}
    nf = len(folds)
    diff_stats = np.empty(B)
    cost_stats = np.empty(B)
    for b in range(B):
        fs = rng.randint(0, nf, nf)
        dvals, cvals = [], []
        for fi in fs:
            f = folds[fi]
            d = diffs[f]
            c = costs[f]
            si = rng.randint(0, len(d), len(d))
            dvals.append(d[si].mean())
            cvals.append(c[si].mean())
        diff_stats[b] = np.mean(dvals)
        cost_stats[b] = np.mean(cvals)
    return (np.percentile(diff_stats, [2.5, 97.5]),
            np.percentile(cost_stats, [2.5, 97.5]))


def holm(pvals):
    """Holm step-down adjustment. Returns adjusted p-values (same order)."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (n - rank) * p[idx]
        running = max(running, val)
        adj[idx] = min(running, 1.0)
    return adj


def compare(bl_df, m_df, rho=None):
    """All statistics for one method vs baseline."""
    merged = bl_df.merge(m_df, on=["fold", "seed"], suffixes=("_b", "_m"))
    run_p = np.nan
    try:
        run_p = stats.wilcoxon(merged["cost_per_1000_b"],
                               merged["cost_per_1000_m"]).pvalue
    except Exception:
        pass

    bf = fold_means(bl_df)
    mf = fold_means(m_df)
    common = bf.index.intersection(mf.index)
    bf, mf = bf[common], mf[common]
    d_folds = (bf - mf).values
    n = len(d_folds)

    t_stat, t_p = stats.ttest_rel(bf, mf)
    w_p = exact_wilcoxon(bf.values, mf.values)
    sd = d_folds.std(ddof=1)
    d_eff = d_folds.mean() / sd if sd > 0 else np.nan
    nb_t, nb_p = nb_corrected_t(pd.Series(d_folds), rho) if rho else (np.nan, np.nan)
    ci_diff, ci_cost = hier_boot_ci(bl_df, m_df)

    # variance decomposition of the method's cost
    per_run = m_df.groupby(["fold", "seed"])["cost_per_1000"].mean()
    fold_var = fold_means(m_df).var(ddof=1)
    seed_var = m_df.groupby("fold")["cost_per_1000"].var(ddof=1).mean()

    sem = mf.std(ddof=1) / np.sqrt(n)
    tci = stats.t.interval(0.95, n - 1, loc=mf.mean(), scale=sem)

    return {
        "n_folds": n,
        "cost_mean": round(mf.mean(), 1),
        "baseline_mean": round(bf.mean(), 1),
        "diff_mean": round(d_folds.mean(), 1),
        "pct_reduction": round(d_folds.mean() / bf.mean() * 100, 1),
        "folds_won": int((d_folds > 0).sum()),
        "run_level_wilcoxon_p": run_p,
        "fold_t_p": t_p,
        "fold_wilcoxon_p": w_p,
        "nb_corrected_p": nb_p,
        "fold_d": round(d_eff, 2) if not np.isnan(d_eff) else np.nan,
        "cost_tCI_lo": round(tci[0], 1), "cost_tCI_hi": round(tci[1], 1),
        "diff_bootCI_lo": round(ci_diff[0], 1), "diff_bootCI_hi": round(ci_diff[1], 1),
        "cost_bootCI_lo": round(ci_cost[0], 1), "cost_bootCI_hi": round(ci_cost[1], 1),
        "fold_var": round(fold_var, 1), "mean_within_fold_seed_var": round(seed_var, 1),
    }


def lofo_selection(config_dfs, baseline_name=BASELINE):
    """Leave-one-fold-out configuration selection.
    For each fold f: choose config minimizing mean fold-level cost over folds != f;
    evaluate the chosen config on fold f."""
    fm = {name: fold_means(df) for name, df in config_dfs.items()
          if name != baseline_name}
    folds = sorted(next(iter(fm.values())).index)
    sel_rows = []
    lofo_costs = {}
    for f in folds:
        scores = {name: s.drop(f).mean() for name, s in fm.items()}
        winner = min(scores, key=scores.get)
        sel_rows.append({"fold": f, "selected": winner,
                         "cost_on_heldout_fold": round(fm[winner][f], 1)})
        lofo_costs[f] = fm[winner][f]
    lofo_series = pd.Series(lofo_costs)
    return pd.DataFrame(sel_rows), lofo_series


def lmm_analysis(bl_df, m_df, label):
    """Mixed-effects sensitivity analysis: cost ~ config + (1|fold)."""
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        return None
    a = bl_df[["fold", "seed", "cost_per_1000"]].copy(); a["cfg"] = 0
    b = m_df[["fold", "seed", "cost_per_1000"]].copy(); b["cfg"] = 1
    d = pd.concat([a, b], ignore_index=True)
    try:
        md = smf.mixedlm("cost_per_1000 ~ cfg", d, groups=d["fold"])
        r = md.fit(reml=True)
        return {"comparison": label,
                "cfg_coef": round(r.params["cfg"], 1),
                "cfg_p": r.pvalues["cfg"],
                "fold_re_var": round(float(r.cov_re.iloc[0, 0]), 1),
                "resid_var": round(r.scale, 1)}
    except Exception as e:
        return {"comparison": label, "error": str(e)[:100]}


def main():
    table1 = []
    lmm_rows = []
    for ds, (folder, k, rho, best) in DATASETS.items():
        d = os.path.join(FINAL, folder)
        config_dfs = {}
        for f in os.listdir(d):
            if f.endswith(".csv") and f[:-4] not in (
                    "deferral_sweep", "policy_comparison", "stats", "summary"):
                config_dfs[f[:-4]] = pd.read_csv(os.path.join(d, f))
        bl = config_dfs[BASELINE]

        print(f"\n{'='*70}\n{ds}: {len(config_dfs)} configs, {k} folds\n{'='*70}")

        # ── All configs vs baseline ──
        rows = []
        for name, mdf in config_dfs.items():
            if name == BASELINE:
                continue
            r = compare(bl, mdf, rho)
            r["model"] = name
            rows.append(r)
        res = pd.DataFrame(rows).set_index("model").sort_values("cost_mean")
        res["fold_t_p_holm"] = holm(res["fold_t_p"].values)
        res.to_csv(os.path.join(OUT, f"{ds}_corrected_stats.csv"))
        cols = ["cost_mean", "pct_reduction", "folds_won", "run_level_wilcoxon_p",
                "fold_t_p", "fold_wilcoxon_p", "nb_corrected_p", "fold_t_p_holm", "fold_d"]
        print(res[cols].to_string())

        # ── LOFO selection ──
        sel, lofo = lofo_selection(config_dfs)
        blf = fold_means(bl)[lofo.index]
        diff = (blf - lofo).values
        t_p = stats.ttest_rel(blf, lofo).pvalue
        w_p = exact_wilcoxon(blf.values, lofo.values)
        d_eff = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else np.nan
        sel.to_csv(os.path.join(OUT, f"{ds}_lofo_selection.csv"), index=False)
        print(f"\nLOFO selection: {sel['selected'].value_counts().to_dict()}")
        print(f"LOFO honest cost: {lofo.mean():.1f} vs baseline {blf.mean():.1f} "
              f"({diff.mean()/blf.mean()*100:+.1f}%), t p={t_p:.4f}, W p={w_p:.4f}, d={d_eff:.2f}")

        # ── Pre-specified default (ext_Proposed = CB_Ens_Auto_BR) ──
        prop_path = os.path.join(SUPP, folder, "ext_Proposed.csv")
        default_row = {}
        if os.path.exists(prop_path):
            prop = pd.read_csv(prop_path)
            r = compare(bl, prop, rho)
            print(f"\nDefault (CB_Ens_Auto_BR): cost={r['cost_mean']} "
                  f"({r['pct_reduction']:+.1f}%), fold t p={r['fold_t_p']:.4f}, "
                  f"W p={r['fold_wilcoxon_p']:.4f}, d={r['fold_d']}")
            default_row = r
            lmm_rows.append(lmm_analysis(bl, prop, f"{ds}: default vs baseline"))

        # ── External baselines vs Proposed at fold level ──
        supp_dir = os.path.join(SUPP, folder)
        if os.path.exists(prop_path):
            prop = pd.read_csv(prop_path)
            ext_rows = []
            for f in sorted(os.listdir(supp_dir)):
                if f.startswith("ext_") and f.endswith(".csv") and f != "ext_Proposed.csv":
                    ext = pd.read_csv(os.path.join(supp_dir, f))
                    r = compare(ext, prop, rho)  # 'baseline' = external method
                    r["method"] = f[4:-4]
                    ext_rows.append(r)
            if ext_rows:
                edf = pd.DataFrame(ext_rows).set_index("method").sort_values(
                    "baseline_mean")
                edf = edf.rename(columns={
                    "baseline_mean": "method_cost", "cost_mean": "proposed_cost"})
                edf.to_csv(os.path.join(OUT, f"{ds}_external_corrected.csv"))
                print(f"\nExternal baselines vs Proposed (fold level):")
                print(edf[["method_cost", "proposed_cost", "diff_mean", "folds_won",
                           "fold_t_p", "fold_wilcoxon_p", "fold_d"]].to_string())

        # ── LMM for primary comparison ──
        lmm_rows.append(lmm_analysis(bl, config_dfs[best], f"{ds}: {best} vs baseline"))

        # ── Table 1 numbers ──
        primary = compare(bl, config_dfs[best], rho)
        table1.append({
            "dataset": ds, "row": "baseline", "config": BASELINE,
            "cost": primary["baseline_mean"],
        })
        table1.append({
            "dataset": ds, "row": "best(LOFO-validated)", "config": best,
            "cost": primary["cost_mean"], "pct": primary["pct_reduction"],
            "fold_t_p": round(primary["fold_t_p"], 4),
            "fold_w_p": round(primary["fold_wilcoxon_p"], 4),
            "nb_p": round(primary["nb_corrected_p"], 4) if not np.isnan(primary["nb_corrected_p"]) else "",
            "fold_d": primary["fold_d"],
            "folds_won": f"{primary['folds_won']}/{primary['n_folds']}",
            "cost_tCI": f"[{primary['cost_tCI_lo']}, {primary['cost_tCI_hi']}]",
            "cost_bootCI": f"[{primary['cost_bootCI_lo']}, {primary['cost_bootCI_hi']}]",
            "diff_bootCI": f"[{primary['diff_bootCI_lo']}, {primary['diff_bootCI_hi']}]",
            "lofo_selected": sel["selected"].value_counts().index[0],
            "lofo_unanimity": f"{sel['selected'].value_counts().iloc[0]}/{len(sel)}",
        })
        if default_row:
            table1.append({
                "dataset": ds, "row": "default", "config": "CB_Ens_Auto_BR",
                "cost": default_row["cost_mean"], "pct": default_row["pct_reduction"],
                "fold_t_p": round(default_row["fold_t_p"], 4),
                "fold_w_p": round(default_row["fold_wilcoxon_p"], 4),
                "nb_p": round(default_row["nb_corrected_p"], 4) if not np.isnan(default_row["nb_corrected_p"]) else "",
                "fold_d": default_row["fold_d"],
                "folds_won": f"{default_row['folds_won']}/{default_row['n_folds']}",
                "cost_tCI": f"[{default_row['cost_tCI_lo']}, {default_row['cost_tCI_hi']}]",
                "cost_bootCI": f"[{default_row['cost_bootCI_lo']}, {default_row['cost_bootCI_hi']}]",
            })

    pd.DataFrame(table1).to_csv(os.path.join(OUT, "table1_corrected.csv"), index=False)
    lmm_df = pd.DataFrame([r for r in lmm_rows if r])
    if len(lmm_df):
        lmm_df.to_csv(os.path.join(OUT, "lmm_sensitivity.csv"), index=False)
        print(f"\nLMM sensitivity analysis:\n{lmm_df.to_string(index=False)}")

    print("\nDONE — outputs in results/revision/")


if __name__ == "__main__":
    main()
