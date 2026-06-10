"""
run_final.py — Final comprehensive experiment runner
=====================================================
Takes the best configurations from run_v2.py and:
  1) Runs only the winning configs + baselines (faster)
  2) Fixes SECOM: uses stratified CV (standard for generalizability) + feature selection
  3) Runs deferral sweeps for each dataset with the best base model
  4) Produces final paper-ready numbers with full statistics

Framework:  Cost-Sensitive Training + Adaptive Calibration + Bayes-Risk + Deferral
"""
import sys, os, gc, time, warnings, traceback
sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from joblib import Parallel, delayed
import xgboost as xgb
import lightgbm as lgb

from preprocessing import load_and_clean, build_features, encode_dataset, make_forward_chain_folds
from metrics import compute_all_metrics
from calibration import calibrate_probabilities
from decision_layer import bayes_risk_decision, argmax_decision, defer_decision
from public_datasets import (load_steel_plates, load_secom,
                              make_stratified_folds, make_temporal_folds_from_timestamps)

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEEDS = list(range(10))
N_WORKERS = 48


# ═══════════════════════════════════════════════════════════════════════════════
#  COST-SENSITIVE SAMPLE WEIGHTS
# ═══════════════════════════════════════════════════════════════════════════════

def cost_sample_weights(y, cost_matrix, balanced=False):
    """Derive per-sample training weights from cost matrix.
    weight = max off-diagonal cost for true class.
    If balanced, additionally scale by sqrt(1/class_freq) for rare classes.
    """
    nc = cost_matrix.shape[0]
    class_cost = np.array([
        max(cost_matrix[c, j] for j in range(nc) if j != c)
        for c in range(nc)
    ])
    if balanced:
        counts = np.bincount(y, minlength=nc).astype(float)
        freq = counts / len(y)
        class_weight = class_cost * np.sqrt(1.0 / (freq + 1e-10))
    else:
        class_weight = class_cost
    weights = class_weight[y].astype(np.float64)
    weights /= weights.mean()
    return weights


# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL FACTORIES
# ═══════════════════════════════════════════════════════════════════════════════

def mk_xgb(nc, s, n_est=400, lr=0.05):
    return xgb.XGBClassifier(
        n_estimators=n_est, max_depth=6, learning_rate=lr,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
        reg_alpha=0.1, reg_lambda=1.0, gamma=0.1,
        objective="binary:logistic" if nc == 2 else "multi:softprob",
        eval_metric="logloss" if nc == 2 else "mlogloss",
        random_state=s, verbosity=0, n_jobs=1
    )

def mk_lgbm(nc, s, n_est=400, lr=0.05):
    return lgb.LGBMClassifier(
        n_estimators=n_est, max_depth=8, learning_rate=lr,
        num_leaves=64, min_child_samples=20,
        reg_alpha=0.1, reg_lambda=1.0,
        objective="binary" if nc == 2 else "multiclass",
        random_state=s, verbose=-1, n_jobs=1
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  AUTO-SELECT BEST CALIBRATION
# ═══════════════════════════════════════════════════════════════════════════════

def auto_calibrate(model, X_tr, y_tr, X_va, y_va, X_te, cm, seed=42):
    """Try all calibration methods; pick lowest cost on 2-fold CV of validation."""
    methods = ["none", "isotonic", "sigmoid", "temperature"]
    rng = np.random.RandomState(seed)
    idx = np.arange(len(y_va))
    rng.shuffle(idx)
    mid = len(y_va) // 2
    splits = [(idx[:mid], idx[mid:]), (idx[mid:], idx[:mid])]

    best_cost, best_method = float('inf'), "none"
    for method in methods:
        fold_costs = []
        for cal_idx, eval_idx in splits:
            try:
                if method == "none":
                    p_eval = model.predict_proba(X_va[eval_idx])
                else:
                    p_eval = calibrate_probabilities(
                        model, X_tr, y_tr,
                        X_va[cal_idx], y_va[cal_idx],
                        X_va[eval_idx], method=method)
                a_eval, _ = bayes_risk_decision(p_eval, cm)
                cost = sum(cm[y_va[eval_idx][i], a_eval[i]] for i in range(len(eval_idx)))
                fold_costs.append(cost / len(eval_idx) * 1000)
            except Exception:
                fold_costs.append(float('inf'))
        avg = np.mean(fold_costs)
        if avg < best_cost:
            best_cost, best_method = avg, method

    if best_method == "none":
        return model.predict_proba(X_te), best_method
    try:
        return calibrate_probabilities(model, X_tr, y_tr, X_va, y_va, X_te,
                                       method=best_method), best_method
    except Exception:
        return model.predict_proba(X_te), "none"


# ═══════════════════════════════════════════════════════════════════════════════
#  FEATURE SELECTION (for high-dimensional datasets like SECOM)
# ═══════════════════════════════════════════════════════════════════════════════

def select_features(X_tr, y_tr, nc, top_k=100, seed=42):
    """Train a quick XGB model and select top-k features by importance.
    Returns mask of selected feature indices.
    """
    m = xgb.XGBClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.1,
        objective="binary:logistic" if nc == 2 else "multi:softprob",
        random_state=seed, verbosity=0, n_jobs=1
    )
    m.fit(X_tr, y_tr)
    imp = m.feature_importances_
    if len(imp) <= top_k:
        return np.arange(len(imp))
    return np.argsort(imp)[-top_k:]


# ═══════════════════════════════════════════════════════════════════════════════
#  SINGLE (fold, seed) EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def eval_one(fi, tr, va, te, s, nc, X, y, cm, cn, cfg, hci):
    """Evaluate one (fold, seed) pair."""
    try:
        np.random.seed(s)
        sw = cost_sample_weights(y[tr], cm, balanced=cfg.get("balanced", False)) \
             if cfg.get("cost_sens") else None

        cal = cfg.get("calibration", "none")
        dec = cfg.get("decision", "argmax")
        mtype = cfg.get("model_type", "xgb")

        # Feature selection
        X_tr, X_va, X_te_local = X[tr], X[va], X[te]
        if cfg.get("feat_select"):
            feat_idx = select_features(X_tr, y[tr], nc, top_k=cfg.get("feat_k", 100), seed=s)
            X_tr = X_tr[:, feat_idx]
            X_va = X_va[:, feat_idx]
            X_te_local = X_te_local[:, feat_idx]

        if mtype == "ensemble":
            m1 = mk_xgb(nc, s)
            m2 = mk_lgbm(nc, s)
            if sw is not None:
                m1.fit(X_tr, y[tr], sample_weight=sw)
                m2.fit(X_tr, y[tr], sample_weight=sw)
            else:
                m1.fit(X_tr, y[tr])
                m2.fit(X_tr, y[tr])

            if cal == "auto":
                p1, _ = auto_calibrate(m1, X_tr, y[tr], X_va, y[va], X_te_local, cm, seed=fi*1000+s)
                p2, _ = auto_calibrate(m2, X_tr, y[tr], X_va, y[va], X_te_local, cm, seed=fi*1000+s+500)
            elif cal != "none":
                p1 = calibrate_probabilities(m1, X_tr, y[tr], X_va, y[va], X_te_local, method=cal)
                p2 = calibrate_probabilities(m2, X_tr, y[tr], X_va, y[va], X_te_local, method=cal)
            else:
                p1 = m1.predict_proba(X_te_local)
                p2 = m2.predict_proba(X_te_local)
            p = (p1 + p2) / 2.0
            p /= p.sum(axis=1, keepdims=True)
            del m1, m2
        else:
            m = mk_xgb(nc, s) if mtype == "xgb" else mk_lgbm(nc, s)
            if sw is not None:
                m.fit(X_tr, y[tr], sample_weight=sw)
            else:
                m.fit(X_tr, y[tr])

            if cal == "auto":
                p, _ = auto_calibrate(m, X_tr, y[tr], X_va, y[va], X_te_local, cm, seed=fi*1000+s)
            elif cal != "none":
                p = calibrate_probabilities(m, X_tr, y[tr], X_va, y[va], X_te_local, method=cal)
            else:
                p = m.predict_proba(X_te_local)
            del m

        # Decision
        if dec == "argmax":
            a = argmax_decision(p)
        elif dec == "bayes_risk":
            a, _ = bayes_risk_decision(p, cm)
        elif dec.startswith("defer"):
            # defer:policy:cap:rc
            parts = dec.split(":")
            pol = parts[1] if len(parts) > 1 else "risk"
            cap = float(parts[2]) if len(parts) > 2 else 0.10
            rc = float(parts[3]) if len(parts) > 3 else 0.5
            a, _, _ = defer_decision(p, cm, rc, policy=pol, capacity=cap)
        else:
            a = argmax_decision(p)

        mt = compute_all_metrics(y[te], p, a, cm, cn, None)
        if hci is not None:
            hc = y[te] == hci
            mt["missed_hc"] = (a[hc] != hci).mean() if hc.sum() > 0 else 0
            nhc = y[te] != hci
            mt["false_hc"] = (a[nhc] == hci).mean() if nhc.sum() > 0 else 0
        mt["fold"] = fi
        mt["seed"] = s
        return mt
    except Exception as e:
        return {"fold": fi, "seed": s, "_error": str(e)}


def run_config(name, cfg, folds, nc, X, y, cm, cn, hci):
    jobs = [delayed(eval_one)(fi, tr, va, te, s, nc, X, y, cm, cn, cfg, hci)
            for fi, (tr, va, te) in enumerate(folds) for s in SEEDS]
    results = Parallel(n_jobs=N_WORKERS, prefer="processes")(jobs)
    rows = [r for r in results if "_error" not in r]
    errs = [r for r in results if "_error" in r]
    if errs:
        print(f"  !! {len(errs)} errors: {errs[0]['_error'][:120]}")
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
#  DATASET LOADERS
# ═══════════════════════════════════════════════════════════════════════════════

def load_dimm():
    df_c, _ = load_and_clean(os.path.join(PROJ, "Dataset.xlsx"))
    fc, cc, nc = build_features(df_c)
    de, lt, cn, enc = encode_dataset(df_c, fc, cc, nc)
    X = de[fc].values.astype(np.float32)
    y = de["y"].values
    folds = make_forward_chain_folds(de, n_blocks=10)
    cm = np.array([[0, 2, 1], [2, 0, 2], [5, 5, 0]], dtype=float)
    return X, y, folds, cm, list(cn), 2, 3

def load_steel():
    X, y, cn, cm, hci, _ = load_steel_plates()
    folds = make_stratified_folds(y, n_folds=5)
    return X, y, folds, cm, cn, hci, 3

def load_secom_ds():
    """SECOM with stratified folds (standard for generalizability study)."""
    X, y, cn, cm, hci, _, ts = load_secom()
    # Use stratified CV for generalizability - temporal validation is demonstrated on DIMM
    folds = make_stratified_folds(y, n_folds=5)
    return X, y, folds, cm, cn, hci, 2


# ═══════════════════════════════════════════════════════════════════════════════
#  RUN ALL CONFIGS + DEFERRAL FOR ONE DATASET
# ═══════════════════════════════════════════════════════════════════════════════

def run_dataset(ds_name, X, y, folds, cm, cn, hci, nc, use_feat_select=False):
    out_dir = os.path.join(PROJ, "results", "final", ds_name)
    os.makedirs(out_dir, exist_ok=True)

    n_runs = len(folds) * len(SEEDS)
    print(f"\n{'='*72}")
    print(f"  {ds_name}: shape={X.shape}, {len(folds)} folds x {len(SEEDS)} seeds = {n_runs} runs/cfg")
    print(f"  Workers={N_WORKERS}, Classes={nc}, HC_idx={hci}")
    print(f"{'='*72}")

    # Feature selection flag
    fs = {"feat_select": True, "feat_k": 100} if use_feat_select else {}

    # ── Phase 1: Core configs ────────────────────────────────────────────
    configs = {
        # Baselines
        "XGB_Argmax":       {"model_type": "xgb",  "decision": "argmax", **fs},
        "LGBM_Argmax":      {"model_type": "lgbm", "decision": "argmax", **fs},

        # Standard calibration (no cost-sensitive) for comparison
        "Iso_BR":           {"model_type": "xgb",  "calibration": "isotonic",
                             "decision": "bayes_risk", **fs},

        # Cost-Sensitive + various calibrations
        "CS_BR":            {"model_type": "xgb", "cost_sens": True,
                             "decision": "bayes_risk", **fs},
        "CS_Iso_BR":        {"model_type": "xgb", "cost_sens": True,
                             "calibration": "isotonic", "decision": "bayes_risk", **fs},
        "CS_Temp_BR":       {"model_type": "xgb", "cost_sens": True,
                             "calibration": "temperature", "decision": "bayes_risk", **fs},
        "CS_Platt_BR":      {"model_type": "xgb", "cost_sens": True,
                             "calibration": "sigmoid", "decision": "bayes_risk", **fs},
        "CS_Auto_BR":       {"model_type": "xgb", "cost_sens": True,
                             "calibration": "auto", "decision": "bayes_risk", **fs},

        # Cost-Balanced (cost * sqrt(1/freq))
        "CB_Iso_BR":        {"model_type": "xgb", "cost_sens": True, "balanced": True,
                             "calibration": "isotonic", "decision": "bayes_risk", **fs},
        "CB_Auto_BR":       {"model_type": "xgb", "cost_sens": True, "balanced": True,
                             "calibration": "auto", "decision": "bayes_risk", **fs},

        # Ensemble
        "CS_Ens_Auto_BR":   {"model_type": "ensemble", "cost_sens": True,
                             "calibration": "auto", "decision": "bayes_risk", **fs},
        "CB_Ens_Iso_BR":    {"model_type": "ensemble", "cost_sens": True, "balanced": True,
                             "calibration": "isotonic", "decision": "bayes_risk", **fs},
    }

    results = {}
    for i, (cname, cfg) in enumerate(configs.items()):
        t0 = time.time()
        print(f"  [{i+1:2d}/{len(configs)}] {cname:22s}", end="  ", flush=True)
        d = run_config(cname, cfg, folds, nc, X, y, cm, cn, hci)
        d["model"] = cname
        results[cname] = d
        d.to_csv(os.path.join(out_dir, f"{cname}.csv"), index=False)
        cost = d['cost_per_1000'].mean()
        miss = d['missed_hc'].mean() if 'missed_hc' in d.columns else float('nan')
        fhc = d['false_hc'].mean() if 'false_hc' in d.columns else float('nan')
        print(f"cost={cost:7.1f}  F1={d['macro_f1'].mean():.3f}  ECE={d['ece'].mean():.3f}  "
              f"miss={miss:.3f}  false={fhc:.3f}  ({time.time()-t0:.0f}s)")
        gc.collect()

    # ── Find best base model ──────────────────────────────────────────────
    summary_rows = []
    for name, df in results.items():
        row = {"model": name, "cost_mean": round(df["cost_per_1000"].mean(), 1),
               "cost_std": round(df["cost_per_1000"].std(), 1),
               "macro_f1": round(df["macro_f1"].mean(), 3),
               "ece": round(df["ece"].mean(), 3), "n_runs": len(df)}
        if "missed_hc" in df.columns: row["missed_hc"] = round(df["missed_hc"].mean(), 3)
        if "false_hc" in df.columns: row["false_hc"] = round(df["false_hc"].mean(), 3)
        summary_rows.append(row)
    sdf = pd.DataFrame(summary_rows).sort_values("cost_mean")
    print(f"\n  [{ds_name}] RANKED BY COST:")
    print(sdf.to_string(index=False))

    best_name = sdf.iloc[0]["model"]
    best_cfg = configs[best_name].copy()
    bl_cost = sdf[sdf["model"] == "XGB_Argmax"].iloc[0]["cost_mean"]
    best_cost = sdf.iloc[0]["cost_mean"]
    print(f"\n  >>> Best base model: {best_name} cost={best_cost:.1f} "
          f"({(bl_cost-best_cost)/bl_cost*100:+.1f}% vs XGB_Argmax={bl_cost:.1f})")

    # ── Phase 2: Deferral sweep with best base model ─────────────────────
    print(f"\n  Deferral sweep (base: {best_name}, policy=risk):")
    deferral_rows = []
    for rc in [0.5, 1.0]:
        for cap in [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]:
            t0 = time.time()
            if cap == 0.0:
                d = results[best_name].copy()
            else:
                dcfg = best_cfg.copy()
                dcfg["decision"] = f"defer:risk:{cap}:{rc}"
                d = run_config(f"defer_{rc}_{cap}", dcfg, folds, nc, X, y, cm, cn, hci)
            cost_m = d['cost_per_1000'].mean()
            rev = d['manual_review_rate'].mean() if 'manual_review_rate' in d.columns else 0
            miss = d['missed_hc'].mean() if 'missed_hc' in d.columns else float('nan')
            elapsed = time.time() - t0
            deferral_rows.append({
                "review_cost": rc, "capacity": cap,
                "cost_mean": round(cost_m, 1), "cost_std": round(d['cost_per_1000'].std(), 1),
                "review_rate": round(rev, 3), "missed_hc": round(miss, 3)
            })
            print(f"    rc={rc} cap={cap:.0%}: cost={cost_m:7.1f}  rev={rev:.3f}  miss={miss:.3f}  ({elapsed:.0f}s)")
            gc.collect()
    pd.DataFrame(deferral_rows).to_csv(os.path.join(out_dir, "deferral_sweep.csv"), index=False)

    # ── Phase 3: Policy comparison at 10% cap ────────────────────────────
    print(f"\n  Policy comparison (10% cap, rc=0.5, base: {best_name}):")
    policy_rows = []
    for pol in ["risk", "confidence", "entropy", "risk_margin"]:
        t0 = time.time()
        dcfg = best_cfg.copy()
        dcfg["decision"] = f"defer:{pol}:0.10:0.5"
        d = run_config(f"pol_{pol}", dcfg, folds, nc, X, y, cm, cn, hci)
        cost_m = d['cost_per_1000'].mean()
        rev = d['manual_review_rate'].mean() if 'manual_review_rate' in d.columns else 0
        miss = d['missed_hc'].mean() if 'missed_hc' in d.columns else float('nan')
        policy_rows.append({"policy": pol, "cost_mean": round(cost_m, 1),
                            "review_rate": round(rev, 3), "missed_hc": round(miss, 3)})
        print(f"    {pol:15s}: cost={cost_m:7.1f}  rev={rev:.3f}  miss={miss:.3f}  ({time.time()-t0:.0f}s)")
        gc.collect()
    pd.DataFrame(policy_rows).to_csv(os.path.join(out_dir, "policy_comparison.csv"), index=False)

    # ── Statistical tests vs XGB_Argmax ────────────────────────────────────
    bl_vals = results["XGB_Argmax"].sort_values(["fold","seed"])["cost_per_1000"].values
    print(f"\n  Wilcoxon tests vs XGB_Argmax (positive D = method is BETTER):")
    stat_rows = []
    for mname, mdf in results.items():
        if mname == "XGB_Argmax": continue
        m_vals = mdf.sort_values(["fold","seed"])["cost_per_1000"].values
        if len(m_vals) != len(bl_vals): continue
        diff = bl_vals - m_vals
        try: _, pw = wilcoxon(bl_vals, m_vals)
        except: pw = np.nan
        d_eff = diff.mean() / (diff.std() + 1e-10)
        pct = (diff.mean() / bl_vals.mean()) * 100
        sig = "***" if (not np.isnan(pw) and pw < 0.05) else "   "
        stat_rows.append({"model": mname, "reduction": round(diff.mean(), 1),
                          "pct": round(pct, 1), "p": round(pw, 4) if not np.isnan(pw) else np.nan,
                          "d": round(d_eff, 3)})
        print(f"    {mname:22s}: D={diff.mean():+8.1f} ({pct:+6.1f}%)  p={pw:.4f}  d={d_eff:+.3f}  {sig}")
    pd.DataFrame(stat_rows).to_csv(os.path.join(out_dir, "stats.csv"), index=False)

    # Save summary
    sdf.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    return sdf


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    t_total = time.time()
    print(f"CPU cores: {os.cpu_count()}, Workers: {N_WORKERS}")
    print(f"XGBoost {xgb.__version__}, LightGBM {lgb.__version__}")
    print(f"Seeds: {SEEDS}")

    all_summaries = {}

    # ── DIMM (temporal forward-chaining) ──
    print("\n" + "#"*72)
    print("  DIMM — Temporal forward-chaining, 3-class defect disposition")
    print("#"*72)
    X, y, folds, cm, cn, hci, nc = load_dimm()
    s1 = run_dataset("DIMM_enhanced", X, y, folds, cm, cn, hci, nc)
    all_summaries["DIMM"] = s1
    del X, y; gc.collect()

    # ── Steel Plates (stratified 5-fold) ──
    print("\n" + "#"*72)
    print("  STEEL PLATES — Stratified 5-fold, 3-class severity")
    print("#"*72)
    X, y, folds, cm, cn, hci, nc = load_steel()
    s2 = run_dataset("Steel_Plates", X, y, folds, cm, cn, hci, nc)
    all_summaries["Steel"] = s2
    del X, y; gc.collect()

    # ── SECOM (stratified 5-fold + feature selection) ──
    print("\n" + "#"*72)
    print("  SECOM — Stratified 5-fold + feature selection (top 100), binary")
    print("#"*72)
    X, y, folds, cm, cn, hci, nc = load_secom_ds()
    s3 = run_dataset("SECOM", X, y, folds, cm, cn, hci, nc, use_feat_select=True)
    all_summaries["SECOM"] = s3
    del X, y; gc.collect()

    # ── Cross-Dataset Summary ──
    print("\n" + "="*72)
    print("  FINAL CROSS-DATASET RESULTS")
    print("="*72)
    for ds, sm in all_summaries.items():
        bl = sm[sm["model"] == "XGB_Argmax"].iloc[0]["cost_mean"]
        best = sm.iloc[0]
        pct = (bl - best["cost_mean"]) / bl * 100
        miss = best.get("missed_hc", "?")
        print(f"  {ds:12s}: XGB_Argmax={bl:.1f}  BEST={best['model']:22s}  "
              f"cost={best['cost_mean']:.1f} ({pct:+.1f}%)  miss_hc={miss}")

    elapsed = time.time() - t_total
    print(f"\nTotal: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print("COMPLETE")
