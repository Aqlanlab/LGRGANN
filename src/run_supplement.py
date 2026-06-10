"""
run_supplement.py — Supplementary experiments for reviewer robustness
=====================================================================
Addresses 5 major concerns:
  1. Stronger external baselines (RF, ExtraTrees, CatBoost, BalancedRF, EasyEnsemble, SMOTE+XGB, threshold moving)
  2. Scrap recall constrained experiments (gamma >= 30%, 40%, 50%, 60%)
  3. Deferral sensitivity analysis (r_c sweep, expert accuracy sweep)
  4. System-level deferral metrics (deferred HC count, system miss rate accounting for imperfect experts)
  5. Cost matrix sensitivity analysis (mild, base, severe, extreme)

Output: results/supplement/ with CSV files for each analysis
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
from catboost import CatBoostClassifier
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier)
from imblearn.ensemble import (BalancedRandomForestClassifier,
                                EasyEnsembleClassifier, RUSBoostClassifier)
from imblearn.over_sampling import SMOTE

from preprocessing import load_and_clean, build_features, encode_dataset, make_forward_chain_folds
from metrics import compute_all_metrics
from calibration import calibrate_probabilities
from decision_layer import bayes_risk_decision, argmax_decision, defer_decision
from public_datasets import (load_steel_plates, load_secom,
                              make_stratified_folds)

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEEDS = list(range(10))
N_WORKERS = 48


# ═══════════════════════════════════════════════════════════════════════════════
#  REUSED FROM run_final.py
# ═══════════════════════════════════════════════════════════════════════════════

def cost_sample_weights(y, cost_matrix, balanced=False):
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

def auto_calibrate(model, X_tr, y_tr, X_va, y_va, X_te, cm, seed=42):
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
                        model, X_tr, y_tr, X_va[cal_idx], y_va[cal_idx],
                        X_va[eval_idx], method=method)
                a_eval, _ = bayes_risk_decision(p_eval, cm)
                cost = sum(cm[y_va[eval_idx][i], a_eval[i]] for i in range(len(eval_idx)))
                fold_costs.append(cost / len(eval_idx) * 1000)
            except:
                fold_costs.append(float('inf'))
        if np.mean(fold_costs) < best_cost:
            best_cost, best_method = np.mean(fold_costs), method
    if best_method == "none":
        return model.predict_proba(X_te), best_method
    try:
        return calibrate_probabilities(model, X_tr, y_tr, X_va, y_va, X_te,
                                       method=best_method), best_method
    except:
        return model.predict_proba(X_te), "none"

def select_features(X_tr, y_tr, nc, top_k=100, seed=42):
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
#  DATASET LOADERS (reuse from run_final.py)
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
    X, y, cn, cm, hci, _, ts = load_secom()
    folds = make_stratified_folds(y, n_folds=5)
    return X, y, folds, cm, cn, hci, 2


# ═══════════════════════════════════════════════════════════════════════════════
#  PART 1: STRONGER EXTERNAL BASELINES
# ═══════════════════════════════════════════════════════════════════════════════

def eval_external_baseline(fi, tr, va, te, s, nc, X, y, cm, cn, hci, method, feat_select=False):
    """Evaluate a single external baseline on one (fold, seed)."""
    try:
        np.random.seed(s)
        X_tr, X_va, X_te = X[tr], X[va], X[te]
        y_tr, y_va, y_te = y[tr], y[va], y[te]

        if feat_select:
            feat_idx = select_features(X_tr, y_tr, nc, top_k=100, seed=s)
            X_tr, X_va, X_te = X_tr[:, feat_idx], X_va[:, feat_idx], X_te[:, feat_idx]

        if method == "RF":
            m = RandomForestClassifier(n_estimators=400, max_depth=12, min_samples_leaf=5,
                                        random_state=s, n_jobs=1)
            m.fit(X_tr, y_tr)
            p = m.predict_proba(X_te)
            a = np.argmax(p, axis=1)

        elif method == "ExtraTrees":
            m = ExtraTreesClassifier(n_estimators=400, max_depth=12, min_samples_leaf=5,
                                      random_state=s, n_jobs=1)
            m.fit(X_tr, y_tr)
            p = m.predict_proba(X_te)
            a = np.argmax(p, axis=1)

        elif method == "CatBoost":
            m = CatBoostClassifier(iterations=400, depth=6, learning_rate=0.05,
                                    random_seed=s, verbose=0, thread_count=1,
                                    auto_class_weights='Balanced')
            m.fit(X_tr, y_tr)
            p = m.predict_proba(X_te)
            a = np.argmax(p, axis=1)

        elif method == "BalancedRF":
            m = BalancedRandomForestClassifier(n_estimators=400, max_depth=12,
                                               min_samples_leaf=5, random_state=s, n_jobs=1)
            m.fit(X_tr, y_tr)
            p = m.predict_proba(X_te)
            a = np.argmax(p, axis=1)

        elif method == "EasyEnsemble":
            m = EasyEnsembleClassifier(n_estimators=20, random_state=s, n_jobs=1)
            m.fit(X_tr, y_tr)
            p = m.predict_proba(X_te)
            a = np.argmax(p, axis=1)

        elif method == "RUSBoost":
            m = RUSBoostClassifier(n_estimators=200, random_state=s)
            m.fit(X_tr, y_tr)
            p = m.predict_proba(X_te)
            a = np.argmax(p, axis=1)

        elif method == "SMOTE_XGB":
            sm = SMOTE(random_state=s, k_neighbors=min(5, min(np.bincount(y_tr)) - 1))
            try:
                X_res, y_res = sm.fit_resample(X_tr, y_tr)
            except:
                X_res, y_res = X_tr, y_tr
            m = mk_xgb(nc, s)
            m.fit(X_res, y_res)
            p = m.predict_proba(X_te)
            a = np.argmax(p, axis=1)

        elif method == "ThresholdMove":
            # Train standard XGB, find optimal thresholds on validation set
            m = mk_xgb(nc, s)
            m.fit(X_tr, y_tr)
            p_val = m.predict_proba(X_va)
            # Optimize: multiply each class prob by a factor, find best cost on val
            best_cost, best_factors = float('inf'), np.ones(nc)
            for _ in range(500):
                factors = np.random.dirichlet(np.ones(nc) * 2)
                factors = factors / factors.max()  # normalize so max=1
                p_adj = p_val * factors
                p_adj = p_adj / p_adj.sum(axis=1, keepdims=True)
                a_try, _ = bayes_risk_decision(p_adj, cm)
                cost = sum(cm[y_va[i], a_try[i]] for i in range(len(y_va))) / len(y_va) * 1000
                if cost < best_cost:
                    best_cost, best_factors = cost, factors
            p = m.predict_proba(X_te)
            p_adj = p * best_factors
            p_adj = p_adj / p_adj.sum(axis=1, keepdims=True)
            a, _ = bayes_risk_decision(p_adj, cm)
            p = p_adj

        elif method == "CostXGB_BR":
            # Cost-sensitive XGB with Bayes-risk (our proposed without calibration)
            sw = cost_sample_weights(y_tr, cm, balanced=False)
            m = mk_xgb(nc, s)
            m.fit(X_tr, y_tr, sample_weight=sw)
            p = m.predict_proba(X_te)
            a, _ = bayes_risk_decision(p, cm)

        elif method == "FocalXGB":
            # XGBoost with focal-loss-like behavior via high gamma
            m = xgb.XGBClassifier(
                n_estimators=400, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                reg_alpha=0.1, reg_lambda=1.0, gamma=2.0,
                objective="binary:logistic" if nc == 2 else "multi:softprob",
                eval_metric="logloss" if nc == 2 else "mlogloss",
                random_state=s, verbosity=0, n_jobs=1,
                scale_pos_weight=5.0 if nc == 2 else None
            )
            if nc > 2:
                sw = cost_sample_weights(y_tr, cm, balanced=True)
                m.fit(X_tr, y_tr, sample_weight=sw)
            else:
                m.fit(X_tr, y_tr)
            p = m.predict_proba(X_te)
            a = np.argmax(p, axis=1)

        elif method == "Proposed":
            # Our best: CB_Ens_Iso_BR (DIMM) / CS_Ens_Auto_BR (Steel) / CS_BR (SECOM)
            # Use ensemble + auto-cal as a representative single "proposed" method
            sw = cost_sample_weights(y_tr, cm, balanced=True)
            m1 = mk_xgb(nc, s)
            m2 = mk_lgbm(nc, s)
            m1.fit(X_tr, y_tr, sample_weight=sw)
            m2.fit(X_tr, y_tr, sample_weight=sw)
            p1, _ = auto_calibrate(m1, X_tr, y_tr, X_va, y_va, X_te, cm, seed=fi*1000+s)
            p2, _ = auto_calibrate(m2, X_tr, y_tr, X_va, y_va, X_te, cm, seed=fi*1000+s+500)
            p = (p1 + p2) / 2.0
            p /= p.sum(axis=1, keepdims=True)
            a, _ = bayes_risk_decision(p, cm)
            del m1, m2

        else:
            raise ValueError(f"Unknown method: {method}")

        mt = compute_all_metrics(y_te, p, a, cm, cn, None)
        if hci is not None:
            hc_mask = y_te == hci
            mt["missed_hc"] = (a[hc_mask] != hci).mean() if hc_mask.sum() > 0 else 0
            nhc_mask = y_te != hci
            mt["false_hc"] = (a[nhc_mask] == hci).mean() if nhc_mask.sum() > 0 else 0
            mt["hc_recall"] = 1.0 - mt["missed_hc"]
        mt["fold"] = fi
        mt["seed"] = s
        return mt
    except Exception as e:
        return {"fold": fi, "seed": s, "_error": f"{method}: {str(e)[:200]}"}


def run_external_baselines(ds_name, X, y, folds, cm, cn, hci, nc, feat_select=False):
    """Run all external baselines on one dataset."""
    methods = ["RF", "ExtraTrees", "CatBoost", "BalancedRF", "EasyEnsemble",
               "RUSBoost", "SMOTE_XGB", "ThresholdMove", "FocalXGB", "Proposed"]

    out_dir = os.path.join(PROJ, "results", "supplement", ds_name)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n  External baselines for {ds_name} ({len(folds)} folds x {len(SEEDS)} seeds):")
    all_results = {}

    for method in methods:
        t0 = time.time()
        print(f"    {method:18s}", end="  ", flush=True)
        jobs = [delayed(eval_external_baseline)(
                    fi, tr, va, te, s, nc, X, y, cm, cn, hci, method, feat_select)
                for fi, (tr, va, te) in enumerate(folds) for s in SEEDS]
        results = Parallel(n_jobs=N_WORKERS, prefer="processes")(jobs)
        rows = [r for r in results if "_error" not in r]
        errs = [r for r in results if "_error" in r]
        if errs:
            print(f"({len(errs)} errors) ", end="")
        if not rows:
            print(f"ALL FAILED: {errs[0]['_error'][:120] if errs else 'unknown'}")
            continue
        df = pd.DataFrame(rows)
        df["model"] = method
        all_results[method] = df
        df.to_csv(os.path.join(out_dir, f"ext_{method}.csv"), index=False)
        cost = df['cost_per_1000'].mean()
        miss = df['missed_hc'].mean() if 'missed_hc' in df.columns else float('nan')
        fhc = df['false_hc'].mean() if 'false_hc' in df.columns else float('nan')
        print(f"cost={cost:7.1f}  F1={df['macro_f1'].mean():.3f}  miss={miss:.3f}  "
              f"false={fhc:.3f}  ({time.time()-t0:.0f}s)")
        gc.collect()

    # Summary table
    summary_rows = []
    for name, df in all_results.items():
        row = {"model": name, "cost_mean": round(df["cost_per_1000"].mean(), 1),
               "cost_std": round(df["cost_per_1000"].std(), 1),
               "macro_f1": round(df["macro_f1"].mean(), 3),
               "n_runs": len(df)}
        if "missed_hc" in df.columns: row["missed_hc"] = round(df["missed_hc"].mean(), 3)
        if "false_hc" in df.columns: row["false_hc"] = round(df["false_hc"].mean(), 3)
        if "hc_recall" in df.columns: row["hc_recall"] = round(df["hc_recall"].mean(), 3)
        summary_rows.append(row)
    sdf = pd.DataFrame(summary_rows).sort_values("cost_mean")
    sdf.to_csv(os.path.join(out_dir, "external_baselines_summary.csv"), index=False)
    print(f"\n  [{ds_name}] External baselines ranked:")
    print(sdf.to_string(index=False))

    # Wilcoxon vs Proposed
    if "Proposed" in all_results:
        prop_vals = all_results["Proposed"].sort_values(["fold","seed"])["cost_per_1000"].values
        print(f"\n  Wilcoxon tests vs Proposed:")
        for mname, mdf in all_results.items():
            if mname == "Proposed": continue
            m_vals = mdf.sort_values(["fold","seed"])["cost_per_1000"].values
            if len(m_vals) != len(prop_vals): continue
            diff = m_vals - prop_vals  # positive = Proposed is better
            try: _, pw = wilcoxon(m_vals, prop_vals)
            except: pw = np.nan
            d_eff = diff.mean() / (diff.std() + 1e-10)
            sig = "***" if (not np.isnan(pw) and pw < 0.05) else "   "
            print(f"    {mname:18s}: D={diff.mean():+8.1f}  p={pw:.4f}  d={d_eff:+.3f}  {sig}")

    return all_results


# ═══════════════════════════════════════════════════════════════════════════════
#  PART 2: SCRAP RECALL CONSTRAINED EXPERIMENTS (DIMM ONLY)
# ═══════════════════════════════════════════════════════════════════════════════

def eval_constrained_recall(fi, tr, va, te, s, nc, X, y, cm, cn, hci, gamma):
    """Evaluate with recall constraint: adjust Bayes-risk cost to enforce min recall >= gamma for HC class."""
    try:
        np.random.seed(s)
        X_tr, X_va, X_te = X[tr], X[va], X[te]
        y_tr, y_va, y_te = y[tr], y[va], y[te]

        # Train cost-balanced ensemble
        sw = cost_sample_weights(y_tr, cm, balanced=True)
        m1 = mk_xgb(nc, s)
        m2 = mk_lgbm(nc, s)
        m1.fit(X_tr, y_tr, sample_weight=sw)
        m2.fit(X_tr, y_tr, sample_weight=sw)

        # Isotonic calibration
        try:
            p1 = calibrate_probabilities(m1, X_tr, y_tr, X_va, y_va, X_te, method="isotonic")
            p2 = calibrate_probabilities(m2, X_tr, y_tr, X_va, y_va, X_te, method="isotonic")
        except:
            p1 = m1.predict_proba(X_te)
            p2 = m2.predict_proba(X_te)
        p = (p1 + p2) / 2.0
        p /= p.sum(axis=1, keepdims=True)
        del m1, m2

        # Standard Bayes-risk with MODIFIED cost matrix to enforce recall
        # Increase the off-diagonal costs for the HC class row
        # Binary search for the multiplier that achieves target recall on test
        # (In practice, this would be tuned on validation; for analysis, tune on test to show achievability)

        # First, get validation predictions for threshold tuning
        sw_v = cost_sample_weights(y_tr, cm, balanced=True)
        m1v = mk_xgb(nc, s)
        m2v = mk_lgbm(nc, s)
        m1v.fit(X_tr, y_tr, sample_weight=sw_v)
        m2v.fit(X_tr, y_tr, sample_weight=sw_v)
        try:
            pv1 = calibrate_probabilities(m1v, X_tr, y_tr, X_va, y_va, X_va, method="isotonic")
            pv2 = calibrate_probabilities(m2v, X_tr, y_tr, X_va, y_va, X_va, method="isotonic")
        except:
            pv1 = m1v.predict_proba(X_va)
            pv2 = m2v.predict_proba(X_va)
        pv = (pv1 + pv2) / 2.0
        pv /= pv.sum(axis=1, keepdims=True)
        del m1v, m2v

        # Binary search for cost multiplier on validation
        best_mult, best_cost = 1.0, float('inf')
        for mult in [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0, 50.0]:
            cm_adj = cm.copy()
            for j in range(nc):
                if j != hci:
                    cm_adj[hci, j] *= mult
            a_v, _ = bayes_risk_decision(pv, cm_adj)
            hc_v = y_va == hci
            if hc_v.sum() > 0:
                recall_v = (a_v[hc_v] == hci).mean()
            else:
                recall_v = 0
            if recall_v >= gamma:
                cost_v = sum(cm[y_va[i], a_v[i]] for i in range(len(y_va))) / len(y_va) * 1000
                if cost_v < best_cost:
                    best_cost = cost_v
                    best_mult = mult

        # Apply best multiplier to test
        cm_adj = cm.copy()
        for j in range(nc):
            if j != hci:
                cm_adj[hci, j] *= best_mult
        a, _ = bayes_risk_decision(p, cm_adj)

        # Compute metrics with ORIGINAL cost matrix
        mt = compute_all_metrics(y_te, p, a, cm, cn, None)
        hc_mask = y_te == hci
        mt["missed_hc"] = (a[hc_mask] != hci).mean() if hc_mask.sum() > 0 else 0
        nhc_mask = y_te != hci
        mt["false_hc"] = (a[nhc_mask] == hci).mean() if nhc_mask.sum() > 0 else 0
        mt["hc_recall"] = 1.0 - mt["missed_hc"]
        mt["cost_multiplier"] = best_mult
        mt["gamma"] = gamma
        mt["fold"] = fi
        mt["seed"] = s
        return mt
    except Exception as e:
        return {"fold": fi, "seed": s, "_error": str(e)[:200]}


def run_recall_constraints(ds_name, X, y, folds, cm, cn, hci, nc):
    """Run recall-constrained experiments for a single dataset."""
    out_dir = os.path.join(PROJ, "results", "supplement", ds_name)
    os.makedirs(out_dir, exist_ok=True)

    gammas = [0.0, 0.30, 0.40, 0.50, 0.60]
    print(f"\n  Recall-constrained experiments ({ds_name}):")
    all_rows = []

    for gamma in gammas:
        t0 = time.time()
        print(f"    gamma={gamma:.0%}", end="  ", flush=True)
        jobs = [delayed(eval_constrained_recall)(
                    fi, tr, va, te, s, nc, X, y, cm, cn, hci, gamma)
                for fi, (tr, va, te) in enumerate(folds) for s in SEEDS]
        results = Parallel(n_jobs=N_WORKERS, prefer="processes")(jobs)
        rows = [r for r in results if "_error" not in r]
        errs = [r for r in results if "_error" in r]
        if errs: print(f"({len(errs)} errors) ", end="")
        if not rows:
            print("ALL FAILED")
            continue
        df = pd.DataFrame(rows)
        cost_m = df['cost_per_1000'].mean()
        recall = df['hc_recall'].mean() if 'hc_recall' in df.columns else float('nan')
        false_hc = df['false_hc'].mean() if 'false_hc' in df.columns else float('nan')
        mult = df['cost_multiplier'].mean() if 'cost_multiplier' in df.columns else 1.0
        print(f"cost={cost_m:7.1f}  recall={recall:.3f}  false_hc={false_hc:.3f}  mult={mult:.1f}  ({time.time()-t0:.0f}s)")
        all_rows.append({"gamma": gamma, "cost_mean": round(cost_m, 1),
                         "cost_std": round(df['cost_per_1000'].std(), 1),
                         "hc_recall": round(recall, 3), "false_hc": round(false_hc, 3),
                         "cost_multiplier": round(mult, 1), "n_runs": len(df)})
        gc.collect()

    rdf = pd.DataFrame(all_rows)
    rdf.to_csv(os.path.join(out_dir, "recall_constraints.csv"), index=False)
    print(f"\n  [{ds_name}] Recall constraint table:")
    print(rdf.to_string(index=False))
    return rdf


# ═══════════════════════════════════════════════════════════════════════════════
#  PART 3: DEFERRAL SENSITIVITY (r_c sweep + expert accuracy)
# ═══════════════════════════════════════════════════════════════════════════════

def eval_deferral_with_expert_accuracy(fi, tr, va, te, s, nc, X, y, cm, cn, hci,
                                        rc, cap, expert_acc, feat_select=False):
    """Evaluate deferral with imperfect expert accuracy.
    System cost = auto_cost + rc * n_deferred + expert_error_cost * n_expert_errors
    """
    try:
        np.random.seed(s)
        X_tr, X_va, X_te = X[tr], X[va], X[te]
        y_tr, y_va, y_te = y[tr], y[va], y[te]

        if feat_select:
            feat_idx = select_features(X_tr, y_tr, nc, top_k=100, seed=s)
            X_tr, X_va, X_te = X_tr[:, feat_idx], X_va[:, feat_idx], X_te[:, feat_idx]

        # Train cost-balanced ensemble (best DIMM model)
        sw = cost_sample_weights(y_tr, cm, balanced=True)
        m1 = mk_xgb(nc, s)
        m2 = mk_lgbm(nc, s)
        m1.fit(X_tr, y_tr, sample_weight=sw)
        m2.fit(X_tr, y_tr, sample_weight=sw)
        try:
            p1 = calibrate_probabilities(m1, X_tr, y_tr, X_va, y_va, X_te, method="isotonic")
            p2 = calibrate_probabilities(m2, X_tr, y_tr, X_va, y_va, X_te, method="isotonic")
        except:
            p1 = m1.predict_proba(X_te)
            p2 = m2.predict_proba(X_te)
        p = (p1 + p2) / 2.0
        p /= p.sum(axis=1, keepdims=True)
        del m1, m2

        # Get base Bayes-risk actions
        actions, risk = bayes_risk_decision(p, cm)
        best_risk = np.min(risk, axis=1)
        n = len(y_te)

        # Risk-based deferral
        defer_mask = best_risk > rc
        if cap < 1.0:
            max_defer = int(cap * n)
            if defer_mask.sum() > max_defer:
                scores = best_risk.copy()
                scores[~defer_mask] = -np.inf
                top_k = np.argsort(scores)[-max_defer:]
                new_defer = np.zeros(n, dtype=bool)
                new_defer[top_k] = True
                defer_mask = new_defer

        # Compute system-level metrics
        auto_mask = ~defer_mask
        n_auto = auto_mask.sum()
        n_deferred = defer_mask.sum()

        # Auto cost (from model decisions)
        auto_cost_sum = sum(cm[y_te[i], actions[i]] for i in range(n) if auto_mask[i])

        # Expert review: imperfect accuracy
        # Expert gets it right with probability expert_acc, wrong randomly otherwise
        rng = np.random.RandomState(s * 10000 + fi)
        expert_correct = rng.random(n_deferred) < expert_acc
        expert_cost_sum = 0
        deferred_hc = 0
        deferred_hc_correct = 0
        j = 0
        for i in range(n):
            if defer_mask[i]:
                if y_te[i] == hci:
                    deferred_hc += 1
                if expert_correct[j]:
                    # Expert gets it right: no cost
                    if y_te[i] == hci:
                        deferred_hc_correct += 1
                else:
                    # Expert gets it wrong: assign random wrong action
                    wrong_actions = [a for a in range(nc) if a != y_te[i]]
                    wrong_a = rng.choice(wrong_actions)
                    expert_cost_sum += cm[y_te[i], wrong_a]
                j += 1

        # Total cost
        review_cost_total = rc * n_deferred
        total_cost = (auto_cost_sum + review_cost_total + expert_cost_sum) / n * 1000

        # System-level miss rates
        # Auto HC misses
        auto_hc = (y_te[auto_mask] == hci)
        auto_hc_missed = (actions[auto_mask][auto_hc] != hci).sum() if auto_hc.sum() > 0 else 0

        # Expert HC misses (from imperfect review)
        expert_hc_missed = deferred_hc - deferred_hc_correct

        # System-level HC miss = (auto misses + expert misses) / total HC
        total_hc = (y_te == hci).sum()
        system_hc_missed = (auto_hc_missed + expert_hc_missed)
        system_miss_rate = system_hc_missed / total_hc if total_hc > 0 else 0

        # Auto-only miss rate (among non-deferred)
        auto_miss_rate = (actions[auto_mask][auto_hc] != hci).mean() if auto_hc.sum() > 0 else 0

        return {
            "fold": fi, "seed": s,
            "review_cost": rc, "capacity": cap, "expert_accuracy": expert_acc,
            "total_cost": round(total_cost, 1),
            "n_auto": n_auto, "n_deferred": n_deferred,
            "review_rate": round(n_deferred / n, 3),
            "deferred_hc": deferred_hc,
            "auto_miss_rate": round(auto_miss_rate, 3),
            "system_miss_rate": round(system_miss_rate, 3),
            "total_hc": total_hc,
            "auto_hc_missed": auto_hc_missed,
            "expert_hc_missed": expert_hc_missed,
        }
    except Exception as e:
        return {"fold": fi, "seed": s, "_error": str(e)[:200]}


def run_deferral_sensitivity(ds_name, X, y, folds, cm, cn, hci, nc, feat_select=False):
    """Run deferral sensitivity analysis."""
    out_dir = os.path.join(PROJ, "results", "supplement", ds_name)
    os.makedirs(out_dir, exist_ok=True)

    # Part A: Review cost sensitivity (rc sweep, perfect expert)
    print(f"\n  Deferral sensitivity — Review cost sweep ({ds_name}):")
    rows_rc = []
    for rc in [0.25, 0.5, 1.0, 2.0, 3.0]:
        t0 = time.time()
        print(f"    rc={rc:.2f}, cap=10%, q=1.00", end="  ", flush=True)
        jobs = [delayed(eval_deferral_with_expert_accuracy)(
                    fi, tr, va, te, s, nc, X, y, cm, cn, hci,
                    rc=rc, cap=0.10, expert_acc=1.0, feat_select=feat_select)
                for fi, (tr, va, te) in enumerate(folds) for s in SEEDS]
        results = Parallel(n_jobs=N_WORKERS, prefer="processes")(jobs)
        good = [r for r in results if "_error" not in r]
        if not good:
            print("FAILED")
            continue
        df = pd.DataFrame(good)
        cost = df['total_cost'].mean()
        rev = df['review_rate'].mean()
        smiss = df['system_miss_rate'].mean()
        rows_rc.append({"rc": rc, "cap": 0.10, "expert_acc": 1.0,
                        "cost": round(cost, 1), "review_rate": round(rev, 3),
                        "system_miss_rate": round(smiss, 3), "n_runs": len(df)})
        print(f"cost={cost:7.1f}  rev={rev:.3f}  sys_miss={smiss:.3f}  ({time.time()-t0:.0f}s)")
        gc.collect()

    # Part B: Expert accuracy sensitivity (fix rc=0.5, cap=10%)
    print(f"\n  Deferral sensitivity — Expert accuracy sweep ({ds_name}):")
    rows_q = []
    for q in [0.80, 0.90, 0.95, 1.00]:
        t0 = time.time()
        print(f"    rc=0.50, cap=10%, q={q:.2f}", end="  ", flush=True)
        jobs = [delayed(eval_deferral_with_expert_accuracy)(
                    fi, tr, va, te, s, nc, X, y, cm, cn, hci,
                    rc=0.5, cap=0.10, expert_acc=q, feat_select=feat_select)
                for fi, (tr, va, te) in enumerate(folds) for s in SEEDS]
        results = Parallel(n_jobs=N_WORKERS, prefer="processes")(jobs)
        good = [r for r in results if "_error" not in r]
        if not good:
            print("FAILED")
            continue
        df = pd.DataFrame(good)
        cost = df['total_cost'].mean()
        smiss = df['system_miss_rate'].mean()
        def_hc = df['deferred_hc'].mean()
        rows_q.append({"rc": 0.5, "cap": 0.10, "expert_acc": q,
                        "cost": round(cost, 1), "system_miss_rate": round(smiss, 3),
                        "deferred_hc": round(def_hc, 1), "n_runs": len(df)})
        print(f"cost={cost:7.1f}  sys_miss={smiss:.3f}  def_hc={def_hc:.1f}  ({time.time()-t0:.0f}s)")
        gc.collect()

    all_rows = rows_rc + rows_q
    pd.DataFrame(all_rows).to_csv(os.path.join(out_dir, "deferral_sensitivity.csv"), index=False)

    print(f"\n  [{ds_name}] Review cost sensitivity (q=1.0):")
    print(pd.DataFrame(rows_rc).to_string(index=False))
    print(f"\n  [{ds_name}] Expert accuracy sensitivity (rc=0.5):")
    print(pd.DataFrame(rows_q).to_string(index=False))
    return rows_rc, rows_q


# ═══════════════════════════════════════════════════════════════════════════════
#  PART 4: COST MATRIX SENSITIVITY
# ═══════════════════════════════════════════════════════════════════════════════

def run_cost_matrix_sensitivity(ds_name, X, y, folds, cn, hci, nc):
    """Evaluate proposed method under varying cost matrix severity."""
    out_dir = os.path.join(PROJ, "results", "supplement", ds_name)
    os.makedirs(out_dir, exist_ok=True)

    # DIMM cost matrices with varying Scrap penalty
    scenarios = {
        "mild":    np.array([[0, 2, 1], [2, 0, 2], [3, 3, 0]], dtype=float),
        "base":    np.array([[0, 2, 1], [2, 0, 2], [5, 5, 0]], dtype=float),
        "severe":  np.array([[0, 2, 1], [2, 0, 2], [7, 7, 0]], dtype=float),
        "extreme": np.array([[0, 2, 1], [2, 0, 2], [10, 10, 0]], dtype=float),
    }

    print(f"\n  Cost matrix sensitivity ({ds_name}):")
    rows = []
    for scenario, cm in scenarios.items():
        t0 = time.time()
        print(f"    {scenario:8s} (HC penalty={cm[hci,0]:.0f})", end="  ", flush=True)

        jobs = [delayed(eval_external_baseline)(
                    fi, tr, va, te, s, nc, X, y, cm, cn, hci, "Proposed")
                for fi, (tr, va, te) in enumerate(folds) for s in SEEDS]
        results = Parallel(n_jobs=N_WORKERS, prefer="processes")(jobs)
        good = [r for r in results if "_error" not in r]
        if not good:
            print("FAILED")
            continue
        df = pd.DataFrame(good)
        cost = df['cost_per_1000'].mean()
        recall = df['hc_recall'].mean() if 'hc_recall' in df.columns else float('nan')
        false_hc = df['false_hc'].mean() if 'false_hc' in df.columns else float('nan')
        rows.append({"scenario": scenario, "hc_penalty": cm[hci, 0],
                      "cost_mean": round(cost, 1), "cost_std": round(df['cost_per_1000'].std(), 1),
                      "hc_recall": round(recall, 3), "false_hc": round(false_hc, 3),
                      "n_runs": len(df)})
        print(f"cost={cost:7.1f}  recall={recall:.3f}  false={false_hc:.3f}  ({time.time()-t0:.0f}s)")
        gc.collect()

    rdf = pd.DataFrame(rows)
    rdf.to_csv(os.path.join(out_dir, "cost_sensitivity.csv"), index=False)
    print(f"\n  [{ds_name}] Cost sensitivity table:")
    print(rdf.to_string(index=False))
    return rdf


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    t_total = time.time()
    print(f"CPU cores: {os.cpu_count()}, Workers: {N_WORKERS}")
    print(f"Seeds: {SEEDS}")
    print("=" * 72)
    print("  SUPPLEMENTARY EXPERIMENTS")
    print("=" * 72)

    # ── Load all datasets ─────────────────────────────────────────────────
    print("\nLoading datasets...")
    X_d, y_d, folds_d, cm_d, cn_d, hci_d, nc_d = load_dimm()
    print(f"  DIMM: {X_d.shape}, {len(folds_d)} folds")
    X_s, y_s, folds_s, cm_s, cn_s, hci_s, nc_s = load_steel()
    print(f"  Steel: {X_s.shape}, {len(folds_s)} folds")
    X_e, y_e, folds_e, cm_e, cn_e, hci_e, nc_e = load_secom_ds()
    print(f"  SECOM: {X_e.shape}, {len(folds_e)} folds")

    # ══════════════════════════════════════════════════════════════════════
    #  PART 1: External baselines (all 3 datasets)
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "#" * 72)
    print("  PART 1: EXTERNAL BASELINES")
    print("#" * 72)

    run_external_baselines("DIMM_enhanced", X_d, y_d, folds_d, cm_d, cn_d, hci_d, nc_d)
    gc.collect()
    run_external_baselines("Steel_Plates", X_s, y_s, folds_s, cm_s, cn_s, hci_s, nc_s)
    gc.collect()
    run_external_baselines("SECOM", X_e, y_e, folds_e, cm_e, cn_e, hci_e, nc_e, feat_select=True)
    gc.collect()

    # ══════════════════════════════════════════════════════════════════════
    #  PART 2: Scrap recall constraints (DIMM only)
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "#" * 72)
    print("  PART 2: RECALL-CONSTRAINED EXPERIMENTS (DIMM)")
    print("#" * 72)

    run_recall_constraints("DIMM_enhanced", X_d, y_d, folds_d, cm_d, cn_d, hci_d, nc_d)
    gc.collect()

    # ══════════════════════════════════════════════════════════════════════
    #  PART 3: Deferral sensitivity (all 3 datasets)
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "#" * 72)
    print("  PART 3: DEFERRAL SENSITIVITY ANALYSIS")
    print("#" * 72)

    run_deferral_sensitivity("DIMM_enhanced", X_d, y_d, folds_d, cm_d, cn_d, hci_d, nc_d)
    gc.collect()
    run_deferral_sensitivity("Steel_Plates", X_s, y_s, folds_s, cm_s, cn_s, hci_s, nc_s)
    gc.collect()
    run_deferral_sensitivity("SECOM", X_e, y_e, folds_e, cm_e, cn_e, hci_e, nc_e, feat_select=True)
    gc.collect()

    # ══════════════════════════════════════════════════════════════════════
    #  PART 4: Cost matrix sensitivity (DIMM only)
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "#" * 72)
    print("  PART 4: COST MATRIX SENSITIVITY (DIMM)")
    print("#" * 72)

    run_cost_matrix_sensitivity("DIMM_enhanced", X_d, y_d, folds_d, cn_d, hci_d, nc_d)
    gc.collect()

    # ── Done ──────────────────────────────────────────────────────────────
    elapsed = time.time() - t_total
    print(f"\n{'=' * 72}")
    print(f"  ALL SUPPLEMENTARY EXPERIMENTS COMPLETE")
    print(f"  Total: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"{'=' * 72}")
