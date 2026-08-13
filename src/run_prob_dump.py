"""
run_prob_dump.py — Re-run key configurations, saving per-case probabilities
===========================================================================
Revision support run. Re-trains the best, default, and baseline configuration
on each dataset using the identical fold/seed protocol as run_final.py, and
saves per-case calibrated test probabilities and true labels per (fold, seed).

These archives enable, WITHOUT further retraining:
  - rate-matched deferral policy comparison (Reviewer 2, Fig. S5)
  - per-run deferral outcomes for fold-level CIs (Table 1 deferral rows)
  - Monte Carlo cost-matrix uncertainty analysis (Reviewer 2)

Output: results/revision/predictions/<dataset>/<config>/f<fold>_s<seed>.npz
        (keys: probs [n_test, K], y_true [n_test])
Runtime: ~10-20 min on 48 workers.
"""
import os
import sys
import time
import gc
import numpy as np
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_final import (cost_sample_weights, mk_xgb, mk_lgbm, auto_calibrate,
                       select_features, load_dimm, load_steel, load_secom_ds,
                       SEEDS, N_WORKERS, PROJ)
from calibration import calibrate_probabilities

OUT = os.path.join(PROJ, "results", "revision", "predictions")

CONFIGS = {
    "XGB_Argmax":     {"model_type": "xgb"},
    "CB_Ens_Auto_BR": {"model_type": "ensemble", "cost_sens": True,
                       "balanced": True, "calibration": "auto"},
    "CB_Ens_Iso_BR":  {"model_type": "ensemble", "cost_sens": True,
                       "balanced": True, "calibration": "isotonic"},
    "CS_Ens_Auto_BR": {"model_type": "ensemble", "cost_sens": True,
                       "calibration": "auto"},
    "CS_BR":          {"model_type": "xgb", "cost_sens": True},
}

# dataset -> configs to dump (baseline + default + dataset's best)
PLAN = {
    "DIMM_enhanced": ["XGB_Argmax", "CB_Ens_Auto_BR", "CB_Ens_Iso_BR"],
    "Steel_Plates":  ["XGB_Argmax", "CB_Ens_Auto_BR", "CS_Ens_Auto_BR"],
    "SECOM":         ["XGB_Argmax", "CB_Ens_Auto_BR", "CS_BR"],
}


def eval_probs(fi, tr, va, te, s, nc, X, y, cm, cfg, out_dir):
    """Train one (fold, seed) and save calibrated test probabilities."""
    try:
        np.random.seed(s)
        sw = cost_sample_weights(y[tr], cm, balanced=cfg.get("balanced", False)) \
            if cfg.get("cost_sens") else None
        cal = cfg.get("calibration", "none")
        mtype = cfg.get("model_type", "xgb")

        X_tr, X_va, X_te = X[tr], X[va], X[te]
        if cfg.get("feat_select"):
            fidx = select_features(X_tr, y[tr], nc, top_k=cfg.get("feat_k", 100), seed=s)
            X_tr, X_va, X_te = X_tr[:, fidx], X_va[:, fidx], X_te[:, fidx]

        if mtype == "ensemble":
            m1, m2 = mk_xgb(nc, s), mk_lgbm(nc, s)
            for m in (m1, m2):
                m.fit(X_tr, y[tr], sample_weight=sw) if sw is not None else m.fit(X_tr, y[tr])
            if cal == "auto":
                p1, _ = auto_calibrate(m1, X_tr, y[tr], X_va, y[va], X_te, cm, seed=fi * 1000 + s)
                p2, _ = auto_calibrate(m2, X_tr, y[tr], X_va, y[va], X_te, cm, seed=fi * 1000 + s + 500)
            elif cal != "none":
                p1 = calibrate_probabilities(m1, X_tr, y[tr], X_va, y[va], X_te, method=cal)
                p2 = calibrate_probabilities(m2, X_tr, y[tr], X_va, y[va], X_te, method=cal)
            else:
                p1, p2 = m1.predict_proba(X_te), m2.predict_proba(X_te)
            p = (p1 + p2) / 2.0
            p /= p.sum(axis=1, keepdims=True)
        else:
            m = mk_xgb(nc, s)
            m.fit(X_tr, y[tr], sample_weight=sw) if sw is not None else m.fit(X_tr, y[tr])
            if cal == "auto":
                p, _ = auto_calibrate(m, X_tr, y[tr], X_va, y[va], X_te, cm, seed=fi * 1000 + s)
            elif cal != "none":
                p = calibrate_probabilities(m, X_tr, y[tr], X_va, y[va], X_te, method=cal)
            else:
                p = m.predict_proba(X_te)

        np.savez_compressed(os.path.join(out_dir, f"f{fi}_s{s}.npz"),
                            probs=p.astype(np.float32), y_true=y[te].astype(np.int16))
        return "ok"
    except Exception as e:
        return f"ERR f{fi} s{s}: {str(e)[:150]}"


def main():
    t0 = time.time()
    loaders = {"DIMM_enhanced": load_dimm, "Steel_Plates": load_steel,
               "SECOM": load_secom_ds}
    for ds, cfg_names in PLAN.items():
        X, y, folds, cm, cn, hci, nc = loaders[ds]()
        fs = {"feat_select": True, "feat_k": 100} if ds == "SECOM" else {}
        for cname in cfg_names:
            cfg = {**CONFIGS[cname], **fs}
            out_dir = os.path.join(OUT, ds, cname)
            os.makedirs(out_dir, exist_ok=True)
            t1 = time.time()
            jobs = [delayed(eval_probs)(fi, tr, va, te, s, nc, X, y, cm, cfg, out_dir)
                    for fi, (tr, va, te) in enumerate(folds) for s in SEEDS]
            res = Parallel(n_jobs=N_WORKERS, prefer="processes")(jobs)
            errs = [r for r in res if r != "ok"]
            print(f"{ds}/{cname}: {len(res) - len(errs)}/{len(res)} ok "
                  f"({time.time() - t1:.0f}s)" + (f"  FIRST ERR: {errs[0]}" if errs else ""),
                  flush=True)
        del X, y
        gc.collect()
    # save cost matrices alongside for post-processing
    np.savez(os.path.join(OUT, "cost_matrices.npz"),
             DIMM_enhanced=np.array([[0, 2, 1], [2, 0, 2], [5, 5, 0]], float),
             Steel_Plates=np.array([[0, 2, 1], [3, 0, 2], [5, 3, 0]], float),
             SECOM=np.array([[0, 1], [5, 0]], float))
    print(f"TOTAL {time.time() - t0:.0f}s")
    print("PROB DUMP COMPLETE")


if __name__ == "__main__":
    main()
