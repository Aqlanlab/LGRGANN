"""
run_cost_oat.py — One-at-a-time cost-matrix entry perturbation (DIMM)
=====================================================================
Addresses Reviewer 2: "How uncertainty in every cost entry affects the
conclusions". Each of the 6 off-diagonal DIMM cost entries is perturbed
by -50%, -25%, +25%, +50% (24 scenarios), and the Proposed configuration
(CB_Ens_Auto_BR) is fully retrained and re-evaluated under each perturbed
matrix (70 runs each: training weights, auto-calibration selection, and
Bayes-risk decisions all see the perturbed matrix).

The XGB_Argmax baseline under the same perturbed matrices is computed by
post-processing saved probabilities (its training and argmax decisions do
not depend on the matrix; see analyze_cost_uncertainty.py).

Output: results/revision/cost_oat/oat_<entry>_<pct>.csv (per-run rows)
Runtime: ~15-25 min at 40 workers.
"""
import os
import sys
import time
import gc
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_supplement as rs
from run_final import load_dimm, SEEDS, PROJ

N_WORKERS = 40
OUT = os.path.join(PROJ, "results", "revision", "cost_oat")
os.makedirs(OUT, exist_ok=True)

BASE_CM = np.array([[0, 2, 1], [2, 0, 2], [5, 5, 0]], dtype=float)
# (row, col) of the 6 off-diagonal entries, with readable labels
ENTRIES = {
    "Rep_as_RTV":  (0, 1),   # true Repair, action RTV        (base 2)
    "Rep_as_Scr":  (0, 2),   # true Repair, action Scrap      (base 1)
    "RTV_as_Rep":  (1, 0),   # true RTV, action Repair        (base 2)
    "RTV_as_Scr":  (1, 2),   # true RTV, action Scrap         (base 2)
    "Scr_as_Rep":  (2, 0),   # true Scrap, action Repair      (base 5)
    "Scr_as_RTV":  (2, 1),   # true Scrap, action RTV         (base 5)
}
PCTS = [-50, -25, 25, 50]


def main():
    t0 = time.time()
    X, y, folds, _, cn, hci, nc = load_dimm()
    print(f"DIMM loaded: {X.shape}, {len(folds)} folds x {len(SEEDS)} seeds", flush=True)

    for label, (r, c) in ENTRIES.items():
        for pct in PCTS:
            tag = f"{label}_{pct:+d}"
            out_path = os.path.join(OUT, f"oat_{tag}.csv")
            if os.path.exists(out_path):
                print(f"skip {tag} (exists)", flush=True)
                continue
            cm = BASE_CM.copy()
            cm[r, c] = BASE_CM[r, c] * (1 + pct / 100.0)
            t1 = time.time()
            jobs = [delayed(rs.eval_external_baseline)(
                        fi, tr, va, te, s, nc, X, y, cm, cn, hci, "Proposed")
                    for fi, (tr, va, te) in enumerate(folds) for s in SEEDS]
            results = Parallel(n_jobs=N_WORKERS, prefer="processes")(jobs)
            good = [x for x in results if "_error" not in x]
            df = pd.DataFrame(good)
            df["entry"] = label
            df["pct"] = pct
            df["perturbed_value"] = cm[r, c]
            df.to_csv(out_path, index=False)
            print(f"{tag}: cost={df['cost_per_1000'].mean():7.1f}  "
                  f"recall={df['hc_recall'].mean():.3f}  n={len(df)}  "
                  f"({time.time() - t1:.0f}s)", flush=True)
            gc.collect()

    print(f"TOTAL {time.time() - t0:.0f}s")
    print("OAT COMPLETE")


if __name__ == "__main__":
    main()
