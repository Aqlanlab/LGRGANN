"""
steel_lofo_deferral.py — Steel Plates cross-fitted (LOFO) estimates
===================================================================
LOFO selection on Steel is unstable (CS_Ens_Auto_BR in folds 0,2,3;
CS_BR in folds 1,4). This script makes Table 1's Steel rows fully
consistent with the cross-fitted protocol:

1. Dumps CS_BR probabilities on Steel (the only missing archive).
2. Computes the LOFO cost estimate's hierarchical bootstrap CI
   (fold f uses the per-seed costs of the configuration selected
   without fold f).
3. Computes the LOFO deferral estimate: threshold rule (rc=0.5,
   10% cap) applied per fold to that fold's selected configuration,
   with fold-level t CI.
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_final import load_steel, SEEDS, N_WORKERS, PROJ
from run_prob_dump import eval_probs, CONFIGS

PRED = os.path.join(PROJ, "results", "revision", "predictions")
FINAL = os.path.join(PROJ, "results", "final", "Steel_Plates")
RC, CAP = 0.5, 0.10
RNG = np.random.RandomState(42)

SELECTED = {0: "CS_Ens_Auto_BR", 1: "CS_BR", 2: "CS_Ens_Auto_BR",
            3: "CS_Ens_Auto_BR", 4: "CS_BR"}

# ── 1. dump CS_BR probabilities on Steel ──
out_dir = os.path.join(PRED, "Steel_Plates", "CS_BR")
os.makedirs(out_dir, exist_ok=True)
if len([f for f in os.listdir(out_dir) if f.endswith(".npz")]) < 50:
    X, y, folds, cm, cn, hci, nc = load_steel()
    jobs = [delayed(eval_probs)(fi, tr, va, te, s, nc, X, y, cm, CONFIGS["CS_BR"], out_dir)
            for fi, (tr, va, te) in enumerate(folds) for s in SEEDS]
    res = Parallel(n_jobs=N_WORKERS, prefer="processes")(jobs)
    print(f"CS_BR Steel dump: {sum(1 for r in res if r == 'ok')}/{len(res)} ok")
else:
    print("CS_BR Steel dump already present")

CM = np.load(os.path.join(PRED, "cost_matrices.npz"))["Steel_Plates"]

# ── 2. LOFO cost CI (hierarchical bootstrap from per-run CSVs) ──
per_seed = {}   # fold -> per-seed costs of the selected config
for f, cfg in SELECTED.items():
    df = pd.read_csv(os.path.join(FINAL, f"{cfg}.csv"))
    per_seed[f] = df[df["fold"] == f].sort_values("seed")["cost_per_1000"].values
bl = pd.read_csv(os.path.join(FINAL, "XGB_Argmax.csv"))
bl_seed = {f: bl[bl["fold"] == f].sort_values("seed")["cost_per_1000"].values
           for f in SELECTED}

folds = sorted(SELECTED)
nf = len(folds)
boots = np.empty(10000)
boots_diff = np.empty(10000)
for b in range(10000):
    fs = RNG.randint(0, nf, nf)
    cvals, dvals = [], []
    for fi in fs:
        f = folds[fi]
        c = per_seed[f]
        blv = bl_seed[f]
        si = RNG.randint(0, len(c), len(c))
        cvals.append(c[si].mean())
        dvals.append(blv[si].mean() - c[si].mean())
    boots[b] = np.mean(cvals)
    boots_diff[b] = np.mean(dvals)
ci = np.percentile(boots, [2.5, 97.5])
ci_d = np.percentile(boots_diff, [2.5, 97.5])
lofo_mean = np.mean([per_seed[f].mean() for f in folds])
print(f"\nLOFO cost: {lofo_mean:.1f}  bootCI [{ci[0]:.0f}, {ci[1]:.0f}]  "
      f"diff bootCI [{ci_d[0]:.1f}, {ci_d[1]:.1f}]")

# ── 3. LOFO deferral (threshold rule per fold with selected config) ──
def threshold_deferral(probs, ytrue, cm, rc=RC, cap=CAP):
    n = len(ytrue)
    risk = probs @ cm
    actions = risk.argmin(axis=1)
    best_risk = risk.min(axis=1)
    defer = best_risk > rc
    if defer.sum() > int(cap * n):
        idx = np.argsort(-best_risk)[:int(cap * n)]
        defer = np.zeros(n, bool)
        defer[idx] = True
    case_cost = cm[ytrue, actions]
    total = (case_cost[~defer].sum() + defer.sum() * rc) / n * 1000
    return total, defer.mean()

rows = []
for f, cfg in SELECTED.items():
    d = os.path.join(PRED, "Steel_Plates", cfg)
    for s in SEEDS:
        z = np.load(os.path.join(d, f"f{f}_s{s}.npz"))
        total, rate = threshold_deferral(z["probs"].astype(float), z["y_true"], CM)
        rows.append({"fold": f, "seed": s, "total": total, "rate": rate})
rdf = pd.DataFrame(rows)
fm = rdf.groupby("fold")["total"].mean()
m = fm.mean()
sem = fm.std(ddof=1) / np.sqrt(len(fm))
tci = stats.t.interval(0.95, len(fm) - 1, loc=m, scale=sem)
baseline = 244.3
print(f"LOFO deferral total: {m:.1f}  foldCI [{tci[0]:.0f}, {tci[1]:.0f}]  "
      f"rate {rdf['rate'].mean():.3f}  combined reduction {(baseline-m)/baseline*100:.1f}%")
rdf.to_csv(os.path.join(PROJ, "results", "revision", "policy_matched",
                        "Steel_LOFO_threshold_deferral.csv"), index=False)
print("DONE")
