"""
Generate calibration reliability diagram for Scientific Reports paper.
Runs one representative temporal fold of DIMM dataset, comparing
uncalibrated vs Platt vs isotonic vs temperature-scaled probabilities.

Output: fig_calibration.svg + fig_calibration.png
Usage:  conda activate ml_env_new && python make_calibration_fig.py
"""
import sys, os, warnings
warnings.filterwarnings('ignore')
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from sklearn.calibration import calibration_curve
import xgboost as xgb

from preprocessing import load_and_clean, build_features, encode_dataset, make_forward_chain_folds
from calibration import platt_calibrate, isotonic_calibrate, temperature_scale

SEED = 0
FOLD_IDX = 3  # middle fold (0-indexed, fold 4 of 7)
N_BINS = 10

COLORS = {
    'Uncalibrated': '#D32F2F',
    'Platt':        '#1976D2',
    'Isotonic':     '#2E7D32',
    'Temperature':  '#F57C00',
}
CLASS_NAMES = ['Repair', 'Return-to-Vendor', 'Scrap']

def main():
    # Load DIMM
    df_c, _ = load_and_clean("Dataset.xlsx")
    fc, cc, nc = build_features(df_c)
    de, lt, cn, enc = encode_dataset(df_c, fc, cc, nc)
    X = de[fc].values.astype(np.float32)
    y = de["y"].values
    folds = make_forward_chain_folds(de, n_blocks=10)

    # Cost matrix
    cm = np.array([[0, 2, 1], [2, 0, 2], [5, 5, 0]], dtype=float)

    # Cost-sensitive sample weights (balanced=True for best config)
    nc_cls = cm.shape[0]
    class_cost = np.array([max(cm[c, j] for j in range(nc_cls) if j != c) for c in range(nc_cls)])
    counts = np.bincount(y, minlength=nc_cls).astype(float)
    freq = counts / len(y)
    class_weight = class_cost * np.sqrt(1.0 / (freq + 1e-10))

    # Get fold
    tr, va, te = folds[FOLD_IDX]
    X_tr, X_va, X_te = X[tr], X[va], X[te]
    y_tr, y_va, y_te = y[tr], y[va], y[te]

    # Train cost-balanced XGBoost
    sw = class_weight[y_tr].astype(np.float64)
    sw /= sw.mean()

    model = xgb.XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
        reg_alpha=0.1, reg_lambda=1.0, gamma=0.1,
        objective="multi:softprob", eval_metric="mlogloss",
        random_state=SEED, verbosity=0, n_jobs=1
    )
    model.fit(X_tr, y_tr, sample_weight=sw)
    print(f"Model trained on fold {FOLD_IDX+1}, train={len(tr)}, val={len(va)}, test={len(te)}")

    # Get probabilities under each calibration method
    probs_uncal = model.predict_proba(X_te)

    probs_platt, _ = platt_calibrate(model, X_va, y_va, X_te)
    probs_iso, _ = isotonic_calibrate(model, X_va, y_va, X_te)

    probs_val = model.predict_proba(X_va)
    probs_test_raw = model.predict_proba(X_te)
    probs_temp, T = temperature_scale(probs_val, y_va, probs_test_raw)
    print(f"Temperature T = {T:.3f}")

    methods = {
        'Uncalibrated': probs_uncal,
        'Platt':        probs_platt,
        'Isotonic':     probs_iso,
        'Temperature':  probs_temp,
    }

    # Compute ECE for each method
    def compute_ece(probs, y_true, n_bins=15):
        preds = probs.argmax(axis=1)
        confs = probs.max(axis=1)
        correct = (preds == y_true).astype(float)
        bin_edges = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            mask = (confs > bin_edges[i]) & (confs <= bin_edges[i+1])
            if mask.sum() > 0:
                avg_conf = confs[mask].mean()
                avg_acc = correct[mask].mean()
                ece += mask.sum() / len(y_true) * abs(avg_conf - avg_acc)
        return ece

    # Create figure: 3 columns (one per class) + histogram row
    fig, axes = plt.subplots(2, 3, figsize=(10, 6.5),
                              gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.08, 'wspace': 0.28})

    for k, class_name in enumerate(CLASS_NAMES):
        ax_cal = axes[0, k]
        ax_hist = axes[1, k]
        y_binary = (y_te == k).astype(int)

        for method_name, probs in methods.items():
            prob_k = probs[:, k]
            try:
                fraction_pos, mean_pred = calibration_curve(
                    y_binary, prob_k, n_bins=N_BINS, strategy='uniform'
                )
                ax_cal.plot(mean_pred, fraction_pos,
                           color=COLORS[method_name], linewidth=1.8,
                           marker='o', markersize=4, label=method_name,
                           zorder=3)
            except Exception as e:
                print(f"  Warning: {method_name} class {class_name}: {e}")

        # Perfect calibration diagonal
        ax_cal.plot([0, 1], [0, 1], 'k--', linewidth=1.0, alpha=0.5, label='Perfect')
        ax_cal.set_xlim(-0.02, 1.02)
        ax_cal.set_ylim(-0.02, 1.02)
        ax_cal.set_title(class_name, fontsize=12, fontweight='bold', pad=8)
        ax_cal.xaxis.set_major_locator(MultipleLocator(0.2))
        ax_cal.yaxis.set_major_locator(MultipleLocator(0.2))
        ax_cal.grid(True, alpha=0.3, linewidth=0.5)
        ax_cal.set_aspect('equal')

        if k == 0:
            ax_cal.set_ylabel('Fraction of positives', fontsize=10)
        else:
            ax_cal.set_yticklabels([])

        ax_cal.set_xticklabels([])

        # Histogram of predictions (uncalibrated vs isotonic)
        bins = np.linspace(0, 1, N_BINS + 1)
        ax_hist.hist(probs_uncal[:, k], bins=bins, color=COLORS['Uncalibrated'],
                     alpha=0.4, label='Uncal.', edgecolor='none')
        ax_hist.hist(probs_iso[:, k], bins=bins, color=COLORS['Isotonic'],
                     alpha=0.4, label='Iso.', edgecolor='none')
        ax_hist.set_xlim(-0.02, 1.02)
        ax_hist.xaxis.set_major_locator(MultipleLocator(0.2))
        ax_hist.set_xlabel('Mean predicted probability', fontsize=9)
        if k == 0:
            ax_hist.set_ylabel('Count', fontsize=9)
        else:
            ax_hist.set_yticklabels([])
        ax_hist.tick_params(labelsize=8)

    # ECE annotation
    ece_uncal = compute_ece(probs_uncal, y_te)
    ece_iso = compute_ece(probs_iso, y_te)
    ece_platt = compute_ece(probs_platt, y_te)
    ece_temp = compute_ece(probs_temp, y_te)
    print(f"ECE — Uncal: {ece_uncal:.3f}, Platt: {ece_platt:.3f}, Iso: {ece_iso:.3f}, Temp: {ece_temp:.3f}")

    # Single legend at top
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=5, fontsize=9,
               bbox_to_anchor=(0.5, 0.98), frameon=False)

    # ECE text box
    ece_text = (f'ECE: Uncal={ece_uncal:.3f}  Platt={ece_platt:.3f}  '
                f'Iso={ece_iso:.3f}  Temp={ece_temp:.3f}')
    fig.text(0.5, 0.01, ece_text, ha='center', fontsize=8.5, style='italic',
             color='#444444')

    plt.subplots_adjust(top=0.92, bottom=0.10, left=0.08, right=0.97)

    # Save
    out_dir = os.path.dirname(os.path.abspath(__file__))
    for ext in ['svg', 'png']:
        path = os.path.join(out_dir, f'fig_calibration.{ext}')
        fig.savefig(path, dpi=600, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"Saved {path}")

    plt.close()
    print("Done.")

if __name__ == '__main__':
    main()
