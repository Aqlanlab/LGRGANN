# Cost-Calibrated Selective Decision Support for Manufacturing Defect Disposition

This repository contains the code and experimental framework for our paper:

**"Cost-Calibrated Selective Decision Support for Manufacturing Defect Disposition under Asymmetric Misclassification Costs"**

## Overview

We propose a cost-calibrated decision-support framework for manufacturing defect disposition that integrates four components:

1. **Cost-sensitive training** — sample weights derived from a misclassification cost matrix, with optional class-frequency balancing
2. **Post-hoc probability calibration** — Platt scaling, isotonic regression, or temperature scaling, selected automatically via inner cross-validation
3. **Bayes-risk action selection** — decisions minimise expected cost rather than maximising predicted probability
4. **Risk-based deferral** — high-risk cases are routed to human review, subject to capacity constraints

The framework is validated on three manufacturing datasets (DIMM memory modules, Steel Plates faults, SECOM semiconductor) with statistically significant cost reductions on all three.

## Repository Structure

```
├── src/
│   ├── calibration.py          # Platt, isotonic, temperature scaling
│   ├── decision_layer.py       # Bayes-risk decisions and deferral policies
│   ├── metrics.py              # Cost-per-1000, ECE, classification metrics
│   ├── preprocessing.py        # DIMM dataset loading and feature engineering
│   ├── public_datasets.py      # Steel Plates and SECOM loaders
│   ├── run_final.py            # Main experiments (12 configs × 3 datasets)
│   └── run_supplement.py       # Supplementary experiments (baselines, sensitivity)
├── data/
│   └── README.md               # Dataset acquisition instructions
├── figures/
│   ├── generate_figures.py     # Result visualisation (Tables → Figures)
│   └── generate_calibration_fig.py  # Calibration reliability diagrams
├── results/                    # Output directory (populated by experiments)
├── requirements.txt
├── environment.yml
└── .gitignore
```

## Quick Start

### 1. Environment Setup

```bash
# Option A: Conda (recommended)
conda env create -f environment.yml
conda activate cost_calibrated

# Option B: pip
pip install -r requirements.txt
```

### 2. Obtain Datasets

See [`data/README.md`](data/README.md) for acquisition instructions. Place the DIMM dataset as `Dataset.xlsx` in the repository root. Steel Plates and SECOM are downloaded automatically from UCI on first run.

### 3. Run Experiments

```bash
# Main experiments: 12 configurations × 3 datasets
# Includes deferral sweeps, policy comparison, and Wilcoxon tests
cd src
python run_final.py

# Supplementary experiments: external baselines, recall constraints,
# deferral sensitivity, cost matrix sensitivity
python run_supplement.py
```

Results are saved to `results/final/` and `results/supplement/` as CSV files.

### 4. Generate Figures

```bash
cd figures
python generate_figures.py
python generate_calibration_fig.py
```

## Framework Architecture

```
Input features x
        │
        ▼
┌─────────────────────┐
│  Cost-Sensitive      │  w(i) = max off-diagonal cost for class y_i
│  Base Classifier     │  Optional: w_balanced = w · √(N / (K · n_c))
│  (XGBoost / LightGBM)│
└─────────┬───────────┘
          │ raw P(y|x)
          ▼
┌─────────────────────┐
│  Ensemble (optional) │  Average calibrated probabilities
│  XGB + LGBM          │  from two independently trained models
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Calibration Layer   │  Platt / Isotonic / Temperature
│  (auto-selected)     │  2-fold inner CV on validation cost
└─────────┬───────────┘
          │ calibrated P(y|x)
          ▼
┌─────────────────────┐
│  Bayes-Risk Decision │  a* = argmin_a Σ_c C(c,a) · P(c|x)
│  Rule                │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Risk-Based Deferral │  Defer if min_a R(a|x) > review_cost
│  (capacity-bounded)  │  Subject to capacity constraint B
└─────────┬───────────┘
          │
          ▼
    Final disposition
    or human review
```

## Configuration Key

Experiment configurations follow a naming convention:

| Prefix | Meaning |
|--------|---------|
| `CS_`  | Cost-sensitive sample weights (`balanced=False`) |
| `CB_`  | Cost-balanced weights: cost × √(1/freq) (`balanced=True`) |
| `Ens`  | XGB + LightGBM ensemble |
| `Iso`  | Isotonic calibration |
| `Auto` | Auto-selected calibration (inner CV) |
| `Platt`| Platt / sigmoid scaling |
| `Temp` | Temperature scaling |
| `BR`   | Bayes-risk decision rule |

Example: `CB_Ens_Iso_BR` = cost-balanced ensemble with isotonic calibration and Bayes-risk decisions.

## Cost Matrices

### DIMM (3-class: Repair, Return-to-Vendor, Scrap)

| True \ Action | Repair | RTV | Scrap |
|---|---|---|---|
| Repair | 0 | 2 | 1 |
| RTV | 2 | 0 | 2 |
| Scrap | 5 | 5 | 0 |

### Steel Plates (3-class: Minor, Moderate, Severe)

| True \ Action | Minor | Moderate | Severe |
|---|---|---|---|
| Minor | 0 | 2 | 1 |
| Moderate | 3 | 0 | 2 |
| Severe | 5 | 3 | 0 |

### SECOM (binary: Pass, Fail)

| True \ Action | Pass | Fail |
|---|---|---|
| Pass | 0 | 1 |
| Fail | 5 | 0 |

## Validation Strategy

- **DIMM**: Blocked forward-chaining temporal validation (10 time blocks → 7 folds × 10 random seeds = 70 runs per configuration)
- **Steel Plates**: Stratified 5-fold CV × 10 seeds = 50 runs
- **SECOM**: Stratified 5-fold CV × 10 seeds = 50 runs, with per-fold feature selection (top 100 by XGBoost importance)

## Key Results

| Dataset | Baseline (XGB Argmax) | Best Config | Cost | Reduction | p-value |
|---|---|---|---|---|---|
| DIMM | 543.5 | CB_Ens_Iso_BR | 410.5 | −24.5% | < 0.0001 |
| Steel | 244.3 | CS_Ens_Auto_BR | 236.6 | −3.2% | 0.019 |
| SECOM | 326.5 | CS_BR | 311.4 | −4.6% | 0.0001 |

With 10% deferral capacity and risk-based routing, total operational costs (including review fees) drop a further 11–16% across datasets.

## Software Versions

| Package | Version |
|---------|---------|
| Python | 3.11 |
| scikit-learn | 1.2.2 |
| XGBoost | 2.1.4 |
| LightGBM | 4.6.0 |
| CatBoost | 1.2.7 |
| imbalanced-learn | 0.12.4 |
| NumPy | 1.24+ |
| pandas | 2.0+ |
| SciPy | 1.10+ |
| matplotlib | 3.7+ |

## Citation

If you use this code, please cite:

```bibtex
@article{srivastava2025costcalibrated,
  title={Cost-Calibrated Selective Decision Support for Manufacturing Defect Disposition under Asymmetric Misclassification Costs},
  author={Srivastava, Sudhanshu},
  journal={Scientific Reports},
  year={2025}
}
```

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
