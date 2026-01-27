#  Defect Disposition Prediction in Manufacturing Environments: Integrating Cost-Sensitive Bayesian Optimization with Label Reconstruction under Incomplete Annotations

This repository provides a **two-stage decision-support pipeline** for defect disposition prediction in manufacturing when (i) **target labels are partially missing** and (ii) misclassification errors have **asymmetric business costs**. The approach reconstructs missing disposition labels (Stage 1) and then trains a cost-sensitive classifier with **joint hyperparameter + class-weight optimization** (Stage 2). 


## Abstract

In today’s complex manufacturing environments, predictive models are crucial for reducing costly disruptions. However, model performance is often undermined by **missing outcome annotations** and **asymmetric misclassification costs**. We propose an integrated two-stage approach validated on an industrial dataset from a leading computer server manufacturer.  

**Stage 1** introduces a novel **label reconstruction** method that combines class-informed neighbor search with residual error refinement to correct structured pseudo-label errors.  
**Stage 2** introduces **cost-sensitive weighted Bayesian optimization** to jointly tune classifier hyperparameters and class weights using a custom F1-based objective prioritizing the high-cost defect class.  

On an industrial dataset with **20% missing disposition labels**, reconstructed labels improved downstream performance to **F1 = 0.801** (vs **0.762** baseline). Cost-sensitive optimization improved the high-impact class performance and the overall framework achieved a **25.3% reduction in expected misclassification cost**. 

## Proposed Methodology

![alt text](assets/Figure%201.png)

**Figure:** Proposed Methodology is shown in two panels: (a) Stage 1 (LGRGANN label reconstruction) and (b) Stage 2 (WBO-CIMP cost-sensitive optimization), with outputs labeled.

## Key Features

- **Two-stage design (reconstruction → cost-sensitive learning)** to separate label uncertainty reduction from business-aligned decision learning.
- **LGRGANN (Stage 1):** MI-weighted Grey-KNN pseudo-labeling + conditional residual refiner (GAN-based) to correct systematic reconstruction errors. 
- **WBO-CIMP (Stage 2):** Bayesian Optimization that **jointly searches** class weights and hyperparameters under a **priority constraint** for the cost-critical class.  
- **Temporal generalization (forward-chaining evaluation)** to avoid future-to-past leakage and test performance on later production blocks.  
- **Business-oriented reporting** via expected misclassification cost per 1000 decisions (relative units). 


## Key Results

### Stage 1 — Label Reconstruction (LGRGANN)
![alt text](assets/Figure%202.png)

**Figure:** Downstream disposition prediction performance under 20% missing Disposition labels (DIMM dataset). Bars report mean ± standard deviation over 10 random seeds × 5 temporal folds (n = 50) using the time-ordered evaluation (train earlier batches, test later batches). All methods use XGBoost as the downstream classifier; differences reflect the quality of label reconstruction / label inference in the training window.

**Table:** Direct label reconstruction quality on the real DIMM dataset (mask-and-reconstruct), mean ± SD (n = 50)

![alt text](assets/Table%201.png)

### Stage 2 — Cost-Sensitive Classification (WBO-CIMP)
**Table:** Performance comparison of WBO-CIMP against Baselines and Standard Cost Sensitive Methods
![alt text](assets/Table%202.png)


![alt text](assets/Figure%203.png)
**Figure:** F1-score progression over iterations (left) and average ‘Scrap’ F1 vs wscrap (right) for WBO-CIMP


## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Usage

```bash
python scripts/run_all.py --config configs/dimm_example.yaml
python scripts/run_synthetic_rmse.py --config configs/synthetic_example.yaml
python scripts/run_sensitivity_mnar.py --config configs/dimm_example.yaml
python scripts/run_missingness_diagnostic.py --config configs/dimm_example.yaml
```

## Structure

```
src/lgrgann_wbocimp/
├── reconstruction_mi.py
├── reconstruction_grey_knn.py
├── reconstruction_gan.py
├── reconstruction_lgrgann.py
├── wbo_opt.py
├── baselines_reconstruction.py
├── eval_metrics.py
├── eval_runner.py
├── missingness_diagnostic.py
└── synthetic.py
```

## Citation

```bibtex
@article{srivastava2025costsensitive,
  title={Defect Disposition Prediction in Manufacturing Environments: Integrating Cost-Sensitive Bayesian Optimization with Label Reconstruction under Incomplete Annotations},
  author={Srivastava, Sudhanshu and Aqlan, Faisal and Parikh, Pratik J. and Noor-E-Alam, Md.},
  year={2026}
}
```
