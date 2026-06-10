# Dataset Acquisition

This study uses three manufacturing datasets. Due to licensing restrictions, datasets are not included in this repository.

## 1. DIMM Memory Module Dataset (Primary)

**Source**: Proprietary semiconductor manufacturing data
**Records**: 3,818 confirmed labels (5,098 total)
**Features**: 32 (mostly categorical: Defect Code, Manufacturing Site, Supplier, Process Stage, Product Type, Memory Type/Size)
**Target**: Disposition — Return-to-Vendor (76.8%), Repair (20.3%), Scrap (2.9%)

Place the dataset file as `Dataset.xlsx` in the repository root directory.

### Expected columns
The preprocessing pipeline expects an Excel file with at minimum:
- `Incident Date` — timestamp for temporal fold generation
- `Dispo` — raw disposition string (mapped to Repair / Return to Vendor / Scrap)
- Feature columns (categorical and numeric) — remaining columns after dropping identifiers

## 2. Steel Plates Faults (UCI #198)

**Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/198/steel+plates+faults)
**Records**: 1,941 instances, 27 numeric features, 7 fault types
**Grouped**: Minor (6.5%), Moderate (38.1%), Severe (55.4%)

Downloaded automatically via `ucimlrepo` on first run. Cached to `data_public/steel_plates.csv`.

## 3. SECOM Semiconductor (UCI #179)

**Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/179/secom)
**Records**: 1,567 instances, 590 features (474 after cleanup), binary Pass/Fail
**Class distribution**: Pass (93.4%), Fail (6.6%)

Downloaded automatically via `ucimlrepo` on first run. Cached to `data_public/secom_full.csv`.

## Notes

- The `data_public/` directory is created automatically by `public_datasets.py` when loading UCI datasets for the first time.
- Feature selection (top 100 by XGBoost importance) is applied per fold for SECOM due to its high dimensionality.
- The DIMM dataset uses blocked forward-chaining temporal validation; Steel and SECOM use stratified 5-fold cross-validation.
