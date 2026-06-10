"""Download and preprocess public datasets for generalizability study."""
import os, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJ, "data_public")
os.makedirs(DATA_DIR, exist_ok=True)


# ── Steel Plates Faults ─────────────────────────────────────────────────────
def load_steel_plates():
    """Load Steel Plates Faults (UCI #198).
    1941 instances, 27 numeric features, 7 fault types.
    Grouped into 3 severity classes for cost-sensitive disposition framing:
      0 = Minor   (Stains + Dirtiness)            ~  127 (6.5%)
      1 = Moderate (Pastry + Z_Scratch + K_Scratch) ~  739 (38.1%)
      2 = Severe   (Bumps + Other_Faults)           ~ 1075 (55.4%)
    """
    cache = os.path.join(DATA_DIR, "steel_plates.csv")
    if os.path.exists(cache):
        df = pd.read_csv(cache)
    else:
        from ucimlrepo import fetch_ucirepo
        ds = fetch_ucirepo(id=198)
        X_df = ds.data.features
        y_df = ds.data.targets
        # y_df has 7 binary columns: one-hot
        fault_cols = list(y_df.columns)
        fault_name = y_df.idxmax(axis=1)
        df = X_df.copy()
        df["fault_type"] = fault_name.values
        df.to_csv(cache, index=False)

    # Group into 3 severity classes
    severity_map = {
        "Stains": "Minor", "Dirtiness": "Minor",
        "Pastry": "Moderate", "Z_Scratch": "Moderate", "K_Scatch": "Moderate",
        "K_Scratch": "Moderate",
        "Bumps": "Severe", "Other_Faults": "Severe",
    }
    df["severity"] = df["fault_type"].map(severity_map)
    df = df.dropna(subset=["severity"])

    feature_cols = [c for c in df.columns if c not in ("fault_type", "severity")]
    X = df[feature_cols].values.astype(np.float32)
    le = LabelEncoder()
    y = le.fit_transform(df["severity"])  # Minor=0, Moderate=1, Severe=2
    class_names = list(le.classes_)

    # Cost matrix: rows=true, cols=action  (same structure as DIMM)
    # Severe missed (true=Severe, action=Minor) = 5 (most costly)
    # Minor scrapped (true=Minor, action=Severe) = 1 (wasteful but not dangerous)
    idx_minor = list(le.classes_).index("Minor")
    idx_mod = list(le.classes_).index("Moderate")
    idx_sev = list(le.classes_).index("Severe")

    cost_matrix = np.zeros((3, 3), dtype=float)
    # true=Minor
    cost_matrix[idx_minor, idx_mod] = 2
    cost_matrix[idx_minor, idx_sev] = 1
    # true=Moderate
    cost_matrix[idx_mod, idx_minor] = 3
    cost_matrix[idx_mod, idx_sev] = 2
    # true=Severe
    cost_matrix[idx_sev, idx_minor] = 5
    cost_matrix[idx_sev, idx_mod] = 3

    high_cost_idx = idx_sev
    print(f"  Steel Plates: {len(X)} records, {X.shape[1]} features, classes={class_names}")
    print(f"  Distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    return X, y, class_names, cost_matrix, high_cost_idx, feature_cols


# ── SECOM ────────────────────────────────────────────────────────────────────
def load_secom():
    """Load SECOM semiconductor dataset (UCI #179).
    1567 instances, 590 features, binary pass/fail.
      0 = Pass (1463, 93.4%)
      1 = Fail (104, 6.6%)
    Has timestamps for temporal validation.
    """
    cache = os.path.join(DATA_DIR, "secom_full.csv")
    if os.path.exists(cache):
        full_df = pd.read_csv(cache)
    else:
        from ucimlrepo import fetch_ucirepo
        ds = fetch_ucirepo(id=179)
        # SECOM has features=None; use ds.data.original
        full_df = ds.data.original
        full_df.to_csv(cache, index=False)

    # Target: -1=Pass, 1=Fail -> 0=Pass, 1=Fail
    y_raw = full_df["class"].values
    y = np.where(y_raw == 1, 1, 0).astype(int)

    # Handle timestamps
    timestamps = None
    if "timestamp" in full_df.columns:
        try:
            timestamps = pd.to_datetime(full_df["timestamp"], dayfirst=True)
        except:
            try:
                timestamps = pd.to_datetime(full_df["timestamp"], format="mixed")
            except:
                pass

    # Feature columns = everything except class and timestamp
    feat_cols = [c for c in full_df.columns if c not in ("class", "timestamp")]
    X_df = full_df[feat_cols]

    # Feature preprocessing
    X_raw = X_df.values.astype(np.float64)

    # Remove near-zero variance features (std < 1e-6)
    stds = np.nanstd(X_raw, axis=0)
    keep_mask = stds > 1e-6
    X_raw = X_raw[:, keep_mask]
    kept_cols = [X_df.columns[i] for i, k in enumerate(keep_mask) if k]

    # Median imputation
    col_medians = np.nanmedian(X_raw, axis=0)
    for j in range(X_raw.shape[1]):
        mask = np.isnan(X_raw[:, j])
        X_raw[mask, j] = col_medians[j]

    # Remove any remaining constant columns after imputation
    stds2 = X_raw.std(axis=0)
    keep2 = stds2 > 1e-8
    X_raw = X_raw[:, keep2]
    kept_cols = [kept_cols[i] for i, k in enumerate(keep2) if k]

    X = X_raw.astype(np.float32)
    class_names = ["Pass", "Fail"]

    # Cost matrix (2x2): missing a Fail (true=Fail, action=Pass) = 5, false Fail = 1
    cost_matrix = np.array([
        [0, 1],   # true=Pass: false alarm costs 1
        [5, 0],   # true=Fail: missed defect costs 5
    ], dtype=float)

    high_cost_idx = 1  # Fail
    print(f"  SECOM: {len(X)} records, {X.shape[1]} features (after cleanup), classes={class_names}")
    print(f"  Distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    return X, y, class_names, cost_matrix, high_cost_idx, kept_cols, timestamps


# ── Fold generators ──────────────────────────────────────────────────────────
def make_stratified_folds(y, n_folds=5, val_frac=0.2, random_state=42):
    """Stratified K-fold with internal train/val split for calibration."""
    folds = []
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    for train_val_idx, test_idx in skf.split(np.zeros(len(y)), y):
        y_tv = y[train_val_idx]
        sss = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=random_state)
        for train_sub, val_sub in sss.split(np.zeros(len(y_tv)), y_tv):
            train_idx = train_val_idx[train_sub]
            val_idx = train_val_idx[val_sub]
            folds.append((train_idx, val_idx, test_idx))
    return folds


def make_temporal_folds_from_timestamps(timestamps, n_blocks=8):
    """Build temporal forward-chain folds from a timestamp Series."""
    sorted_idx = np.argsort(timestamps.values)
    n = len(sorted_idx)
    block_size = n // n_blocks
    blocks = []
    for i in range(n_blocks):
        start = i * block_size
        end = (i + 1) * block_size if i < n_blocks - 1 else n
        blocks.append(sorted_idx[start:end])

    folds = []
    for j in range(2, n_blocks - 1):
        train = np.concatenate(blocks[:j])
        val = blocks[j]
        test = blocks[j + 1]
        folds.append((train, val, test))
    return folds
