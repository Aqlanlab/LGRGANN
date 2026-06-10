import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer


def _map_dispo(d):
    if pd.isna(d):
        return None
    dl = d.lower().strip()
    if "scrap" in dl:
        return "Scrap"
    if any(k in dl for k in ["rma", "vendor", "samsung", "hynix", "micron", "ipr", "rochester"]):
        return "Return to Vendor"
    return "Repair"


def load_and_clean(path="Dataset.xlsx"):
    df = pd.read_excel(path)
    df.columns = df.columns.str.strip()

    df["Incident Date"] = pd.to_datetime(df["Incident Date"], errors="coerce")
    df = df.dropna(subset=["Incident Date"]).copy()
    df.rename(columns={"Incident Date": "Date"}, inplace=True)

    df["Disposition"] = df["Dispo"].apply(_map_dispo)

    drop_cols = [
        "Dispo", "Dispo Date", "Serial Number", "Original SN",
        "Machine Serial ID", "Defect ID", "Update Id", "Update TS",
        "Description", "Operator ID", "Ref Code.1", "FA Table Index",
        "Incident", "Parent DefID", "Key Call", "Key Comment",
        "Reportable", "Root Cause",
        "Ref Code", "Part Number", "Original PN", "Vintage Date",
        "Fail Comp", "Failing Comp", "Op Num",
        "Root Cause ", "Valid",
    ]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    df_all = df.copy()
    df_confirmed = df.dropna(subset=["Disposition"]).copy()
    df_confirmed = df_confirmed[df_confirmed["Disposition"].isin(
        ["Scrap", "Repair", "Return to Vendor"]
    )].copy()
    df_confirmed.sort_values("Date", inplace=True)
    df_confirmed.reset_index(drop=True, inplace=True)
    return df_confirmed, df_all


def build_features(df):
    date_col = "Date"
    target_col = "Disposition"
    exclude = [date_col, target_col, "y"]

    feature_cols = [c for c in df.columns if c not in exclude]

    datetime_cols = df[feature_cols].select_dtypes(include=["datetime64", "datetimetz"]).columns.tolist()
    for c in datetime_cols:
        feature_cols.remove(c)

    cat_cols = df[feature_cols].select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = df[feature_cols].select_dtypes(include=["number"]).columns.tolist()

    return feature_cols, cat_cols, num_cols


def encode_dataset(df, feature_cols, cat_cols, num_cols):
    df_enc = df.copy()
    le_target = LabelEncoder()
    df_enc["y"] = le_target.fit_transform(df_enc["Disposition"])
    class_names = le_target.classes_

    encoders = {}
    for c in cat_cols:
        df_enc[c] = df_enc[c].astype(str).fillna("MISSING")
        le = LabelEncoder()
        df_enc[c] = le.fit_transform(df_enc[c])
        encoders[c] = le

    for c in num_cols:
        df_enc[c] = pd.to_numeric(df_enc[c], errors="coerce")
    if num_cols:
        imp = SimpleImputer(strategy="median")
        df_enc[num_cols] = imp.fit_transform(df_enc[num_cols])

    return df_enc, le_target, class_names, encoders


def make_forward_chain_folds(df, n_blocks=6):
    df = df.sort_values("Date").reset_index(drop=True)
    n = len(df)
    block_size = n // n_blocks
    blocks = []
    for i in range(n_blocks):
        start = i * block_size
        end = (i + 1) * block_size if i < n_blocks - 1 else n
        blocks.append(list(range(start, end)))

    folds = []
    for j in range(2, n_blocks - 1):
        train_idx = []
        for b in range(j):
            train_idx.extend(blocks[b])
        val_idx = blocks[j]
        test_idx = blocks[j + 1]
        folds.append((train_idx, val_idx, test_idx))
    return folds
