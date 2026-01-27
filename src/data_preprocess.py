from __future__ import annotations
from typing import Dict,List,Tuple,Any
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler,OrdinalEncoder,OneHotEncoder

def fit_preprocess(df:pd.DataFrame,num_cols:List[str],cat_cols:List[str])->Dict[str,Any]:
    ns=StandardScaler().fit(df[num_cols].fillna(df[num_cols].median()))if num_cols else None
    oe=OrdinalEncoder(handle_unknown="use_encoded_value",unknown_value=-1).fit(df[cat_cols].fillna("__NA__").astype(str))if cat_cols else None
    ohe=OneHotEncoder(sparse_output=False,handle_unknown="ignore").fit(df[cat_cols].fillna("__NA__").astype(str))if cat_cols else None
    return{"num_cols":num_cols,"cat_cols":cat_cols,"num_scaler":ns,"ord_enc":oe,"ohe":ohe,"num_medians":df[num_cols].median().to_dict()if num_cols else{}}

def transform_ordinal(df:pd.DataFrame,art:Dict[str,Any])->np.ndarray:
    nc,cc,ns,oe,med=art["num_cols"],art["cat_cols"],art["num_scaler"],art["ord_enc"],art["num_medians"]
    parts=[]
    if nc:
        Xn=df[nc].copy();[Xn[c].fillna(med.get(c,0),inplace=True)for c in nc]
        parts.append(ns.transform(Xn))
    if cc:parts.append(oe.transform(df[cc].fillna("__NA__").astype(str)))
    return np.hstack(parts)if parts else np.zeros((len(df),0))

def transform_onehot_scaled(df:pd.DataFrame,art:Dict[str,Any])->np.ndarray:
    nc,cc,ns,ohe,med=art["num_cols"],art["cat_cols"],art["num_scaler"],art["ohe"],art["num_medians"]
    parts=[]
    if nc:
        Xn=df[nc].copy();[Xn[c].fillna(med.get(c,0),inplace=True)for c in nc]
        parts.append(ns.transform(Xn))
    if cc:parts.append(ohe.transform(df[cc].fillna("__NA__").astype(str)))
    return np.hstack(parts)if parts else np.zeros((len(df),0))
