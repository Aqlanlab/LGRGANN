from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score
from .data_loading import DatasetSpec,load_table,infer_columns,sort_by_time
from .data_preprocess import fit_preprocess,transform_ordinal
from .data_splits import outer_time_series_splits
from .utils_seeding import seed_everything

@dataclass
class MissingnessDiagConfig:
    model:str="logreg";max_iter:int=5000;solver:str="saga";penalty:str="l2";C:float=1.0

def run_missingness_diagnostic(cfg:dict)->pd.DataFrame:
    spec=DatasetSpec(**cfg["data"]);df=load_table(spec);df=sort_by_time(df,spec.time_col)
    nc,cc=infer_columns(df,spec.label_col,spec.time_col,cfg["data"].get("categorical_cols"),cfg["data"].get("numeric_cols"))
    seeds=cfg.get("evaluation",{}).get("seeds",[0]);of=int(cfg.get("evaluation",{}).get("outer_folds",5))
    dc=MissingnessDiagConfig(**cfg.get("missingness_diagnostic",{}));rows=[]
    for fi,(tri,tei)in enumerate(outer_time_series_splits(len(df),of),start=1):
        dtr,dte=df.iloc[tri].reset_index(drop=True),df.iloc[tei].reset_index(drop=True)
        art=fit_preprocess(dtr,num_cols=nc,cat_cols=cc);Xtr,Xte=transform_ordinal(dtr,art),transform_ordinal(dte,art)
        ytr=pd.isna(dtr[spec.label_col].to_numpy()).astype(int);yte=pd.isna(dte[spec.label_col].to_numpy()).astype(int)
        if len(np.unique(ytr))<2 or len(np.unique(yte))<2:
            rows.append({"fold":fi,"seed":None,"roc_auc":float("nan"),"pr_auc":float("nan"),"prevalence_test":float(yte.mean()),"note":"Skipped (single-class)"})
            continue
        for seed in seeds:
            seed_everything(seed)
            m=LogisticRegression(max_iter=dc.max_iter,solver=dc.solver,penalty=dc.penalty,C=dc.C,n_jobs=-1)
            m.fit(Xtr,ytr);pr=m.predict_proba(Xte)[:,1]
            rows.append({"fold":fi,"seed":seed,"roc_auc":float(roc_auc_score(yte,pr)),"pr_auc":float(average_precision_score(yte,pr)),"prevalence_test":float(yte.mean()),"note":""})
    return pd.DataFrame(rows)
