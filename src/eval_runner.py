from __future__ import annotations
from typing import Dict,List
import numpy as np
import pandas as pd
from tqdm import tqdm
from xgboost import XGBClassifier
from .utils_seeding import seed_everything
from .data_loading import DatasetSpec,load_table,infer_columns,sort_by_time
from .data_preprocess import fit_preprocess,transform_ordinal
from .data_splits import outer_time_series_splits,inner_stratified_splits
from .reconstruction_lgrgann import Stage1Config,run_stage1
from .wbo_opt import WboConfig,wbo_cimp_optimize
from .eval_metrics import accuracy,macro_f1,per_class_f1,weighted_f1_custom,expected_cost_multiclass,expected_cost_index,expected_cost_scrap_binary_per_1000

def simulate_missing_labels(yf:np.ndarray,Xt:np.ndarray,mech:str,mr:float,seed:int,cls:np.ndarray)->np.ndarray:
    rng,n,m=np.random.default_rng(seed),len(yf),int(round(mr*len(yf)))
    if m<=0:return np.ones(n,dtype=bool)
    M=mech.upper()
    if M=="MAR":
        w=rng.normal(size=Xt.shape[1]);p=1/(1+np.exp(-Xt@w))
        idx=np.argsort(p)[-m:];mk=np.ones(n,dtype=bool);mk[idx]=False;return mk
    if M=="MNAR_CLASS":
        yi=np.array([int(np.where(cls==yy)[0][0])for yy in yf],dtype=int)
        f=np.bincount(yi,minlength=len(cls)).astype(float);inv=1.0/(f+1e-6);p=inv[yi]
        idx=np.argsort(p)[-m:];mk=np.ones(n,dtype=bool);mk[idx]=False;return mk
    if M=="MNAR_HARDNESS":
        yi=np.array([int(np.where(cls==yy)[0][0])for yy in yf],dtype=int)
        md=XGBClassifier(objective="multi:softprob",num_class=len(cls),eval_metric="mlogloss",n_estimators=300,max_depth=6,learning_rate=0.1,subsample=0.9,colsample_bytree=0.9,random_state=seed,n_jobs=-1)
        md.fit(Xt,yi);conf=md.predict_proba(Xt).max(axis=1)
        idx=np.argsort(conf)[:m];mk=np.ones(n,dtype=bool);mk[idx]=False;return mk
    if M=="MNAR_HYBRID":
        yi=np.array([int(np.where(cls==yy)[0][0])for yy in yf],dtype=int)
        f=np.bincount(yi,minlength=len(cls)).astype(float);ct=1.0/(f+1e-6);ct=ct[yi]
        md=XGBClassifier(objective="multi:softprob",num_class=len(cls),eval_metric="mlogloss",n_estimators=200,max_depth=5,learning_rate=0.12,subsample=0.9,colsample_bytree=0.9,random_state=seed,n_jobs=-1)
        md.fit(Xt,yi);hd=1.0-md.predict_proba(Xt).max(axis=1)
        sc=ct+hd;idx=np.argsort(sc)[-m:];mk=np.ones(n,dtype=bool);mk[idx]=False;return mk
    raise ValueError(f"Unknown mechanism: {mech}")

def train_xgb_with_weights(Xt:np.ndarray,yt:np.ndarray,cls:np.ndarray,xp:dict,cw:np.ndarray,seed:int)->XGBClassifier:
    yi=np.array([int(np.where(cls==yy)[0][0])for yy in yt],dtype=int)
    sw=np.array([cw[i]for i in yi],dtype=float)
    m=XGBClassifier(objective="multi:softprob",num_class=len(cls),eval_metric="mlogloss",random_state=seed,n_jobs=-1,**xp)
    m.fit(Xt,yi,sample_weight=sw)
    return m

def evaluate_multiclass(m:XGBClassifier,Xte:np.ndarray,yte:np.ndarray,cls:np.ndarray,cw:np.ndarray)->Dict[str,float]:
    L=list(range(len(cls)))
    yti=np.array([int(np.where(cls==yy)[0][0])for yy in yte],dtype=int);ypi=m.predict(Xte)
    out={"accuracy":accuracy(yti,ypi),"macro_f1":macro_f1(yti,ypi,L)}
    f1s=per_class_f1(yti,ypi,L)
    for i,c in enumerate(cls):out[f"f1_{c}"]=float(f1s[i])
    out["cv_wfi_on_test"]=weighted_f1_custom(yti,ypi,cw,L)
    return out

def run_dimm_pipeline(cfg:dict)->pd.DataFrame:
    spec=DatasetSpec(**cfg["data"]);df=load_table(spec);df=sort_by_time(df,spec.time_col)
    opt=cfg.get("options",{})or{}
    if opt.get("use_only_labeled_rows",True):df=df[~pd.isna(df[spec.label_col])].reset_index(drop=True)
    nc,cc=infer_columns(df,spec.label_col,spec.time_col,cfg["data"].get("categorical_cols"),cfg["data"].get("numeric_cols"))
    ya=df[spec.label_col].to_numpy();obs=pd.Series(ya).dropna().unique();cls=np.array(sorted(obs.tolist()))
    outputs=[];seeds=cfg["evaluation"]["seeds"];of=int(cfg["evaluation"]["outer_folds"]);nf=int(cfg["evaluation"]["inner_folds"])
    mr=float(cfg["evaluation"]["missing_rate"]);mech=cfg["evaluation"]["missing_mechanism"];pc=cfg["evaluation"]["priority_class"]
    for fi,(tri,tei) in enumerate(outer_time_series_splits(len(df),of),start=1):
        dtr,dte=df.iloc[tri].reset_index(drop=True),df.iloc[tei].reset_index(drop=True)
        art=fit_preprocess(dtr,num_cols=nc,cat_cols=cc);Xtr,Xte=transform_ordinal(dtr,art),transform_ordinal(dte,art)
        ytrf,yte=dtr[spec.label_col].to_numpy(),dte[spec.label_col].to_numpy()
        tmgt=~pd.isna(yte);Xtee,ytee=Xte[tmgt],yte[tmgt]
        for seed in seeds:
            seed_everything(seed);trgt=~pd.isna(ytrf);ytrg,Xtrg=ytrf[trgt],Xtr[trgt]
            lmg=simulate_missing_labels(ytrg,Xtrg,mech,mr,seed+1000*fi,cls)
            yts=ytrg.astype(object).copy();yts[~lmg]=np.nan;lm=~pd.isna(yts)
            s1c=Stage1Config(**cfg["stage1"])
            isp=list(inner_stratified_splits(np.array([int(np.where(cls==yy)[0][0])for yy in ytrg[lmg]],dtype=int),nf,seed))
            s1=run_stage1(Xtrg,yts,lm,cls,seed,s1c,isp);yr=s1["y_recon"]
            sc=cfg["stage2"]["search_space"];wc=WboConfig(**cfg["stage2"]["wbo"])
            op=wbo_cimp_optimize(Xtrg,yr,cls,pc,nf,seed,sc,wc)
            bw,bp=op["best_class_weights"],op["best_xgb_params"]
            md=train_xgb_with_weights(Xtrg,yr,cls,bp,bw,seed);mt=evaluate_multiclass(md,Xtee,ytee,cls,bw)
            ccfg=cfg.get("costs",{});C=np.array(ccfg.get("cost_matrix"))
            if C.size and C.shape[0]==len(cls)and C.shape[1]==len(cls):
                yti=np.array([int(np.where(cls==yy)[0][0])for yy in ytee],dtype=int);ypi=md.predict(Xtee)
                cost=expected_cost_multiclass(yti,ypi,C,list(range(len(cls))))
                uni=np.ones(len(cls))/len(cls);bm=train_xgb_with_weights(Xtrg,yr,cls,bp,uni,seed)
                bc=expected_cost_multiclass(yti,bm.predict(Xtee),C,list(range(len(cls))))
                mt["expected_cost_index"]=expected_cost_index(cost,bc);mt["expected_cost"]=cost;mt["baseline_cost"]=bc
            if pc in list(cls):
                yti=np.array([int(np.where(cls==yy)[0][0])for yy in ytee],dtype=int);ypi=md.predict(Xtee)
                pidx=int(np.where(cls==pc)[0][0]);yts2=(yti==pidx).astype(int);yps2=(ypi==pidx).astype(int)
                bc2=ccfg.get("binary_scrap",{});mt["scrap_binary_cost_per_1000"]=expected_cost_scrap_binary_per_1000(yts2,yps2,float(bc2.get("false_scrap",1.0)),float(bc2.get("missed_scrap",5.0)))
            row={"fold":fi,"seed":seed,"k":s1["k"],"stage1_refiner":s1.get("refiner"),"best_cv_wfi":op["best_cv_wfi"],**mt}
            outputs.append(row)
    return pd.DataFrame(outputs)
