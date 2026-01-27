from __future__ import annotations
from dataclasses import dataclass
from typing import Dict,List,Tuple
import numpy as np
from skopt import gp_minimize
from skopt.space import Real,Integer
from skopt.utils import use_named_args
from xgboost import XGBClassifier
from .eval_metrics import weighted_f1_custom
from .data_splits import inner_stratified_splits

@dataclass
class WboConfig:
    bo_budget:int=200;n_initial:int=20;early_stop_rounds:int=40;acq_func:str="EI"

def _sm(x:np.ndarray)->np.ndarray:x=x-np.max(x);e=np.exp(x);return e/(e.sum()+1e-12)

def parameterize_class_weights(pi:int,betas:np.ndarray,ap:float)->np.ndarray:
    m=len(betas)+1;a=np.zeros(m,dtype=float);a[pi]=ap;j=0
    for i in range(m):
        if i==pi:continue
        a[i]=betas[j]*ap;j+=1
    a=np.maximum(a,0.0)
    return a/a.sum()if a.sum()>0 else np.ones(m)/m

def _space(sc:dict,nc:int)->Tuple[List,List[str]]:
    d,n=[Real(0.0,1.0,name="alpha_p")],["alpha_p"]
    for i in range(nc-1):d.append(Real(0.0,1.0,name=f"beta_{i}"));n.append(f"beta_{i}")
    d+=[Real(sc["reg_alpha"][0],sc["reg_alpha"][1],name="reg_alpha"),Real(sc["reg_lambda"][0],sc["reg_lambda"][1],name="reg_lambda"),Real(sc["gamma"][0],sc["gamma"][1],name="gamma"),Real(sc["learning_rate"][0],sc["learning_rate"][1],name="learning_rate"),Integer(sc["max_depth"][0],sc["max_depth"][1],name="max_depth"),Real(sc["min_child_weight"][0],sc["min_child_weight"][1],name="min_child_weight"),Real(sc["subsample"][0],sc["subsample"][1],name="subsample"),Real(sc["colsample_bytree"][0],sc["colsample_bytree"][1],name="colsample_bytree"),Integer(sc["n_estimators"][0],sc["n_estimators"][1],name="n_estimators")]
    n+=["reg_alpha","reg_lambda","gamma","learning_rate","max_depth","min_child_weight","subsample","colsample_bytree","n_estimators"]
    return d,n

def cv_wfi_score(X:np.ndarray,y:np.ndarray,cls:np.ndarray,pc:str,cw:np.ndarray,xp:dict,nf:int,seed:int)->float:
    L=list(range(len(cls)));yi=np.array([int(np.where(cls==yy)[0][0])for yy in y],dtype=int);sc=[]
    for tr,va in inner_stratified_splits(yi,nf,seed):
        m=XGBClassifier(objective="multi:softprob",num_class=len(cls),eval_metric="mlogloss",n_jobs=-1,random_state=seed,**xp)
        m.fit(X[tr],yi[tr],sample_weight=np.array([cw[yy]for yy in yi[tr]],dtype=float))
        sc.append(weighted_f1_custom(yi[va],m.predict(X[va]),cw,L))
    return float(np.mean(sc))

def wbo_cimp_optimize(X:np.ndarray,y:np.ndarray,cls:np.ndarray,pc:str,nf:int,seed:int,sc:dict,wc:WboConfig)->Dict[str,object]:
    nc=len(cls)
    if pc not in list(cls):raise ValueError(f"priority_class '{pc}' not in classes")
    pi=int(np.where(cls==pc)[0][0]);dims,names=_space(sc,nc)
    best,best_x,ni=-1.0,None,0
    @use_named_args(dims)
    def obj(**p):
        nonlocal best,best_x,ni
        ap=float(p.pop("alpha_p"));betas=np.array([p.pop(f"beta_{i}")for i in range(nc-1)],dtype=float)
        cw=parameterize_class_weights(pi,betas,ap);xp=p.copy();xp["n_estimators"]=int(xp["n_estimators"]);xp["max_depth"]=int(xp["max_depth"])
        s=cv_wfi_score(X,y,cls,pc,cw,xp,nf,seed)
        if s>best+1e-6:best,best_x,ni=s,(ap,betas,xp,cw),0
        else:ni+=1
        return-s
    res=gp_minimize(obj,dimensions=dims,n_calls=wc.bo_budget,n_initial_points=wc.n_initial,acq_func=wc.acq_func,random_state=seed,verbose=False)
    if best_x is None:raise RuntimeError("Optimization failed")
    ap,betas,xp,cw=best_x
    return{"best_cv_wfi":best,"best_class_weights":cw,"best_xgb_params":xp,"res":res}
