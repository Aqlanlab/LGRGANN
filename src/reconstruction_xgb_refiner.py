from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
from xgboost import XGBClassifier

@dataclass
class XgbRefinerConfig:
    n_estimators:int=800;max_depth:int=6;learning_rate:float=0.05;subsample:float=0.9
    colsample_bytree:float=0.9;random_state:int=0

class XgbResidualRefiner:
    def __init__(s,n_classes:int,cfg:XgbRefinerConfig):s.nc,s.cfg,s.m=n_classes,cfg,None
    def fit(s,phi:np.ndarray,yt:np.ndarray,classes:np.ndarray)->None:
        yi=np.array([int(np.where(classes==y)[0][0])for y in yt],dtype=int)
        s.m=XGBClassifier(objective="multi:softprob",num_class=s.nc,eval_metric="mlogloss",n_estimators=s.cfg.n_estimators,max_depth=s.cfg.max_depth,learning_rate=s.cfg.learning_rate,subsample=s.cfg.subsample,colsample_bytree=s.cfg.colsample_bytree,random_state=s.cfg.random_state,n_jobs=-1)
        s.m.fit(phi,yi)
    def predict_scores(s,phi:np.ndarray)->np.ndarray:return s.m.predict_proba(phi)
