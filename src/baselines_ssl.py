from __future__ import annotations
import numpy as np
from sklearn.semi_supervised import SelfTrainingClassifier,LabelSpreading
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

def self_training_baseline(Xl:np.ndarray,yl:np.ndarray,Xu:np.ndarray,seed:int,nc:int)->np.ndarray:
    b=LogisticRegression(max_iter=2000,n_jobs=-1);c=SelfTrainingClassifier(base_estimator=b)
    c.fit(np.vstack([Xl,Xu]),np.concatenate([yl,np.full(len(Xu),-1,dtype=int)]))
    return c.predict(Xu)

def label_spreading_baseline(Xa:np.ndarray,ya:np.ndarray,seed:int)->np.ndarray:
    ls=LabelSpreading(kernel="rbf",gamma=20,max_iter=50);ls.fit(Xa,ya)
    return ls.transduction_

def xgb_pseudo_labeling(Xl:np.ndarray,yl:np.ndarray,Xu:np.ndarray,seed:int,nc:int)->np.ndarray:
    m=XGBClassifier(objective="multi:softprob",num_class=nc,eval_metric="mlogloss",n_estimators=800,max_depth=6,learning_rate=0.05,subsample=0.9,colsample_bytree=0.9,random_state=seed,n_jobs=-1)
    m.fit(Xl,yl)
    return m.predict(Xu)
