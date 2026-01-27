from __future__ import annotations
from typing import Optional
import numpy as np
from sklearn.feature_selection import mutual_info_classif,mutual_info_regression

def mi_weights(X:np.ndarray,y:np.ndarray,discrete_features:Optional[np.ndarray]=None,task:str="classif",random_state:int=0)->np.ndarray:
    X,y=np.asarray(X),np.asarray(y)
    mi=(mutual_info_classif if task=="classif"else mutual_info_regression)(X,y,discrete_features=discrete_features,random_state=random_state)
    mi=np.maximum(mi,0.0)
    return mi/mi.sum()if mi.sum()>0 else np.ones(X.shape[1])/max(1,X.shape[1])
