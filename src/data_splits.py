from __future__ import annotations
from typing import Iterator,Tuple
import numpy as np
from sklearn.model_selection import TimeSeriesSplit,StratifiedKFold

def outer_time_series_splits(n:int,k:int)->Iterator[Tuple[np.ndarray,np.ndarray]]:
    return((tr,te)for tr,te in TimeSeriesSplit(n_splits=k).split(np.zeros((n,1))))

def inner_stratified_splits(y:np.ndarray,n_splits:int,seed:int)->Iterator[Tuple[np.ndarray,np.ndarray]]:
    return((tr,va)for tr,va in StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=seed).split(np.arange(len(y)),y))
