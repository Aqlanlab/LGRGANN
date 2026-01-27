from __future__ import annotations
from typing import Dict,List
import numpy as np
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix

def per_class_f1(yt:np.ndarray,yp:np.ndarray,labels:List[int])->np.ndarray:return f1_score(yt,yp,labels=labels,average=None,zero_division=0)
def macro_f1(yt:np.ndarray,yp:np.ndarray,labels:List[int])->float:return float(f1_score(yt,yp,labels=labels,average="macro",zero_division=0))
def accuracy(yt:np.ndarray,yp:np.ndarray)->float:return float(accuracy_score(yt,yp))

def weighted_f1_custom(yt:np.ndarray,yp:np.ndarray,cw:np.ndarray,labels:List[int])->float:
    f1s=per_class_f1(yt,yp,labels);cw=np.asarray(cw,dtype=float);cw=cw/(cw.sum()+1e-12)
    return float(np.sum(cw*f1s))

def expected_cost_multiclass(yt:np.ndarray,yp:np.ndarray,cm:np.ndarray,labels:List[int])->float:
    C=confusion_matrix(yt,yp,labels=labels);t=C.sum()
    return 0.0 if t==0 else float((C*cm).sum()/t)

def expected_cost_index(c:float,bc:float)->float:return float("nan")if bc<=0 else float(c/bc)

def expected_cost_scrap_binary_per_1000(yts:np.ndarray,yps:np.ndarray,fsc:float=1.0,msc:float=5.0)->float:
    yts,yps=np.asarray(yts).astype(int),np.asarray(yps).astype(int)
    fp,fn=int(((yts==0)&(yps==1)).sum()),int(((yts==1)&(yps==0)).sum())
    n=len(yts)
    return 0.0 if n==0 else float((fsc*fp+msc*fn)*(1000.0/n))
