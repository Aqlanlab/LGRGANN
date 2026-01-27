from __future__ import annotations
from functools import reduce
from typing import Optional,Tuple
import numpy as np

def _grc(a:float,b:float,t:str="categorical",tol:float=1e-6)->float:
    return(1.0 if a==b else 0.0)if t=="categorical"else(1.0 if abs(a-b)<=tol else np.exp(-abs(a-b)))

def _grg(xa:np.ndarray,xb:np.ndarray,tau:np.ndarray,ft:Optional[np.ndarray]=None,rho:float=0.5,eps:float=1e-8,tol:float=1e-6)->float:
    xa,xb,p=np.asarray(xa),np.asarray(xb),len(xa)
    ft=np.array(["categorical"]*p)if ft is None else ft
    return float(np.sum(tau*np.array([_grc(xa[j],xb[j],ft[j],tol)for j in range(p)])))

def _d(g:float)->float:return 1.0-g

def knn_predict_scores(Xr:np.ndarray,yr:np.ndarray,Xq:np.ndarray,cls:np.ndarray,k:int,tau:np.ndarray,rho:float=0.5,eps:float=1e-8,ft:Optional[np.ndarray]=None)->np.ndarray:
    Xr,yr,Xq=np.asarray(Xr),np.asarray(yr),np.asarray(Xq)
    nq,nr,nc=Xq.shape[0],len(Xr),len(cls)
    yri=np.array([int(np.where(cls==c)[0][0])for c in yr],dtype=int)
    S=np.zeros((nq,nc),dtype=float)
    for i in range(nq):
        D=np.array([_d(_grg(Xq[i],Xr[j],tau,ft,rho,eps))for j in range(nr)])
        nn=np.argsort(D)[:k]
        W=1.0/(D[nn]**2+eps)
        for idx,w in zip(nn,W):S[i,yri[idx]]+=w
        S[i]=S[i]/S[i].sum()if S[i].sum()>0 else 1.0/nc
    return S

def iterative_reconstruct_labels(X:np.ndarray,y:np.ndarray,cls:np.ndarray,lm:np.ndarray,k:int,tau:np.ndarray,rho:float=0.5,eps:float=1e-8,max_iter:int=10,ft:Optional[np.ndarray]=None)->Tuple[np.ndarray,np.ndarray]:
    y,lm,mm=np.asarray(y).copy(),lm.astype(bool),~lm.astype(bool)
    obs=y[lm]
    if len(obs)==0:raise ValueError("No observed labels")
    v,c=np.unique(obs,return_counts=True)
    y[mm]=v[np.argmax(c)]
    prev,phi=None,np.zeros((len(y),len(cls)),dtype=float)
    for _ in range(max_iter):
        pm=knn_predict_scores(X[lm],y[lm],X[mm],cls,k,tau,rho,eps,ft)
        yn=cls[np.argmax(pm,axis=1)]
        y[mm],phi[mm]=yn,pm
        if prev is not None and np.array_equal(y[mm],prev):break
        prev=y[mm].copy()
    for i in np.where(lm)[0]:
        lm2=lm.copy();lm2[i]=False
        phi[i:i+1]=knn_predict_scores(X[lm2],y[lm2],X[i:i+1],cls,min(k,int(lm2.sum())),tau,rho,eps,ft)if lm2.sum()>0 else 1.0/len(cls)
    return y,phi
