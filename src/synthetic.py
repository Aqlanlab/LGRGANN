from __future__ import annotations
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from .reconstruction_lgrgann import Stage1Config,run_stage1
from .data_splits import inner_stratified_splits

def _erf(x):
    x=np.asarray(x);a1,a2,a3,a4,a5,p=0.254829592,-0.284496736,1.421413741,-1.453152027,1.061405429,0.3275911
    s=np.sign(x);t=1.0/(1.0+p*np.abs(x))
    return s*(1.0-(((((a5*t+a4)*t+a3)*t+a2)*t+a1)*t*np.exp(-x*x)))

def make_synthetic_dataset(nm:str,sp:dict,rho:float,seed:int)->Tuple[pd.DataFrame,np.ndarray]:
    rng=np.random.default_rng(seed);n,d,k,ft=int(sp["n"]),int(sp["d"]),int(sp["classes"]),sp["feature_type"]
    if ft=="continuous":
        X=rng.multivariate_normal(np.zeros(d),(1-rho)*np.eye(d)+rho*np.ones((d,d)),size=n)
        X=(X-X.mean(axis=0))/(X.std(axis=0)+1e-12)
        lg=X[:,:k]+0.2*(X[:,k:2*k]**2);y=np.argmax(lg,axis=1)
        return pd.DataFrame(X,columns=[f"x{i}"for i in range(d)]),y
    if ft=="categorical":
        Z=rng.multivariate_normal(np.zeros(d),(1-rho)*np.eye(d)+rho*np.ones((d,d)),size=n)
        X=np.zeros_like(Z,dtype=int)
        for j in range(d):
            qs=np.quantile(Z[:,j],[0.25,0.5,0.75])
            X[:,j]=(Z[:,j]>qs[0]).astype(int)+(Z[:,j]>qs[1]).astype(int)+(Z[:,j]>qs[2]).astype(int)
        lg=np.zeros((n,k))
        for c in range(k):lg[:,c]=(X[:,c]==(c%4)).astype(float)+0.1*rng.normal(size=n)
        y=np.argmax(lg,axis=1)
        return pd.DataFrame(X,columns=[f"x{i}"for i in range(d)]).astype("category"),y
    if ft=="mixed":
        dc,dcat=d//2,d-d//2
        Xc=rng.multivariate_normal(np.zeros(dc),(1-rho)*np.eye(dc)+rho*np.ones((dc,dc)),size=n)
        Xc=(Xc-Xc.mean(axis=0))/(Xc.std(axis=0)+1e-12)
        Z=rng.multivariate_normal(np.zeros(dcat),(1-rho)*np.eye(dcat)+rho*np.ones((dcat,dcat)),size=n)
        Xd=np.zeros_like(Z,dtype=int)
        for j in range(dcat):
            qs=np.quantile(Z[:,j],[0.33,0.66])
            Xd[:,j]=(Z[:,j]>qs[0]).astype(int)+(Z[:,j]>qs[1]).astype(int)
        lg=0.7*Xc[:,:k]+0.3*(Xd[:,:k]==1).astype(float)
        y=np.argmax(lg,axis=1)
        df=pd.DataFrame({f"c{i}":Xc[:,i]for i in range(dc)})
        for j in range(dcat):df[f"d{j}"]=pd.Series(Xd[:,j]).astype("category")
        return df,y
    raise ValueError(f"Unknown feature_type: {ft}")

def mask_entries_mar(X:np.ndarray,rate:float,seed:int)->Tuple[np.ndarray,np.ndarray]:
    rng=np.random.default_rng(seed);mk=rng.uniform(size=X.shape)>rate
    Xm=X.copy();Xm[~mk]=np.nan
    return Xm,mk

def rmse_on_mask(Xt:np.ndarray,Xi:np.ndarray,om:np.ndarray)->float:
    ms=~om;return 0.0 if ms.sum()==0 else float(np.sqrt(np.mean((Xt[ms]-Xi[ms])**2)))

def mice_impute(X:np.ndarray,seed:int)->np.ndarray:
    return IterativeImputer(random_state=seed,max_iter=20,sample_posterior=False).fit_transform(X)

def missforest_impute(X:np.ndarray,seed:int)->np.ndarray:
    rf=RandomForestRegressor(n_estimators=200,random_state=seed,n_jobs=-1,max_depth=None)
    return IterativeImputer(estimator=rf,random_state=seed,max_iter=10,sample_posterior=False).fit_transform(X)

def lgrgann_impute_columnwise(df:pd.DataFrame,y:np.ndarray,s1c:dict,seed:int)->pd.DataFrame:
    out=df.copy();Xe,ct=[],[]
    for c in out.columns:
        if str(out[c].dtype)in("object","category"):ct.append("cat");Xe.append(out[c].cat.codes.replace(-1,np.nan).to_numpy().astype(float))
        else:ct.append("num");Xe.append(out[c].to_numpy().astype(float))
    Xe=np.column_stack(Xe);mn,mx=np.nanmin(Xe,axis=0),np.nanmax(Xe,axis=0);Xs=(Xe-mn)/(mx-mn+1e-12)
    for j,c in enumerate(out.columns):
        col=Xe[:,j]
        if np.isnan(col).sum()==0:continue
        lm=~np.isnan(col);Xj=np.delete(Xs,j,axis=1)
        if ct[j]=="cat":
            cls=np.array(sorted(np.unique(col[lm]).astype(int).tolist()));yc=col.copy()
            cfg=Stage1Config(**s1c)
            yi=np.array([np.where(cls==int(v))[0][0]for v in yc[lm]],dtype=int)
            inn=list(inner_stratified_splits(yi,5,seed))
            res=run_stage1(Xj,yc,lm,cls,seed,cfg,inn)
            out.loc[~lm,c]=pd.Categorical.from_codes(res["y_recon"][~lm].astype(int),categories=out[c].cat.categories)
        else:
            Xt=Xe.copy();Xt[:,j]=col;Xi=missforest_impute(Xt,seed)
            out[c]=Xi[:,j]
    return out
