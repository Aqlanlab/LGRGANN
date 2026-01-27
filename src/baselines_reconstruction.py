from __future__ import annotations
from typing import Dict,List
import numpy as np
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.semi_supervised import SelfTrainingClassifier,LabelSpreading
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score,accuracy_score
from xgboost import XGBClassifier
from .reconstruction_mi import mi_weights

def mice_label_reconstruction(X:np.ndarray,y:np.ndarray,lm:np.ndarray,cls:np.ndarray,seed:int=0,mi:int=20)->np.ndarray:
    ye=np.full(len(y),np.nan,dtype=float)
    for i,yi in enumerate(y):
        if lm[i]:ye[i]=float(np.where(cls==yi)[0][0])
    Xy=np.column_stack([X,ye])
    Xi=IterativeImputer(random_state=seed,max_iter=mi,sample_posterior=False,initial_strategy="most_frequent").fit_transform(Xy)
    yr=cls[np.clip(np.round(Xi[:,-1]).astype(int),0,len(cls)-1)]
    yo=y.copy();yo[~lm]=yr[~lm]
    return yo

def missforest_label_reconstruction(X:np.ndarray,y:np.ndarray,lm:np.ndarray,cls:np.ndarray,seed:int=0,ne:int=100,mi:int=10)->np.ndarray:
    ye=np.full(len(y),np.nan,dtype=float)
    for i,yi in enumerate(y):
        if lm[i]:ye[i]=float(np.where(cls==yi)[0][0])
    rf=RandomForestRegressor(n_estimators=ne,random_state=seed,n_jobs=-1,max_depth=None)
    Xi=IterativeImputer(estimator=rf,random_state=seed,max_iter=mi,sample_posterior=False,initial_strategy="most_frequent").fit_transform(np.column_stack([X,ye]))
    yr=cls[np.clip(np.round(Xi[:,-1]).astype(int),0,len(cls)-1)]
    yo=y.copy();yo[~lm]=yr[~lm]
    return yo

def knn_label_reconstruction(X:np.ndarray,y:np.ndarray,lm:np.ndarray,cls:np.ndarray,k:int=5,seed:int=0)->np.ndarray:
    Xl,yl,Xm=X[lm],y[lm],X[~lm]
    yil=np.array([int(np.where(cls==yi)[0][0])for yi in yl])
    knn=KNeighborsClassifier(n_neighbors=min(k,len(Xl)),weights="uniform");knn.fit(Xl,yil)
    yo=y.copy();yo[~lm]=cls[knn.predict(Xm)]
    return yo

def iterative_knn_label_reconstruction(X:np.ndarray,y:np.ndarray,lm:np.ndarray,cls:np.ndarray,k:int=5,mi:int=10,seed:int=0)->np.ndarray:
    yc,mm=y.copy(),~lm
    v,c=np.unique(y[lm],return_counts=True);yc[mm]=v[np.argmax(c)]
    yi=np.array([int(np.where(cls==yi)[0][0])for yi in yc])
    prev=None
    for _ in range(mi):
        knn=KNeighborsClassifier(n_neighbors=min(k,int(lm.sum())),weights="distance");knn.fit(X[lm],yi[lm])
        yi[mm]=knn.predict(X[mm])
        if prev is not None and np.array_equal(yi[mm],prev):break
        prev=yi[mm].copy()
    yo=y.copy();yo[mm]=cls[yi[mm]]
    return yo

def mi_knn_label_reconstruction(X:np.ndarray,y:np.ndarray,lm:np.ndarray,cls:np.ndarray,k:int=5,seed:int=0)->np.ndarray:
    tau=mi_weights(X[lm],y[lm],task="classif",random_state=seed)
    Xw=X*np.sqrt(tau);Xl,yl,Xm=Xw[lm],y[lm],Xw[~lm]
    yil=np.array([int(np.where(cls==yi)[0][0])for yi in yl])
    knn=KNeighborsClassifier(n_neighbors=min(k,len(Xl)),weights="distance");knn.fit(Xl,yil)
    yo=y.copy();yo[~lm]=cls[knn.predict(Xm)]
    return yo

def grey_knn_label_reconstruction(X:np.ndarray,y:np.ndarray,lm:np.ndarray,cls:np.ndarray,k:int=5,rho:float=0.5,seed:int=0)->np.ndarray:
    eps,Xl,yl,Xm=1e-8,X[lm],y[lm],X[~lm]
    yil,nc,p=np.array([int(np.where(cls==yi)[0][0])for yi in yl]),len(cls),X.shape[1]
    tau=np.ones(p)/p;ypi=[]
    for xq in Xm:
        D=np.empty(len(Xl))
        for j,xr in enumerate(Xl):
            d=np.abs(xq-xr);dmin,dmax=d.min(),d.max()
            grc=(dmin+rho*dmax)/(d+rho*dmax+eps);grg=np.sum(tau*grc);D[j]=1.0-grg
        nn=np.argsort(D)[:k];W=1.0/(D[nn]**2+eps)
        sc=np.zeros(nc)
        for idx,w in zip(nn,W):sc[yil[idx]]+=w
        ypi.append(np.argmax(sc))
    yo=y.copy();yo[~lm]=cls[np.array(ypi)]
    return yo

def fwgknn_label_reconstruction(X:np.ndarray,y:np.ndarray,lm:np.ndarray,cls:np.ndarray,k:int=5,rho:float=0.5,seed:int=0)->np.ndarray:
    eps=1e-8;tau=mi_weights(X[lm],y[lm],task="classif",random_state=seed)
    Xl,yl,Xm=X[lm],y[lm],X[~lm]
    yil,nc=np.array([int(np.where(cls==yi)[0][0])for yi in yl]),len(cls);ypi=[]
    for xq in Xm:
        D=np.empty(len(Xl))
        for j,xr in enumerate(Xl):
            d=np.abs(xq-xr);dmin,dmax=d.min(),d.max()
            grc=(dmin+rho*dmax)/(d+rho*dmax+eps);D[j]=1.0-np.sum(tau*grc)
        nn=np.argsort(D)[:k];W=1.0/(D[nn]**2+eps)
        sc=np.zeros(nc)
        for idx,w in zip(nn,W):sc[yil[idx]]+=w
        ypi.append(np.argmax(sc))
    yo=y.copy();yo[~lm]=cls[np.array(ypi)]
    return yo

def cgknn_label_reconstruction(X:np.ndarray,y:np.ndarray,lm:np.ndarray,cls:np.ndarray,k:int=5,rho:float=0.5,seed:int=0)->np.ndarray:
    eps=1e-8;tau=mi_weights(X[lm],y[lm],task="classif",random_state=seed)
    Xl,yl,Xm=X[lm],y[lm],X[~lm]
    yil,nc=np.array([int(np.where(cls==yi)[0][0])for yi in yl]),len(cls)
    cc=np.bincount(yil,minlength=nc).astype(float);cw=1.0/(cc+eps);cw=cw/cw.sum();ypi=[]
    for xq in Xm:
        D=np.empty(len(Xl))
        for j,xr in enumerate(Xl):
            d=np.abs(xq-xr);dmin,dmax=d.min(),d.max()
            grc=(dmin+rho*dmax)/(d+rho*dmax+eps);D[j]=1.0-np.sum(tau*grc)
        nn=np.argsort(D)[:k];W=1.0/(D[nn]**2+eps)
        sc=np.zeros(nc)
        for idx,w in zip(nn,W):sc[yil[idx]]+=w*cw[yil[idx]]
        ypi.append(np.argmax(sc))
    yo=y.copy();yo[~lm]=cls[np.array(ypi)]
    return yo

def self_training_label_reconstruction(X:np.ndarray,y:np.ndarray,lm:np.ndarray,cls:np.ndarray,seed:int=0,th:float=0.75)->np.ndarray:
    yi=np.full(len(y),-1,dtype=int)
    for i,yv in enumerate(y):
        if lm[i]:yi[i]=int(np.where(cls==yv)[0][0])
    c=SelfTrainingClassifier(base_estimator=LogisticRegression(max_iter=2000,n_jobs=-1,random_state=seed),threshold=th);c.fit(X,yi)
    yo=y.copy();yo[~lm]=cls[c.predict(X)[~lm]]
    return yo

def label_spreading_reconstruction(X:np.ndarray,y:np.ndarray,lm:np.ndarray,cls:np.ndarray,seed:int=0,gamma:float=20.0)->np.ndarray:
    yi=np.full(len(y),-1,dtype=int)
    for i,yv in enumerate(y):
        if lm[i]:yi[i]=int(np.where(cls==yv)[0][0])
    ls=LabelSpreading(kernel="rbf",gamma=gamma,max_iter=50);ls.fit(X,yi)
    yo=y.copy();yo[~lm]=cls[ls.transduction_[~lm]]
    return yo

def xgboost_pseudo_labeling(X:np.ndarray,y:np.ndarray,lm:np.ndarray,cls:np.ndarray,seed:int=0,ne:int=500)->np.ndarray:
    Xl,yl,Xm=X[lm],y[lm],X[~lm]
    yil=np.array([int(np.where(cls==yi)[0][0])for yi in yl])
    m=XGBClassifier(objective="multi:softprob",num_class=len(cls),eval_metric="mlogloss",n_estimators=ne,max_depth=6,learning_rate=0.1,subsample=0.9,colsample_bytree=0.9,random_state=seed,n_jobs=-1)
    m.fit(Xl,yil)
    yo=y.copy();yo[~lm]=cls[m.predict(Xm)]
    return yo

RECONSTRUCTION_METHODS={"MICE":mice_label_reconstruction,"MissForest":missforest_label_reconstruction,"kNN":knn_label_reconstruction,"IKNN":iterative_knn_label_reconstruction,"MI-kNN":mi_knn_label_reconstruction,"GKNN":grey_knn_label_reconstruction,"FWGKNN":fwgknn_label_reconstruction,"CGKNN":cgknn_label_reconstruction,"SelfTraining":self_training_label_reconstruction,"LabelSpreading":label_spreading_reconstruction,"XGBoost-PL":xgboost_pseudo_labeling}

def run_baseline_reconstruction(method:str,X:np.ndarray,y:np.ndarray,lm:np.ndarray,cls:np.ndarray,k:int=5,seed:int=0,**kw)->np.ndarray:
    if method not in RECONSTRUCTION_METHODS:raise ValueError(f"Unknown method: {method}")
    f=RECONSTRUCTION_METHODS[method]
    return f(X,y,lm,cls,k=k,seed=seed,**kw)if method in["kNN","IKNN","MI-kNN","GKNN","FWGKNN","CGKNN"]else f(X,y,lm,cls,seed=seed,**kw)

def evaluate_reconstruction(yt:np.ndarray,yp:np.ndarray,cls:np.ndarray,mm:np.ndarray)->Dict[str,float]:
    ytm,ypm=yt[mm],yp[mm]
    yti=np.array([int(np.where(cls==yi)[0][0])for yi in ytm])
    ypi=np.array([int(np.where(cls==yi)[0][0])for yi in ypm])
    L=list(range(len(cls)))
    m={"accuracy":accuracy_score(yti,ypi),"macro_f1":f1_score(yti,ypi,labels=L,average="macro",zero_division=0)}
    pcf1=f1_score(yti,ypi,labels=L,average=None,zero_division=0)
    for i,c in enumerate(cls):m[f"f1_{c}"]=float(pcf1[i])
    return m
