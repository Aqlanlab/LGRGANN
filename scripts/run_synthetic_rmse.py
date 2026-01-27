from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from lgrgann_wbocimp.config import load_yaml,ensure_dir
from lgrgann_wbocimp.synthetic import make_synthetic_dataset,mask_entries_mar,rmse_on_mask,mice_impute,missforest_impute
from lgrgann_wbocimp.reconstruction_lgrgann import Stage1Config,run_stage1
from lgrgann_wbocimp.baselines_reconstruction import iterative_knn_label_reconstruction,mi_knn_label_reconstruction,grey_knn_label_reconstruction,fwgknn_label_reconstruction,cgknn_label_reconstruction
from lgrgann_wbocimp.data_splits import inner_stratified_splits
from lgrgann_wbocimp.utils_seeding import seed_everything

SD={"SD1":{"n":1520,"d":14,"classes":4,"feature_type":"continuous","base_missing_rate":0.0},"SD2":{"n":4315,"d":25,"classes":5,"feature_type":"categorical","base_missing_rate":0.0414},"SD3":{"n":2556,"d":19,"classes":3,"feature_type":"mixed","base_missing_rate":0.0539}}

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--config",required=True);ap.add_argument("--output",default="outputs")
    args=ap.parse_args();cfg=load_yaml(args.config);od=ensure_dir(args.output)
    reps=int(cfg.get("synthetic",{}).get("replications",50));rho=float(cfg.get("synthetic",{}).get("rho_copula",0.3))
    seeds=cfg.get("synthetic",{}).get("seeds",list(range(100)));mrs=cfg.get("synthetic",{}).get("missing_rates",[0.05,0.10,0.20])
    k=int(cfg.get("synthetic",{}).get("k",7));s1c=cfg.get("stage1",{});rows=[]
    for dn,sp in cfg.get("datasets",SD).items():
        print(f"\n{'='*60}\n{dn}\n{'='*60}")
        for mr in mrs:
            print(f"  Missing: {mr*100:.0f}%")
            for rep in tqdm(range(reps),desc=f"  {dn}@{mr*100:.0f}%"):
                seed=int(seeds[rep%len(seeds)])+10000*rep;seed_everything(seed)
                df,y=make_synthetic_dataset(dn,sp,rho,seed)
                Xt=df.values.astype(float)if sp["feature_type"]=="continuous"else pd.get_dummies(df).values.astype(float)
                cls=np.unique(y);ys=np.array([str(yi)for yi in y]);clss=np.array([str(c)for c in cls])
                Xm,om=mask_entries_mar(Xt,mr,seed+1)
                n=len(y);m=int(round(mr*n));rng=np.random.default_rng(seed+999)
                mi=rng.choice(n,size=m,replace=False);lm=np.ones(n,dtype=bool);lm[mi]=False
                yte=np.array([int(np.where(clss==yi)[0][0])for yi in ys],dtype=float)
                def lrmse(yr):ye=np.array([int(np.where(clss==yi)[0][0])for yi in yr],dtype=float);return float(np.sqrt(np.mean((yte[~lm]-ye[~lm])**2)))
                try:rows.append({"dataset":dn,"missing_rate":mr,"rep":rep,"method":"MICE","rmse":rmse_on_mask(Xt,mice_impute(Xm.copy(),seed),om)})
                except:rows.append({"dataset":dn,"missing_rate":mr,"rep":rep,"method":"MICE","rmse":np.nan})
                try:rows.append({"dataset":dn,"missing_rate":mr,"rep":rep,"method":"MissForest","rmse":rmse_on_mask(Xt,missforest_impute(Xm.copy(),seed),om)})
                except:rows.append({"dataset":dn,"missing_rate":mr,"rep":rep,"method":"MissForest","rmse":np.nan})
                try:rows.append({"dataset":dn,"missing_rate":mr,"rep":rep,"method":"IKNN","rmse":lrmse(iterative_knn_label_reconstruction(Xt,ys.copy(),lm,clss,k,10,seed))})
                except:rows.append({"dataset":dn,"missing_rate":mr,"rep":rep,"method":"IKNN","rmse":np.nan})
                try:rows.append({"dataset":dn,"missing_rate":mr,"rep":rep,"method":"MI-KNN","rmse":lrmse(mi_knn_label_reconstruction(Xt,ys.copy(),lm,clss,k,seed))})
                except:rows.append({"dataset":dn,"missing_rate":mr,"rep":rep,"method":"MI-KNN","rmse":np.nan})
                try:rows.append({"dataset":dn,"missing_rate":mr,"rep":rep,"method":"GKNN","rmse":lrmse(grey_knn_label_reconstruction(Xt,ys.copy(),lm,clss,k,0.5,seed))})
                except:rows.append({"dataset":dn,"missing_rate":mr,"rep":rep,"method":"GKNN","rmse":np.nan})
                try:rows.append({"dataset":dn,"missing_rate":mr,"rep":rep,"method":"FWGKNN","rmse":lrmse(fwgknn_label_reconstruction(Xt,ys.copy(),lm,clss,k,0.5,seed))})
                except:rows.append({"dataset":dn,"missing_rate":mr,"rep":rep,"method":"FWGKNN","rmse":np.nan})
                try:rows.append({"dataset":dn,"missing_rate":mr,"rep":rep,"method":"CGKNN","rmse":lrmse(cgknn_label_reconstruction(Xt,ys.copy(),lm,clss,k,0.5,seed))})
                except:rows.append({"dataset":dn,"missing_rate":mr,"rep":rep,"method":"CGKNN","rmse":np.nan})
                try:
                    ysim=ys.astype(object).copy();ysim[~lm]=np.nan
                    c2={kk:v for kk,v in s1c.items()if kk in Stage1Config.__dataclass_fields__};c2["method"]="LGRGANN"
                    cfg2=Stage1Config(**c2);yil=np.array([int(np.where(clss==yi)[0][0])for yi in ys[lm]],dtype=int)
                    isp=list(inner_stratified_splits(yil,5,seed))
                    s1=run_stage1(Xt,ysim,lm,clss,seed,cfg2,isp)
                    rows.append({"dataset":dn,"missing_rate":mr,"rep":rep,"method":"LGRGANN","rmse":lrmse(s1["y_recon"])})
                except Exception as e:print(f"LGRGANN err: {e}");rows.append({"dataset":dn,"missing_rate":mr,"rep":rep,"method":"LGRGANN","rmse":np.nan})
    res=pd.DataFrame(rows);res.to_csv(od/"synthetic_rmse_runs.csv",index=False)
    agg=res.groupby(["dataset","missing_rate","method"])["rmse"].agg(["mean","std"]).reset_index()
    agg["mean_std"]=agg.apply(lambda r:f"{r['mean']:.4f}±{r['std']:.4f}"if not np.isnan(r["mean"])else"N/A",axis=1)
    for dn in res["dataset"].unique():
        print(f"\n{'='*80}\nRMSE: {dn}\n{'='*80}")
        da=agg[agg["dataset"]==dn];pv=da.pivot(index="missing_rate",columns="method",values="mean_std")
        co=["MICE","MissForest","IKNN","MI-KNN","GKNN","FWGKNN","CGKNN","LGRGANN"];pv=pv[[c for c in co if c in pv.columns]]
        pv.index=[f"{int(r*100)}%"for r in pv.index];print(pv.to_string());pv.to_csv(od/f"table_rmse_{dn}.csv")
    agg.to_csv(od/"synthetic_rmse_aggregate.csv",index=False);print(f"\nSaved to {od}/")

if __name__=="__main__":main()
