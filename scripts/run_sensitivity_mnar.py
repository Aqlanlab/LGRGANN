from __future__ import annotations
import argparse,copy
import pandas as pd
from lgrgann_wbocimp.config import load_yaml,ensure_dir
from lgrgann_wbocimp.eval_runner import run_dimm_pipeline

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--config",required=True)
    args=ap.parse_args();cfg=load_yaml(args.config);od=ensure_dir("outputs")
    mechs=["MAR","MNAR_CLASS","MNAR_HARDNESS","MNAR_HYBRID"];all_res=[]
    for m in mechs:
        c2=copy.deepcopy(cfg);c2["evaluation"]["missing_mechanism"]=m
        df=run_dimm_pipeline(c2);df["mechanism"]=m;all_res.append(df)
    res=pd.concat(all_res,ignore_index=True);res.to_csv(od/"mnar_sensitivity_runs.csv",index=False)
    agg=res.groupby("mechanism").agg({c:["mean","std"]for c in res.columns if c not in["fold","seed","stage1_refiner","mechanism"]}).reset_index()
    agg.to_csv(od/"mnar_sensitivity_aggregate.csv",index=False)
    print(f"Saved to {od}/")

if __name__=="__main__":main()
