from __future__ import annotations
import argparse
from lgrgann_wbocimp.config import load_yaml,ensure_dir
from lgrgann_wbocimp.eval_runner import run_dimm_pipeline

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--config",required=True)
    args=ap.parse_args();cfg=load_yaml(args.config);od=ensure_dir("outputs")
    df=run_dimm_pipeline(cfg);df.to_csv(od/"dimm_results.csv",index=False)
    agg=df.groupby("fold").agg({c:["mean","std"]for c in df.columns if c not in["fold","seed","stage1_refiner"]}).reset_index()
    agg.to_csv(od/"dimm_aggregate_by_fold.csv",index=False)
    ov=df[[c for c in df.columns if c not in["fold","seed","stage1_refiner","k"]]].agg(["mean","std"])
    ov.to_csv(od/"dimm_overall.csv")
    print(f"Results saved to {od}/")

if __name__=="__main__":main()
