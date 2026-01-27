from __future__ import annotations
import argparse
from lgrgann_wbocimp.config import load_yaml, ensure_dir
from lgrgann_wbocimp.missingness_diagnostic import run_missingness_diagnostic


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    od = ensure_dir("outputs")
    df = run_missingness_diagnostic(cfg)
    df.to_csv(od / "missingness_diagnostic_runs.csv", index=False)
    dv = df.dropna(subset=["roc_auc", "pr_auc"])
    fa = (
        dv.groupby("fold")[["roc_auc", "pr_auc", "prevalence_test"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    fa.to_csv(od / "missingness_diagnostic_by_fold.csv", index=False)
    ov = dv[["roc_auc", "pr_auc", "prevalence_test"]].agg(["mean", "std"]).reset_index()
    ov.to_csv(od / "missingness_diagnostic_overall.csv", index=False)
    print(f"Saved to {od}/")


if __name__ == "__main__":
    main()
