from __future__ import annotations
from dataclasses import dataclass,field
from typing import List,Optional,Tuple
from pathlib import Path
import numpy as np
import pandas as pd

@dataclass
class DatasetSpec:
    path:str;format:str="xlsx";label_col:str="Disposition";time_col:str="Date"
    id_col:Optional[str]=None;drop_cols:List[str]=field(default_factory=list)
    categorical_cols:Optional[List[str]]=None;numeric_cols:Optional[List[str]]=None
    na_values:List[str]=field(default_factory=lambda:["","NA","NaN","null"])

def load_table(s:DatasetSpec)->pd.DataFrame:
    p,f=Path(s.path),s.format.lower()
    r={".csv":lambda:pd.read_csv(p,na_values=s.na_values),".xlsx":lambda:pd.read_excel(p,na_values=s.na_values),
       ".xls":lambda:pd.read_excel(p,na_values=s.na_values),".parquet":lambda:pd.read_parquet(p)}
    df=r.get(f".{f}",r.get(p.suffix.lower(),lambda:pd.read_csv(p,na_values=s.na_values)))()
    return df.drop(columns=[c for c in s.drop_cols if c in df.columns],errors="ignore")

def infer_columns(df:pd.DataFrame,lc:str,tc:str,cat:Optional[List[str]],num:Optional[List[str]])->Tuple[List[str],List[str]]:
    exc={lc,tc};cols=[c for c in df.columns if c not in exc]
    if cat is not None and num is not None:return[c for c in num if c in cols],[c for c in cat if c in cols]
    nc=[c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    cc=[c for c in cols if c not in nc and(pd.api.types.is_categorical_dtype(df[c])or pd.api.types.is_object_dtype(df[c]))]
    return nc,cc

def sort_by_time(df:pd.DataFrame,tc:str)->pd.DataFrame:
    return df.sort_values(tc).reset_index(drop=True)if tc in df.columns else df
