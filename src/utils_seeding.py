from __future__ import annotations
import os,random
import numpy as np

def seed_everything(s:int=0)->None:
    random.seed(s);np.random.seed(s);os.environ["PYTHONHASHSEED"]=str(s)
    try:import torch;torch.manual_seed(s);torch.cuda.manual_seed_all(s)
    except ImportError:pass
