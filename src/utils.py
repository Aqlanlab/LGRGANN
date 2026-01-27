from __future__ import annotations
import torch

def to_device(t,d:str="cpu"):return t.to(d)if isinstance(t,torch.Tensor)else torch.tensor(t,device=d)
