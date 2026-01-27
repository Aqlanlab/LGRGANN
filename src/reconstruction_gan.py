from __future__ import annotations
from dataclasses import dataclass
from typing import List,Tuple,Optional,Callable
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,TensorDataset
from .utils import to_device

class _MLP(nn.Module):
    def __init__(s,i:int,h:List[int],o:int,act:Optional[str]=None,dp:float=0.0,bn:bool=False):
        super().__init__()
        L,d=[],i
        for x in h:L+=[nn.Linear(d,x)]+([nn.BatchNorm1d(x)]if bn else[])+[nn.LeakyReLU(0.2)]+([nn.Dropout(dp)]if dp>0 else[]);d=x
        L.append(nn.Linear(d,o));s.net,s.act=nn.Sequential(*L),act
    def forward(s,x):y=s.net(x);return torch.tanh(y)if s.act=="tanh"else(torch.sigmoid(y)if s.act=="sigmoid"else y)

@dataclass
class GanConfig:
    z_dim:int=16;hidden:Tuple[int,int]=(128,128);lr_g:float=2e-4;lr_d:float=5e-4;batch_size:int=128
    epochs:int=300;n_critic:int=2;early_stop_patience:int=25;eval_every:int=5;val_mask_frac:float=0.15
    device:str="cpu";label_smoothing:float=0.1;grad_clip:float=1.0;dropout_g:float=0.0;dropout_d:float=0.1

class ResidualCGAN:
    def __init__(s,nc:int,cfg:GanConfig):
        s.nc,s.cfg=nc,cfg
        s.G=_MLP(nc+cfg.z_dim,list(cfg.hidden),nc,None,cfg.dropout_g).to(cfg.device)
        s.D=_MLP(nc*2,list(cfg.hidden),1,None,cfg.dropout_d).to(cfg.device)
        s.og,s.od=torch.optim.Adam(s.G.parameters(),lr=cfg.lr_g,betas=(0.5,0.999)),torch.optim.Adam(s.D.parameters(),lr=cfg.lr_d,betas=(0.5,0.999))
        s.bce=nn.BCEWithLogitsLoss()
    def _z(s,n:int)->torch.Tensor:return torch.randn(n,s.cfg.z_dim,device=s.cfg.device)
    def fit(s,phi:np.ndarray,yt:np.ndarray,cls:np.ndarray,vm:Optional[np.ndarray]=None,vpp:Callable=None,vyt:Optional[np.ndarray]=None,seed:int=0)->dict:
        torch.manual_seed(seed);np.random.seed(seed)
        pt=torch.tensor(phi,dtype=torch.float32)
        yi=np.array([int(np.where(cls==y)[0][0])for y in yt],dtype=int)
        r=torch.tensor(np.eye(s.nc)[yi]-phi,dtype=torch.float32)
        dl=DataLoader(TensorDataset(pt,r),batch_size=s.cfg.batch_size,shuffle=True,drop_last=False)
        bs,bst,pc,H=(-1.0,None,0,{"epoch":[],"d_loss":[],"g_loss":[],"val_macro_f1":[]})
        rl,fl=1.0-s.cfg.label_smoothing,s.cfg.label_smoothing
        for ep in range(1,s.cfg.epochs+1):
            s.G.train();s.D.train();dL,gL=[],[]
            for pb,rb in dl:
                pb,rb,B=to_device(pb,s.cfg.device),to_device(rb,s.cfg.device),len(pb)
                for _ in range(s.cfg.n_critic):
                    s.od.zero_grad(set_to_none=True);z=s._z(B)
                    with torch.no_grad():rf=s.G(torch.cat([pb,z],1))
                    lr,lf=s.D(torch.cat([pb,rb],1)),s.D(torch.cat([pb,rf],1))
                    ld=s.bce(lr,torch.full_like(lr,rl))+s.bce(lf,torch.full_like(lf,fl))
                    ld.backward()
                    if s.cfg.grad_clip>0:torch.nn.utils.clip_grad_norm_(s.D.parameters(),s.cfg.grad_clip)
                    s.od.step();dL.append(ld.item())
                s.og.zero_grad(set_to_none=True);z=s._z(B)
                rf=s.G(torch.cat([pb,z],1));lg=s.bce(s.D(torch.cat([pb,rf],1)),torch.ones(B,1,device=s.cfg.device))
                lg.backward()
                if s.cfg.grad_clip>0:torch.nn.utils.clip_grad_norm_(s.G.parameters(),s.cfg.grad_clip)
                s.og.step();gL.append(lg.item())
            vf1=None
            if vm is not None and vpp is not None and vyt is not None and ep%s.cfg.eval_every==0:
                vf1=_mf1(vyt,s.predict_labels(vpp(),cls,seed),cls)
                if vf1>bs+1e-6:bs,bst,pc=vf1,{"G":{k:v.cpu().clone()for k,v in s.G.state_dict().items()},"D":{k:v.cpu().clone()for k,v in s.D.state_dict().items()}},0
                else:pc+=1
                if pc>=s.cfg.early_stop_patience:break
            H["epoch"].append(ep);H["d_loss"].append(float(np.mean(dL)if dL else 0));H["g_loss"].append(float(np.mean(gL)if gL else 0));H["val_macro_f1"].append(vf1 if vf1 else float("nan"))
        if bst:s.G.load_state_dict(bst["G"]);s.D.load_state_dict(bst["D"])
        return{"best_val_macro_f1":bs,"history":H,"final_epoch":ep}
    @torch.no_grad()
    def predict_scores(s,phi:np.ndarray,seed:int=0)->np.ndarray:
        torch.manual_seed(seed);s.G.eval()
        pt=torch.tensor(phi,dtype=torch.float32,device=s.cfg.device)
        return torch.softmax(pt+s.G(torch.cat([pt,s._z(len(pt))],1)),dim=1).cpu().numpy()
    @torch.no_grad()
    def predict_labels(s,phi:np.ndarray,cls:np.ndarray,seed:int=0)->np.ndarray:return cls[np.argmax(s.predict_scores(phi,seed),axis=1)]

def _mf1(yt:np.ndarray,yp:np.ndarray,cls:np.ndarray)->float:
    from sklearn.metrics import f1_score
    L=list(range(len(cls)))
    return float(f1_score([int(np.where(cls==y)[0][0])for y in yt],[int(np.where(cls==y)[0][0])for y in yp],labels=L,average="macro",zero_division=0))
