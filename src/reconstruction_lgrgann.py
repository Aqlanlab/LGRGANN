from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
from .reconstruction_mi import mi_weights
from .reconstruction_grey_knn import iterative_reconstruct_labels, knn_predict_scores
from .reconstruction_gan import GanConfig, ResidualCGAN
from .reconstruction_xgb_refiner import XgbResidualRefiner, XgbRefinerConfig


@dataclass
class Stage1Config:
    k_grid: List[int] = field(default_factory=lambda: [3, 5, 7, 9, 11, 15, 21])
    grey_rho: float = 0.5
    eps: float = 1e-8
    max_iter_knn: int = 10
    method: str = "LGRGANN"
    feature_types: Optional[List[str]] = None
    gan: Optional[dict] = None
    xgb_refiner: Optional[dict] = None


def _sel_k(
    X: np.ndarray,
    yo: np.ndarray,
    lm: np.ndarray,
    cls: np.ndarray,
    kg: List[int],
    tau: np.ndarray,
    rho: float,
    eps: float,
    mi: int,
    ispl,
    ft: Optional[np.ndarray] = None,
) -> int:
    from sklearn.metrics import accuracy_score

    yo, il = np.asarray(yo), np.where(lm)[0]
    bk, bs = kg[0], -1.0
    for k in kg:
        sc = []
        for tr, va in ispl:
            ti, vi = il[tr], il[va]
            tm = np.isin(np.arange(len(yo)), ti)
            yf, _ = iterative_reconstruct_labels(
                X, yo.copy(), cls, tm, min(k, len(ti)), tau, rho, eps, mi, ft
            )
            yti = np.array([int(np.where(cls == y)[0][0]) for y in yo[vi]])
            ypi = np.array([int(np.where(cls == y)[0][0]) for y in yf[vi]])
            sc.append(accuracy_score(yti, ypi))
        m = float(np.mean(sc)) if sc else -1.0
        if m > bs:
            bs, bk = m, k
    return bk


def run_stage1(
    X: np.ndarray,
    y: np.ndarray,
    lm: np.ndarray,
    cls: np.ndarray,
    seed: int,
    cfg: Stage1Config,
    inner_splits_for_k=None,
) -> Dict[str, object]:
    ft = np.array(cfg.feature_types) if cfg.feature_types else None
    tau = mi_weights(X[lm], y[lm], None, "classif", seed)
    k = (
        cfg.k_grid[0]
        if inner_splits_for_k is None
        else _sel_k(
            X,
            y,
            lm,
            cls,
            cfg.k_grid,
            tau,
            cfg.grey_rho,
            cfg.eps,
            cfg.max_iter_knn,
            inner_splits_for_k,
            ft,
        )
    )
    yf, phi = iterative_reconstruct_labels(
        X,
        y,
        cls,
        lm,
        min(k, int(lm.sum())),
        tau,
        cfg.grey_rho,
        cfg.eps,
        cfg.max_iter_knn,
        ft,
    )
    if cfg.method.upper() == "GREY_KNN":
        return {"k": k, "tau": tau, "y_recon": yf, "phi": phi, "refiner": None}
    phil, yt, mm = phi[lm], y[lm], ~lm
    if cfg.method.upper() == "GREY_KNN_XGB_REFINER":
        xc = XgbRefinerConfig(random_state=seed, **(cfg.xgb_refiner or {}))
        ref = XgbResidualRefiner(len(cls), xc)
        ref.fit(phil, yt, cls)
        rs = ref.predict_scores(phi[mm])
        yf[mm], phi[mm] = cls[np.argmax(rs, axis=1)], rs
        return {"k": k, "tau": tau, "y_recon": yf, "phi": phi, "refiner": "xgb"}
    dgc = {
        "z_dim": 16,
        "hidden": (128, 128),
        "lr_g": 2e-4,
        "lr_d": 5e-4,
        "batch_size": 128,
        "epochs": 300,
        "n_critic": 2,
        "early_stop_patience": 25,
        "eval_every": 5,
        "val_mask_frac": 0.15,
        "device": "cpu",
        "label_smoothing": 0.1,
        "grad_clip": 1.0,
        "dropout_g": 0.0,
        "dropout_d": 0.1,
    }
    if cfg.gan:
        dgc.update(cfg.gan)
    gc = GanConfig(**dgc)
    gan = ResidualCGAN(len(cls), gc)
    rng = np.random.default_rng(seed)
    il = np.where(lm)[0]
    nv = max(1, int(len(il) * gc.val_mask_frac))
    vi = rng.choice(il, size=nv, replace=False)
    tmv = lm.copy()
    tmv[vi] = False
    vpp = lambda: knn_predict_scores(
        X[tmv],
        y[tmv],
        X[vi],
        cls,
        min(k, int(tmv.sum())),
        tau,
        cfg.grey_rho,
        cfg.eps,
        ft,
    )
    ti = gan.fit(phil, yt, cls, np.isin(il, vi), vpp, y[vi], seed)
    rs = gan.predict_scores(phi[mm], seed)
    yf[mm], phi[mm] = cls[np.argmax(rs, axis=1)], rs
    return {
        "k": k,
        "tau": tau,
        "y_recon": yf,
        "phi": phi,
        "refiner": "gan",
        "gan_train": ti,
    }
