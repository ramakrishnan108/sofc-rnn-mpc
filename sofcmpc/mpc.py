"""Move-blocked MPC with soft constraints; GRU or first-principles predictor."""
from __future__ import annotations

import time
from dataclasses import dataclass
import numpy as np
import torch
from scipy.optimize import minimize

from .sofc_model import SOFCParams, simulate, F


@dataclass
class MPCConfig:
    H: int = 20                # steps (100 s)
    n_blocks: int = 5
    T_max: float = 1100.0      # K
    xtpb_min: float = 0.10
    U_max: float = 0.85
    P_s: float = 1.0           # W
    w_fuel: float = 1.0
    w_du: float = 0.5
    rho: float = 50.0
    maxiter: int = 60


class _CostMixin:
    def _expand(self, z, lib):
        blk = self.cfg.H // self.cfg.n_blocks
        zz = z.reshape(self.cfg.n_blocks, 2)
        un = lib.repeat_interleave(zz, blk, dim=0) if lib is torch else np.repeat(zz, blk, axis=0)
        return un

    def _cost(self, un, Y, u_prev_n, P_ref, lib):
        c, prm = self.cfg, self.prm
        lb, ub = self._lb, self._ub
        u = lb + un * (ub - lb)
        i, Fin = u[..., 0], u[..., 1]
        T, V, xt = Y[..., 1], Y[..., 2], Y[..., 3]
        P = i * V * prm.A_cell
        Uf = i * prm.A_cell / (2 * F) / (Fin * prm.x_in)
        relu = (lambda a: torch.clamp(a, min=0)) if lib is torch else (lambda a: np.maximum(a, 0))
        du = un[..., 1:, :] - un[..., :-1, :]
        du0 = un[..., 0, :] - u_prev_n
        J = (((P - P_ref) / c.P_s) ** 2).sum(-1)
        J = J + c.w_fuel * un[..., 1].sum(-1)
        J = J + c.w_du * ((du ** 2).sum((-1, -2)) + (du0 ** 2).sum(-1))
        J = J + c.rho * ((relu(T - c.T_max) / 1.0) ** 2).sum(-1)
        J = J + c.rho * ((relu(c.xtpb_min - xt) / 0.01) ** 2).sum(-1)
        J = J + c.rho * ((relu(Uf - c.U_max) / 0.01) ** 2).sum(-1)
        return J


class RNNPredictor(_CostMixin):
    def __init__(self, model, prm: SOFCParams, cfg: MPCConfig):
        self.model = model.double().eval()
        self.prm, self.cfg = prm, cfg
        self._lb = torch.as_tensor(prm.u_lb, dtype=torch.float64)
        self._ub = torch.as_tensor(prm.u_ub, dtype=torch.float64)

    def cost_and_grad(self, z, x0, u_prev_n, P_ref):
        zt = torch.as_tensor(z, dtype=torch.float64).requires_grad_(True)
        un = self._expand(zt, torch)
        u = self._lb + un * (self._ub - self._lb)
        Y = self.model(torch.as_tensor(x0, dtype=torch.float64)[None], u[None])[0]
        J = self._cost(un, Y, torch.as_tensor(u_prev_n), P_ref, torch)
        J.backward()
        return J.item(), zt.grad.numpy().copy()


class FPPredictor(_CostMixin):
    def __init__(self, prm: SOFCParams, cfg: MPCConfig, eps=1e-4):
        self.prm, self.cfg, self.eps = prm, cfg, eps
        self._lb, self._ub = prm.u_lb, prm.u_ub

    def cost_and_grad(self, z, x0, u_prev_n, P_ref):
        n = z.size
        Z = np.vstack([z, z + self.eps * np.eye(n)])
        UN = np.stack([self._expand(zz, np) for zz in Z])
        U = self._lb + UN * (self._ub - self._lb)
        Y = simulate(np.tile(x0, (n + 1, 1)), U, self.prm)
        J = self._cost(UN, Y, u_prev_n, P_ref, np)
        return J[0], (J[1:] - J[0]) / self.eps


class MPC:
    def __init__(self, predictor, prm: SOFCParams, cfg: MPCConfig):
        self.pred, self.prm, self.cfg = predictor, prm, cfg
        self.z = None

    def __call__(self, x, u_prev, P_ref):
        c, prm = self.cfg, self.prm
        u_prev_n = (np.asarray(u_prev) - prm.u_lb) / (prm.u_ub - prm.u_lb)
        if self.z is None:
            self.z = np.tile(u_prev_n, c.n_blocks)
        t0 = time.perf_counter()
        res = minimize(self.pred.cost_and_grad, self.z, args=(x, u_prev_n, P_ref),
                       jac=True, method="L-BFGS-B",
                       bounds=[(0.0, 1.0)] * self.z.size,
                       options=dict(maxiter=c.maxiter))
        dt = time.perf_counter() - t0
        self.z = res.x
        u0 = prm.u_lb + res.x[:2] * (prm.u_ub - prm.u_lb)
        return u0, dict(solve_time=dt, cost=res.fun, nit=res.nit)
