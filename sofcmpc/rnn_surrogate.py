"""GRU surrogate of the SOFC model: MLP encoder (initial state) + GRU + MLP output head."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class GRUSurrogate(nn.Module):
    def __init__(self, n_x=2, n_u=2, n_y=4, hidden=64, layers=1):
        super().__init__()
        self.hidden, self.layers = hidden, layers
        self.enc = nn.Sequential(nn.Linear(n_x, hidden), nn.Tanh(),
                                 nn.Linear(hidden, hidden * layers), nn.Tanh())
        self.gru = nn.GRU(n_u, hidden, layers, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(),
                                  nn.Linear(hidden, n_y))
        for name, n in [("x_mu", n_x), ("x_sd", n_x), ("u_lb", n_u),
                        ("u_ub", n_u), ("y_mu", n_y), ("y_sd", n_y)]:
            self.register_buffer(name, torch.zeros(n))

    def set_normalisation(self, X0, U_lb, U_ub, Y):
        self.x_mu.copy_(torch.as_tensor(X0.mean(0)))
        self.x_sd.copy_(torch.as_tensor(X0.std(0)))
        self.u_lb.copy_(torch.as_tensor(U_lb))
        self.u_ub.copy_(torch.as_tensor(U_ub))
        Yf = Y.reshape(-1, Y.shape[-1])
        self.y_mu.copy_(torch.as_tensor(Yf.mean(0)))
        self.y_sd.copy_(torch.as_tensor(Yf.std(0)))

    def forward_norm(self, x0n, Un):
        B = x0n.shape[0]
        h0 = self.enc(x0n).view(B, self.layers, self.hidden).transpose(0, 1).contiguous()
        H, _ = self.gru(Un, h0)
        return self.head(H)

    def forward(self, x0, U):
        x0n = (x0 - self.x_mu) / self.x_sd
        Un = (U - self.u_lb) / (self.u_ub - self.u_lb)
        return self.forward_norm(x0n, Un) * self.y_sd + self.y_mu


def make_windows(data, H=20, stride=5):
    X0, U, Y = data["X0"], data["U"], data["Y"]
    xs, us, ys = [], [], []
    n_steps = U.shape[1]
    for k in range(0, n_steps - H + 1, stride):
        x0 = X0 if k == 0 else Y[:, k - 1, :2]
        xs.append(x0); us.append(U[:, k:k + H]); ys.append(Y[:, k:k + H])
    return np.concatenate(xs), np.concatenate(us), np.concatenate(ys)


def train(model, train_w, val_w, epochs=60, batch=256, lr=2e-3, seed=0, log=print):
    torch.manual_seed(seed)
    to_t = lambda a: torch.as_tensor(a, dtype=torch.float32)
    x_tr, u_tr, y_tr = map(to_t, train_w)
    x_va, u_va, y_va = map(to_t, val_w)

    def norm(x, u, y):
        return ((x - model.x_mu) / model.x_sd,
                (u - model.u_lb) / (model.u_ub - model.u_lb),
                (y - model.y_mu) / model.y_sd)

    x_tr, u_tr, y_tr = norm(x_tr, u_tr, y_tr)
    x_va, u_va, y_va = norm(x_va, u_va, y_va)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    history, best, best_state = [], np.inf, None
    n = x_tr.shape[0]
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        tot = 0.0
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            pred = model.forward_norm(x_tr[idx], u_tr[idx])
            loss = nn.functional.mse_loss(pred, y_tr[idx])
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(idx)
        sched.step()
        model.eval()
        with torch.no_grad():
            vl = nn.functional.mse_loss(model.forward_norm(x_va, u_va), y_va).item()
        history.append((tot / n, vl))
        if vl < best:
            best = vl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if ep % 10 == 0 or ep == epochs - 1:
            log(f"epoch {ep:3d}  train MSE {tot / n:.2e}  val MSE {vl:.2e}")
    model.load_state_dict(best_state)
    return np.array(history)
