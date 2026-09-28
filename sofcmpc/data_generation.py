"""APRBS excitation of the first-principles model to generate training data."""
from __future__ import annotations

import numpy as np
from .sofc_model import SOFCParams, simulate, F


def aprbs_inputs(n_traj, n_steps, prm: SOFCParams, rng, hold=(2, 8),
                 util_range=(0.30, 0.92)):
    U = np.empty((n_traj, n_steps, 2))
    for b in range(n_traj):
        k = 0
        while k < n_steps:
            h = rng.integers(hold[0], hold[1] + 1)
            i = rng.uniform(prm.u_lb[0], prm.u_ub[0])
            uf = rng.uniform(*util_range)
            fin = i * prm.A_cell / (2 * F) / (uf * prm.x_in)
            fin = np.clip(fin, prm.u_lb[1], prm.u_ub[1])
            U[b, k:k + h] = (i, fin)
            k += h
    return U


def generate_dataset(n_traj=2000, n_steps=80, seed=0, prm: SOFCParams | None = None):
    prm = prm or SOFCParams()
    rng = np.random.default_rng(seed)
    X0 = np.column_stack([rng.uniform(0.10, 0.90, n_traj),      # x_ch
                          rng.uniform(1030.0, 1110.0, n_traj)])  # T [K]
    U = aprbs_inputs(n_traj, n_steps, prm, rng)
    Y = simulate(X0, U, prm)
    return dict(X0=X0, U=U, Y=Y)
