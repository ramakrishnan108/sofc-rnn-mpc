"""Dynamic first-principles model of a planar SOFC with a dusty-gas porous anode."""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

R = 8.314          # J/(mol K)
F = 96485.0        # C/mol
P0 = 101325.0      # Pa
M_H2, M_H2O = 2.016e-3, 18.015e-3   # kg/mol


@dataclass
class SOFCParams:
    # geometry / cell
    A_cell: float = 0.01          # m^2
    L_an: float = 1.0e-3          # m
    eps: float = 0.30
    tau: float = 5.0
    r_pore: float = 0.5e-6        # m
    mu_gas: float = 3.0e-5        # Pa s
    # feed / operating
    p_op: float = P0              # Pa
    x_in: float = 0.97
    x_O2: float = 0.21
    # electrochemistry
    i0_an: float = 5300.0         # A/m^2
    i0_ca: float = 2000.0         # A/m^2
    ASR0: float = 1.5e-5          # Ohm m^2
    Ea_ohm: float = 80e3          # J/mol
    T_ref: float = 1073.0         # K
    E_tn: float = 1.285           # V
    # lumped dynamics
    NV_ch: float = 6.0e-3         # mol
    Mcp: float = 150.0            # J/K
    hA: float = 0.60              # W/K
    T_amb: float = 1023.0         # K
    # numerics
    nz: int = 20
    dt_int: float = 0.5           # s
    dt_s: float = 5.0             # s
    # input bounds [i (A/m^2), F_in (mol/s)]
    u_lb: np.ndarray = field(default_factory=lambda: np.array([1000.0, 1.5e-4]))
    u_ub: np.ndarray = field(default_factory=lambda: np.array([9000.0, 1.2e-3]))


def binary_diffusivity(T, p):
    vH2, vH2O = 7.07, 12.7
    MA, MB = 2.016, 18.015
    return 1.013e-2 * T**1.75 * np.sqrt(1 / MA + 1 / MB) / (
        p * (vH2 ** (1 / 3) + vH2O ** (1 / 3)) ** 2)


def knudsen_diffusivity(T, M, r_pore):
    return (2.0 / 3.0) * r_pore * np.sqrt(8 * R * T / (np.pi * M))


def dgm_anode(x_ch, T, i, prm: SOFCParams):
    """Binary H2/H2O dusty-gas model across the anode; returns (x_H2 at TPB, p at TPB)."""
    x_ch, T, i = np.broadcast_arrays(np.asarray(x_ch, float),
                                     np.asarray(T, float),
                                     np.asarray(i, float))
    fe = prm.eps / prm.tau
    D12 = fe * binary_diffusivity(T, prm.p_op)
    DK1 = fe * knudsen_diffusivity(T, M_H2, prm.r_pore)
    DK2 = fe * knudsen_diffusivity(T, M_H2O, prm.r_pore)
    B0 = prm.eps * prm.r_pore**2 / (8 * prm.tau)
    N = i / (2 * F)
    RT = R * T

    def rhs(p, p1):
        num = -RT * (N / DK1 - N / DK2)
        den = 1.0 + (B0 / prm.mu_gas) * (p1 / DK1 + (p - p1) / DK2)
        dp = num / den
        dp1 = -RT * N * (1.0 / DK1 + 1.0 / D12) - (B0 * p1 / (prm.mu_gas * DK1)) * dp
        return dp, dp1

    p = np.full_like(x_ch, prm.p_op)
    p1 = x_ch * prm.p_op
    h = prm.L_an / prm.nz
    for _ in range(prm.nz):
        k1p, k1 = rhs(p, p1)
        k2p, k2 = rhs(p + 0.5 * h * k1p, p1 + 0.5 * h * k1)
        k3p, k3 = rhs(p + 0.5 * h * k2p, p1 + 0.5 * h * k2)
        k4p, k4 = rhs(p + h * k3p, p1 + h * k3)
        p = p + h / 6 * (k1p + 2 * k2p + 2 * k3p + k4p)
        p1 = p1 + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        p1 = np.maximum(p1, 1e-6 * prm.p_op)
    x_tpb = np.clip(p1 / p, 1e-6, 1 - 1e-6)
    return x_tpb, p


def nernst(xH2, p, T, prm: SOFCParams):
    E0 = 1.253 - 2.4516e-4 * T
    pH2 = np.maximum(xH2 * p / P0, 1e-12)
    pH2O = np.maximum((1 - xH2) * p / P0, 1e-12)
    pO2 = prm.x_O2 * prm.p_op / P0
    return E0 + (R * T / (2 * F)) * np.log(pH2 * np.sqrt(pO2) / pH2O)


def cell_voltage(x_ch, T, i, prm: SOFCParams, return_parts: bool = False):
    x_tpb, p_tpb = dgm_anode(x_ch, T, i, prm)
    E_bulk = nernst(x_ch, prm.p_op, T, prm)
    E_tpb = nernst(x_tpb, p_tpb, T, prm)
    eta_a = (R * T / F) * np.arcsinh(i / (2 * prm.i0_an))
    eta_c = (R * T / F) * np.arcsinh(i / (2 * prm.i0_ca))
    ASR = prm.ASR0 * np.exp(prm.Ea_ohm / R * (1 / T - 1 / prm.T_ref))
    eta_ohm = i * ASR
    V = E_tpb - eta_a - eta_c - eta_ohm
    if return_parts:
        return V, dict(x_tpb=x_tpb, E_bulk=E_bulk, E_tpb=E_tpb,
                       eta_conc=E_bulk - E_tpb, eta_a=eta_a,
                       eta_c=eta_c, eta_ohm=eta_ohm)
    return V, x_tpb


def rhs(state, u, prm: SOFCParams):
    """d/dt [x_ch, T] for inputs u = [i, F_in]."""
    x_ch, T = state[..., 0], state[..., 1]
    i, Fin = u[..., 0], u[..., 1]
    V, _ = cell_voltage(x_ch, T, i, prm)
    r_H2 = i * prm.A_cell / (2 * F)
    NV = prm.NV_ch * prm.T_ref / T
    dx = (Fin * (prm.x_in - x_ch) - r_H2) / NV
    dT = (i * prm.A_cell * (prm.E_tn - V) - prm.hA * (T - prm.T_amb)) / prm.Mcp
    return np.stack([dx, dT], axis=-1)


def step(state, u, prm: SOFCParams):
    """Advance one sampling period (zero-order hold); outputs y = [x_ch, T, V, x_TPB]."""
    s = np.array(state, float, copy=True)
    u = np.asarray(u, float)
    n_sub = int(round(prm.dt_s / prm.dt_int))
    h = prm.dt_int
    for _ in range(n_sub):
        k1 = rhs(s, u, prm)
        k2 = rhs(s + 0.5 * h * k1, u, prm)
        k3 = rhs(s + 0.5 * h * k2, u, prm)
        k4 = rhs(s + h * k3, u, prm)
        s = s + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        s[..., 0] = np.clip(s[..., 0], 1e-4, 0.999)
    V, x_tpb = cell_voltage(s[..., 0], s[..., 1], u[..., 0], prm)
    y = np.stack([s[..., 0], s[..., 1], V, x_tpb], axis=-1)
    return s, y


def simulate(state0, U, prm: SOFCParams):
    """state0 (B, 2), U (B, H, 2) -> Y (B, H, 4)."""
    s = np.array(state0, float)
    Y = np.empty(U.shape[:2] + (4,))
    for k in range(U.shape[1]):
        s, Y[:, k] = step(s, U[:, k], prm)
    return Y


def fuel_utilisation(u, prm: SOFCParams):
    u = np.asarray(u, float)
    return u[..., 0] * prm.A_cell / (2 * F) / (u[..., 1] * prm.x_in)


def steady_state(u, prm: SOFCParams, x0=(0.5, 1073.0), t_end=3000.0):
    s = np.array(x0, float)[None]
    uu = np.asarray(u, float)[None]
    for _ in range(int(t_end / prm.dt_s)):
        s, y = step(s, uu, prm)
    return s[0], y[0]
