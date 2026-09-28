"""Closed loop on the first-principles plant: PI + fixed-ratio baseline vs RNN-MPC (Fig. 4)."""
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sofcmpc.sofc_model import SOFCParams, step, fuel_utilisation, F
from sofcmpc.rnn_surrogate import GRUSurrogate
from sofcmpc.mpc import MPC, MPCConfig, RNNPredictor

torch.set_num_threads(4)
prm, cfg = SOFCParams(), MPCConfig()


def P_ref(t):
    if t < 100: return 25.0
    if t < 250: return 38.0
    if t < 400: return 46.0
    if t < 500: return 52.0
    return 30.0
N = 130                           # 650 s


class Baseline:
    """Velocity-form PI on current density + fixed fuel-utilisation ratio."""
    def __init__(self, U_set=0.75, Kp=100.0, Ki=90.0):
        self.U_set, self.Kp, self.Ki, self.e_prev = U_set, Kp, Ki, 0.0
    def __call__(self, x, u_prev, Pref, P_meas):
        e = Pref - P_meas
        i = u_prev[0] + self.Kp * (e - self.e_prev) + self.Ki * e
        i = float(np.clip(i, prm.u_lb[0], prm.u_ub[0]))
        self.e_prev = e
        Fin = i * prm.A_cell / (2 * F) / (self.U_set * prm.x_in)
        Fin = float(np.clip(Fin, prm.u_lb[1], prm.u_ub[1]))
        return np.array([i, Fin]), dict(solve_time=0.0)


def run(ctrl, n_steps, x0, u0, y0, name, k0=0):
    x, u, y = x0.copy(), u0.copy(), y0.copy()
    log = {k: [] for k in ["t", "Pref", "P", "i", "Fin", "x_ch", "T", "V", "x_tpb", "Uf", "solve_time"]}
    for k in range(k0, k0 + n_steps):
        t = k * prm.dt_s
        Pmeas = u[0] * y[2] * prm.A_cell
        if isinstance(ctrl, Baseline):
            u, info = ctrl(x, u, P_ref(t), Pmeas)
        else:
            u, info = ctrl(x, u, P_ref(t))
        xs, ys = step(x[None], u[None], prm)
        x, y = xs[0], ys[0]
        for kk, v in zip(log, [t + prm.dt_s, P_ref(t), u[0] * y[2] * prm.A_cell, u[0], u[1],
                               y[0], y[1], y[2], y[3], fuel_utilisation(u, prm), info["solve_time"]]):
            log[kk].append(float(v))
        if k % 10 == 0:
            print(f"[{name}] t={t:4.0f}s  Pref={P_ref(t):4.1f}  P={log['P'][-1]:5.2f}  "
                  f"T={y[1]:6.1f}  xTPB={y[3]:.3f}  Uf={log['Uf'][-1]:.2f}  "
                  f"solve={info['solve_time']:.2f}s", flush=True)
    return {k: np.array(v) for k, v in log.items()}


x0, u0 = np.array([0.30, 1050.0]), np.array([4000.0, 3.0e-4])
_, y0 = step(x0[None], u0[None], prm); y0 = y0[0]
bl = Baseline()
for _ in range(200):
    u0, _ = bl(x0, u0, 25.0, u0[0] * y0[2] * prm.A_cell)
    xs, ys = step(x0[None], u0[None], prm); x0, y0 = xs[0], ys[0]
print("initial state", x0, "inputs", u0)

model = GRUSurrogate(); model.load_state_dict(torch.load("results/gru_surrogate.pt"))
logs = {}
logs["Baseline (PI + ratio)"] = run(Baseline(), N, x0, u0, y0, "baseline")
logs["RNN-MPC"] = run(MPC(RNNPredictor(model, prm, cfg), prm, cfg), N, x0, u0, y0, "rnn-mpc")

kpi = {}
for name, L in logs.items():
    kpi[name] = dict(
        steps=len(L["t"]),
        tracking_RMSE_W=float(np.sqrt(np.mean((L["P"] - L["Pref"]) ** 2))),
        energy_kJ=float(L["P"].sum() * prm.dt_s / 1e3),
        H2_fed_mol=float(L["Fin"].sum() * prm.dt_s * prm.x_in),
        electrical_eff_LHV=float(L["P"].sum() / (L["Fin"].sum() * prm.x_in * 241.8e3)),
        T_max_K=float(L["T"].max()),
        T_violation_Ks=float(np.maximum(L["T"] - cfg.T_max, 0).sum() * prm.dt_s),
        xTPB_min=float(L["x_tpb"].min()),
        time_below_xTPB_min_s=float((L["x_tpb"] < cfg.xtpb_min).sum() * prm.dt_s),
        mean_solve_time_s=float(L["solve_time"].mean()),
    )
json.dump(kpi, open("results/closed_loop_kpis.json", "w"), indent=2)
np.savez("results/closed_loop_logs.npz", **{f"{n}|{k}": v for n, L in logs.items() for k, v in L.items()})
for n, d in kpi.items():
    print(n, json.dumps(d, indent=1))

sty = {"Baseline (PI + ratio)": dict(c="tab:gray", ls="-"), "RNN-MPC": dict(c="tab:red", ls="-")}
fig, ax = plt.subplots(3, 2, figsize=(12, 8.5), sharex=True)
ax = ax.ravel()
b = logs["RNN-MPC"]
ax[0].step(b["t"], b["Pref"], "k:", where="pre", lw=1.5, label="demand")
for n, L in logs.items():
    ax[0].plot(L["t"], L["P"], label=n, **sty[n])
    ax[1].plot(L["t"], L["T"], **sty[n])
    ax[2].plot(L["t"], L["x_tpb"], **sty[n])
    ax[3].plot(L["t"], L["Uf"], **sty[n])
    ax[4].step(L["t"], L["i"] / 1e4, where="pre", **sty[n])
    ax[5].step(L["t"], L["Fin"] * 1e3, where="pre", **sty[n])
ax[1].axhline(cfg.T_max, color="k", ls="--", lw=1); ax[2].axhline(cfg.xtpb_min, color="k", ls="--", lw=1)
ax[3].axhline(cfg.U_max, color="k", ls="--", lw=1)
for a, lab in zip(ax, ["power [W]", "cell temperature [K]", "H$_2$ mole fraction at TPB",
                       "fuel utilisation [-]", "current density [A/cm²]", "fuel feed [mmol/s]"]):
    a.set_ylabel(lab); a.grid(alpha=.3)
ax[0].legend(fontsize=8); ax[4].set_xlabel("time [s]"); ax[5].set_xlabel("time [s]")
plt.tight_layout(); plt.savefig("results/fig4_closed_loop.png", dpi=200)
print("saved results/fig4_closed_loop.png")
