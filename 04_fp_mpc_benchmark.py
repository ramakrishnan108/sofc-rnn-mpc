"""RNN-MPC vs the same MPC with the first-principles predictor (Fig. 5).

Runs on steps 16-28 (80-140 s) from the RNN-MPC state at step 16; checkpointed,
re-run to continue:  python 04_fp_mpc_benchmark.py --steps 4
Requires results/closed_loop_logs.npz from 03_closed_loop.py.
"""
import argparse, json, os, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sofcmpc.sofc_model import SOFCParams, step, fuel_utilisation
from sofcmpc.mpc import MPC, MPCConfig, FPPredictor

ap = argparse.ArgumentParser()
ap.add_argument("--start", type=int, default=16)
ap.add_argument("--window", type=int, default=12)
ap.add_argument("--steps", type=int, default=12)
args = ap.parse_args()
prm, cfg = SOFCParams(), MPCConfig()
CK = "results/fp_checkpoint.pkl"

raw = np.load("results/closed_loop_logs.npz")
R = {k.split("|")[1]: raw[k] for k in raw.files if k.startswith("RNN-MPC|")}
P_ref = lambda t: 25.0 if t < 100 else 38.0

if os.path.exists(CK):
    st = pickle.load(open(CK, "rb"))
else:
    k = args.start
    st = dict(k=k, x=np.array([R["x_ch"][k - 1], R["T"][k - 1]]),
              u=np.array([R["i"][k - 1], R["Fin"][k - 1]]), z=None,
              log={n: [] for n in ["t", "Pref", "P", "i", "Fin", "x_ch", "T", "V", "x_tpb", "Uf", "solve_time"]})
mpc = MPC(FPPredictor(prm, cfg), prm, cfg); mpc.z = st["z"]
end = args.start + args.window
for _ in range(args.steps):
    k = st["k"]
    if k >= end:
        break
    t = k * prm.dt_s
    u, info = mpc(st["x"], st["u"], P_ref(t))
    xs, ys = step(st["x"][None], u[None], prm)
    x, y = xs[0], ys[0]
    for n, v in zip(st["log"], [t + prm.dt_s, P_ref(t), u[0] * y[2] * prm.A_cell, u[0], u[1],
                               y[0], y[1], y[2], y[3], fuel_utilisation(u, prm), info["solve_time"]]):
        st["log"][n].append(float(v))
    st.update(k=k + 1, x=x, u=u, z=mpc.z)
    pickle.dump(st, open(CK, "wb"))
    print(f"[fp-mpc] t={t:4.0f}s  P={st['log']['P'][-1]:5.2f}  i={u[0]:6.0f}  "
          f"Fin={u[1]*1e3:.3f} mmol/s  solve={info['solve_time']:.1f}s", flush=True)

if st["k"] < end:
    print(f"checkpoint saved at step {st['k']} of {end}; run again to continue")
    raise SystemExit

Fp = {n: np.array(v) for n, v in st["log"].items()}
s = slice(args.start, end)
cmp_ = dict(
    window_s=[args.start * prm.dt_s, end * prm.dt_s],
    RNN_MPC_tracking_RMSE_W=float(np.sqrt(np.mean((R["P"][s] - R["Pref"][s]) ** 2))),
    FP_MPC_tracking_RMSE_W=float(np.sqrt(np.mean((Fp["P"] - Fp["Pref"]) ** 2))),
    max_abs_diff_current_A_m2=float(np.abs(R["i"][s] - Fp["i"]).max()),
    mean_abs_diff_current_A_m2=float(np.abs(R["i"][s] - Fp["i"]).mean()),
    max_abs_diff_fuel_mmol_s=float(np.abs(R["Fin"][s] - Fp["Fin"]).max() * 1e3),
    RNN_MPC_min_xTPB=float(R["x_tpb"][s].min()), FP_MPC_min_xTPB=float(Fp["x_tpb"].min()),
    RNN_MPC_mean_solve_s=float(R["solve_time"][s].mean()),
    FP_MPC_mean_solve_s=float(Fp["solve_time"].mean()),
)
cmp_["speed_up"] = cmp_["FP_MPC_mean_solve_s"] / cmp_["RNN_MPC_mean_solve_s"]
json.dump(cmp_, open("results/fp_vs_rnn_mpc.json", "w"), indent=2)
print(json.dumps(cmp_, indent=1))

fig, ax = plt.subplots(1, 4, figsize=(15, 3.4))
t = Fp["t"]
ax[0].step(t, Fp["Pref"], "k:", where="pre", label="demand")
ax[0].plot(t, R["P"][s], "r-", label="RNN-MPC"); ax[0].plot(t, Fp["P"], "b--", label="FP-MPC")
ax[0].set_ylabel("power [W]"); ax[0].legend(fontsize=8)
ax[1].step(t, R["i"][s], "r-", where="pre"); ax[1].step(t, Fp["i"], "b--", where="pre"); ax[1].set_ylabel("i [A/m²]")
ax[2].step(t, R["Fin"][s] * 1e3, "r-", where="pre"); ax[2].step(t, Fp["Fin"] * 1e3, "b--", where="pre"); ax[2].set_ylabel("F_in [mmol/s]")
ax[3].bar(["RNN-MPC", "FP-MPC"], [cmp_["RNN_MPC_mean_solve_s"], cmp_["FP_MPC_mean_solve_s"]], color=["tab:red", "tab:blue"])
ax[3].set_yscale("log"); ax[3].set_ylabel("mean solve time per step [s]")
ax[3].set_title(f"speed-up ≈ {cmp_['speed_up']:.0f}×")
for a in ax[:3]: a.set_xlabel("time [s]"); a.grid(alpha=.3)
plt.tight_layout(); plt.savefig("results/fig5_rnn_vs_fp_mpc.png", dpi=200)
print("saved results/fig5_rnn_vs_fp_mpc.png")
