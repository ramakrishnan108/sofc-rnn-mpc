"""Train the GRU surrogate and evaluate it on unseen test trajectories (Figs. 1-3)."""
import os, time, json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sofcmpc.sofc_model import SOFCParams
from sofcmpc.rnn_surrogate import GRUSurrogate, make_windows, train

os.makedirs("results", exist_ok=True)
torch.set_num_threads(4)
prm = SOFCParams()
H = 20
load = lambda n: dict(np.load(f"data/{n}.npz"))
tr, va, te = load("train"), load("val"), load("test")
w_tr, w_va = make_windows(tr, H), make_windows(va, H)
print(f"training windows: {len(w_tr[0])}, validation windows: {len(w_va[0])}")

model = GRUSurrogate(hidden=64)
model.set_normalisation(w_tr[0], prm.u_lb, prm.u_ub, w_tr[2])
t0 = time.time()
hist = train(model, w_tr, w_va, epochs=60)
print(f"training time {time.time() - t0:.0f} s")
torch.save(model.state_dict(), "results/gru_surrogate.pt")

names = ["x_ch [-]", "T [K]", "V [V]", "x_TPB [-]"]
model.eval()
def predict(x0, U):
    with torch.no_grad():
        return model(torch.as_tensor(x0, dtype=torch.float32),
                     torch.as_tensor(U, dtype=torch.float32)).numpy()

def r2(p, y):
    return float(1 - ((p - y) ** 2).sum() / ((y - y.mean()) ** 2).sum())

metrics = {}
x0, U, Y = make_windows(te, H)
P = predict(x0, U)
for j, n in enumerate(names):
    e = P[..., j] - Y[..., j]
    metrics[f"H20 {n}"] = dict(RMSE=float(np.sqrt((e**2).mean())), R2=r2(P[..., j], Y[..., j]))
P80 = predict(te["X0"], te["U"])
for j, n in enumerate(names):
    e = P80[..., j] - te["Y"][..., j]
    metrics[f"H80 {n}"] = dict(RMSE=float(np.sqrt((e**2).mean())), R2=r2(P80[..., j], te["Y"][..., j]))
e = P[..., 2] - Y[..., 2]
ok = Y[..., 3] > 0.05
metrics["V RMSE, x_TPB > 0.05"] = dict(RMSE=float(np.sqrt((e[ok] ** 2).mean())), share=float(ok.mean()))
metrics["V RMSE, x_TPB <= 0.05"] = dict(RMSE=float(np.sqrt((e[~ok] ** 2).mean())), share=float((~ok).mean()))
for k, v in metrics.items():
    print(f"{k:22s}  " + "  ".join(f"{a} {b:.4g}" for a, b in v.items()))
json.dump(metrics, open("results/surrogate_metrics.json", "w"), indent=2)

plt.figure(figsize=(5, 3.5))
plt.semilogy(hist[:, 0], label="train"); plt.semilogy(hist[:, 1], label="validation")
plt.xlabel("epoch"); plt.ylabel("MSE (normalised)"); plt.legend(); plt.tight_layout()
plt.savefig("results/fig1_training_curve.png", dpi=150)

fig, ax = plt.subplots(1, 4, figsize=(14, 3.4))
for j, n in enumerate(names):
    ax[j].plot(Y[..., j].ravel()[::7], P[..., j].ravel()[::7], ".", ms=1.5, alpha=.4)
    lo, hi = Y[..., j].min(), Y[..., j].max()
    ax[j].plot([lo, hi], [lo, hi], "k--", lw=1)
    ax[j].set_title(f"{n}  R²={metrics['H20 ' + n]['R2']:.4f}")
    ax[j].set_xlabel("first-principles"); ax[j].set_ylabel("GRU")
plt.tight_layout(); plt.savefig("results/fig2_parity_test.png", dpi=200)

b = 3
t = np.arange(1, 81) * prm.dt_s
fig, ax = plt.subplots(3, 2, figsize=(11, 7), sharex=True)
ax = ax.ravel()
ax[0].step(t, te["U"][b, :, 0], where="pre"); ax[0].set_ylabel("i [A/m²]")
ax[1].step(t, te["U"][b, :, 1] * 1e3, where="pre"); ax[1].set_ylabel("F_in [mmol/s]")
for j in range(4):
    ax[j + 2].plot(t, te["Y"][b, :, j], "k", lw=2, label="first-principles (DGM)")
    ax[j + 2].plot(t, P80[b, :, j], "r--", lw=1.5, label="GRU surrogate")
    ax[j + 2].set_ylabel(names[j])
ax[2].legend(fontsize=8); ax[4].set_xlabel("time [s]"); ax[5].set_xlabel("time [s]")
plt.tight_layout(); plt.savefig("results/fig3_rollout_test.png", dpi=200)
print("figures saved to results/")
