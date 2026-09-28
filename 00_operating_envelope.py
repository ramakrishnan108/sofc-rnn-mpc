"""Steady-state operating map of the first-principles model (Fig. 0)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sofcmpc.sofc_model import SOFCParams, step, F

prm = SOFCParams()
I = np.linspace(1000, 9000, 33)
UF = np.linspace(0.30, 0.90, 25)
II, UU = np.meshgrid(I, UF, indexing="ij")
Fin = II * prm.A_cell / (2 * F) / (UU * prm.x_in)
u = np.stack([II.ravel(), Fin.ravel()], -1)
s = np.tile([0.5, 1073.0], (len(u), 1))
for _ in range(600):                                   # 3000 s
    s, y = step(s, u, prm)
P = (u[:, 0] * y[:, 2] * prm.A_cell).reshape(II.shape)
T = y[:, 1].reshape(II.shape)
xt = y[:, 3].reshape(II.shape)
ok_fin = (Fin >= prm.u_lb[1]) & (Fin <= prm.u_ub[1])
safe = ok_fin & (T <= 1100) & (xt >= 0.10)

fig, ax = plt.subplots(figsize=(6.2, 4.4))
cs = ax.contourf(UU, II / 1e4, np.where(ok_fin, P, np.nan), levels=20, cmap="viridis")
plt.colorbar(cs, label="steady-state power [W]")
ax.contour(UU, II / 1e4, T, levels=[1100], colors="r", linewidths=2)
ax.contour(UU, II / 1e4, xt, levels=[0.10], colors="w", linewidths=2, linestyles="--")
ax.contourf(UU, II / 1e4, (~safe).astype(float), levels=[0.5, 1.5], colors="none", hatches=["///"])
ax.axvline(0.85, color="orange", lw=2, ls=":")
ax.set_xlabel("fuel utilisation $U_f$ [-]"); ax.set_ylabel("current density [A/cm$^2$]")
plt.tight_layout(); plt.savefig("results/fig0_operating_envelope.png", dpi=200)
for uf in [0.5, 0.6, 0.7, 0.8]:
    j = np.argmin(abs(UF - uf))
    print(f"U_f={UF[j]:.2f}: max safe steady power = {np.nanmax(np.where(safe[:, j], P[:, j], np.nan)):.1f} W")
print(f"overall max safe steady power = {P[safe].max():.1f} W")
