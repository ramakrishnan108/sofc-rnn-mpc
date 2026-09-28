"""Generate APRBS training, validation and test data from the first-principles model."""
import os, time
import numpy as np
from sofcmpc.sofc_model import SOFCParams
from sofcmpc.data_generation import generate_dataset

os.makedirs("data", exist_ok=True)
prm = SOFCParams()
t0 = time.time()
for name, n, seed in [("train", 2000, 0), ("val", 300, 1), ("test", 300, 2)]:
    d = generate_dataset(n_traj=n, n_steps=80, seed=seed, prm=prm)
    np.savez_compressed(f"data/{name}.npz", **d)
    print(f"{name:5s}: {n} trajectories x 80 steps (dt = {prm.dt_s} s)")
print(f"done in {time.time() - t0:.1f} s")
