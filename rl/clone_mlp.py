"""
Clone the potential-field teacher into the small MLP policy (torch SGD here,
NumPy weights on the way out so ES workers don't need torch).

Reuses rl/clone_policy.collect() for the demonstrations.

    python -m rl.clone_mlp --episodes 4 --epochs 300
"""
import argparse

import numpy as np
import torch
import torch.nn as nn

from rl.clone_policy import collect
from rl.policies.mlp_policy import HIDDEN, N_OUT, pack
from rl.obs_encoder import OBS_DIM
from rl.policies.potential_field_tuned import make_tuned_policy


def fit_mlp(X, Y, hidden=HIDDEN, epochs=300, lr=1e-2, batch=512, seed=0):
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(OBS_DIM, hidden), nn.Tanh(), nn.Linear(hidden, N_OUT))
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    # standardize targets per-dimension so no single head dominates the loss
    Xt = torch.tensor(X, dtype=torch.float32)
    Yt = torch.tensor(Y, dtype=torch.float32)
    n = Xt.shape[0]

    for ep in range(1, epochs + 1):
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            pred = net(Xt[idx])
            loss = ((pred - Yt[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * idx.numel()
        if ep % 50 == 0 or ep == 1:
            print(f"  epoch {ep:4d}  mse={total / n:.4f}", flush=True)

    with torch.no_grad():
        rmse = float(torch.sqrt(((net(Xt) - Yt) ** 2).mean()))
    W1 = net[0].weight.detach().numpy().T
    b1 = net[0].bias.detach().numpy()
    W2 = net[2].weight.detach().numpy().T
    b2 = net[2].bias.detach().numpy()
    return pack(W1, b1, W2, b2), rmse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--base_seed", type=int, default=7000)
    parser.add_argument("--max_time", type=float, default=1200.0)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--teacher", default="rl/checkpoints/pf_params_warmstart.json")
    parser.add_argument("--out", default="rl/checkpoints/mlp_clone.npy")
    args = parser.parse_args()

    teacher = make_tuned_policy(args.teacher)
    seeds = list(range(args.base_seed, args.base_seed + args.episodes))

    print(f"collecting demonstrations from {args.teacher} ...", flush=True)
    X, Y = collect(teacher, seeds, max_time=args.max_time)
    print(f"collected {X.shape[0]} samples", flush=True)

    theta, rmse = fit_mlp(X, Y, epochs=args.epochs)
    print(f"MLP cloning RMSE = {rmse:.4f}  ({theta.size} parameters)")

    np.save(args.out, theta)
    print(f"saved -> {args.out}")
