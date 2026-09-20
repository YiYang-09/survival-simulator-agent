"""
Behavior-clone the PPO actor-critic from the potential-field controller, so
training starts from a policy that already forages instead of from noise.

Our first PPO run started from a Gaussian centred on "run at half sprint speed
forever", never banked the 100 energy a birth costs, peaked at 5 agents and
died of old age at ~200s. Discovering foraging from scratch is not affordable
at ~2 minutes per episode; improving on a ~700-score policy is a different and
much easier problem.

    python -m rl.clone_ppo --episodes 4 --epochs 300
"""
import argparse

import numpy as np
import torch
import torch.nn as nn

from rl.clone_policy import collect  # reuses the demonstration collector
from rl.obs_encoder import OBS_DIM
from rl.policy_net import ActorCritic, encode_action_targets
from rl.policies.potential_field_tuned import make_tuned_policy


def collect_ppo_targets(teacher, seeds, max_time):
    """collect() hands back (obs, mlp-space targets); we only need obs plus the
    teacher's raw actions, so re-derive targets in the ActorCritic action space."""
    from src.core import SimulationCore
    from rl.env_wrapper import Action
    from rl.obs_encoder import encode_agent_observation
    import random

    X, A, S = [], [], []
    sampler = random.Random(12345)
    for seed in seeds:
        sim = SimulationCore(seed=seed, dt=1 / 10)
        rng = random.Random(seed)
        actions = []
        while True:
            state = sim.step(actions)
            if state["num_agents"] == 0 or sim.env.time > max_time:
                break
            actions = []
            for st in state["observations"]:
                st["sim_time"] = state["sim_time"]
                st["n_agents"] = state["num_agents"]
                raw = teacher(st, rng)
                if sampler.random() < 0.05:
                    X.append(encode_agent_observation(st).astype(np.float32))
                    A.append(encode_action_targets(raw, st["sprint_speed"]))
                    S.append(1.0 if raw["spawn_agent"] else 0.0)
                actions.append((st["agent_id"], Action(**raw)))
        print(f"  seed {seed}: t={sim.env.time:.0f}s, samples {len(X)}", flush=True)
    return np.array(X), np.array(A), np.array(S, dtype=np.float32)


def fit(net, X, A, S, epochs=300, lr=3e-3, batch=512):
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    Xt, At, St = map(lambda z: torch.tensor(z, dtype=torch.float32), (X, A, S))
    n = Xt.shape[0]
    bce = nn.BCEWithLogitsLoss()

    for ep in range(1, epochs + 1):
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            mean, _std, spawn_logit, _value = net.forward(Xt[idx])
            loss = ((mean - At[idx]) ** 2).mean() + bce(spawn_logit, St[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * idx.numel()
        if ep % 50 == 0 or ep == 1:
            print(f"  epoch {ep:4d}  loss={total / n:.4f}", flush=True)
    return net


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--base_seed", type=int, default=7100)
    parser.add_argument("--max_time", type=float, default=1200.0)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--teacher", default="rl/checkpoints/pf_params_warmstart.json")
    parser.add_argument("--out", default="rl/checkpoints/ppo_clone_init.pt")
    args = parser.parse_args()

    teacher = make_tuned_policy(args.teacher)
    seeds = list(range(args.base_seed, args.base_seed + args.episodes))

    print(f"collecting demonstrations from {args.teacher} ...", flush=True)
    X, A, S = collect_ppo_targets(teacher, seeds, args.max_time)
    print(f"collected {X.shape[0]} samples (obs_dim={X.shape[1]}, expected {OBS_DIM})")

    net = ActorCritic(OBS_DIM)
    fit(net, X, A, S, epochs=args.epochs)
    torch.save(net.state_dict(), args.out)
    print(f"saved -> {args.out}")
