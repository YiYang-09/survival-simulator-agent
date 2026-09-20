"""
Behavior cloning: fit the linear policy to imitate the evolved potential-field
controller, by least squares (closed form, seconds).

This is the warm start for neuroevolution -- ES starting from random weights
would have to rediscover basic foraging, and at 1-2 minutes per episode we
cannot afford that. Starting from "roughly as good as the current best" means
every ES iteration is spent improving on it instead of catching up.

    python -m rl.clone_policy --episodes 4 --out rl/checkpoints/linear_clone.npy
"""
import argparse
import random

import numpy as np

from src.core import SimulationCore
from rl.env_wrapper import Action
from rl.obs_encoder import OBS_DIM, encode_agent_observation
from rl.policies.linear_policy import N_OUT, encode_targets, pack


def collect(teacher_fn, seeds, max_time=1200.0, dt=1 / 10,
            keep_prob=0.03, max_samples=150_000):
    """A full episode with ~40 agents produces >1M (agent, tick) pairs, so
    subsample: we only need enough to pin down 396 parameters."""
    X, Y = [], []
    sampler = random.Random(12345)
    for seed in seeds:
        sim = SimulationCore(seed=seed, dt=dt)
        rng = random.Random(seed)
        actions = []
        while True:
            state = sim.step(actions)
            if state["num_agents"] == 0 or sim.env.time > max_time:
                break

            actions = []
            for agent_status in state["observations"]:
                agent_status["sim_time"] = state["sim_time"]
                agent_status["n_agents"] = state["num_agents"]
                raw = teacher_fn(agent_status, rng)

                if len(X) < max_samples and sampler.random() < keep_prob:
                    X.append(encode_agent_observation(agent_status).astype(np.float64))
                    Y.append(encode_targets(raw, agent_status["sprint_speed"]))

                actions.append((agent_status["agent_id"], Action(**raw)))
        print(f"  seed {seed}: t={sim.env.time:.0f}s, samples so far {len(X)}", flush=True)
    return np.array(X), np.array(Y)


def fit(X, Y, ridge=1e-3):
    """Ridge-regularized least squares with a bias column."""
    Xb = np.hstack([X, np.ones((X.shape[0], 1))])
    A = Xb.T @ Xb + ridge * np.eye(Xb.shape[1])
    B = Xb.T @ Y
    W = np.linalg.solve(A, B)
    pred = Xb @ W
    rmse = float(np.sqrt(((pred - Y) ** 2).mean(axis=0).mean()))
    return W, rmse


if __name__ == "__main__":
    from rl.policies.potential_field_tuned import make_tuned_policy

    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--base_seed", type=int, default=7000)
    parser.add_argument("--max_time", type=float, default=1200.0)
    parser.add_argument("--teacher", default="rl/checkpoints/pf_params_best.json")
    parser.add_argument("--out", default="rl/checkpoints/linear_clone.npy")
    args = parser.parse_args()

    teacher = make_tuned_policy(args.teacher)
    seeds = list(range(args.base_seed, args.base_seed + args.episodes))

    print(f"collecting demonstrations from {args.teacher} ...", flush=True)
    X, Y = collect(teacher, seeds, max_time=args.max_time)
    print(f"collected {X.shape[0]} samples, obs_dim={X.shape[1]} (expected {OBS_DIM})")

    W, rmse = fit(X, Y)
    print(f"cloning RMSE (pre-activation space) = {rmse:.4f}")

    np.save(args.out, pack(W))
    print(f"saved {W.size} parameters -> {args.out}")
