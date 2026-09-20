"""
Neuroevolution of the linear policy weights with an OpenAI-ES style update
(mirrored sampling + rank shaping).

Why ES rather than PPO here:
  * fitness is the true undiscounted episode score -- exactly what the
    competition scores. No discount factor, so the ~600-tick delay between
    "pay 100 energy to reproduce" and "that child keeps the run alive" is not
    discounted away (gamma=0.99 at 10 ticks/s gave our PPO a 10-SECOND
    horizon, which is why it never banked energy and died at ~200s).
  * no reward shaping, so no way to optimize the wrong objective.
  * embarrassingly parallel across CPU cores, which is the only compute that
    actually helps here (the bottleneck is the Python simulation, not matmuls).

    python -m rl.evolve_nn --iterations 200 --pop 12 --workers 10 \
        --init rl/checkpoints/linear_clone.npy
"""
import argparse
import multiprocessing as mp
import os
import random

import numpy as np

from rl.eval import run_episode
from rl.policies import linear_policy, mlp_policy

BEST_PATH = "rl/checkpoints/es_best.npy"

ARCHS = {
    "linear": (linear_policy.N_PARAMS, linear_policy.make_linear_policy),
    "mlp": (mlp_policy.N_PARAMS, mlp_policy.make_mlp_policy),
}


def _eval_theta(args):
    theta, seeds, max_time, arch = args
    policy_fn = ARCHS[arch][1](theta)
    scores = [run_episode(policy_fn, s, max_time=max_time)["score"] for s in seeds]
    return float(np.mean(scores))


def _rank_utilities(fitnesses):
    """Map raw fitness to centered rank utilities in [-0.5, 0.5]: makes the
    update invariant to fitness scale and robust to the huge per-seed
    variance this game has."""
    n = len(fitnesses)
    order = np.argsort(np.argsort(fitnesses))  # ranks, 0 = worst
    return order / (n - 1) - 0.5 if n > 1 else np.zeros(n)


def evolve(iterations, pop_size, n_seeds, workers, sigma, lr, max_time,
           init_theta=None, seed=0, out_path=BEST_PATH, arch="mlp"):
    rng = np.random.default_rng(seed)
    seed_pool = random.Random(seed)

    n_params = ARCHS[arch][0]
    theta = np.zeros(n_params) if init_theta is None else np.array(init_theta, dtype=np.float64)
    assert theta.size == n_params, f"expected {n_params} params for arch={arch}, got {theta.size}"

    half = max(1, pop_size // 2)
    pool = mp.Pool(workers) if workers > 1 else None
    best_fitness = -1e18

    try:
        for it in range(1, iterations + 1):
            # common random numbers: every candidate this iteration is judged
            # on the SAME seeds, so differences reflect the weights, not luck
            seeds = [seed_pool.randint(0, 2**31 - 1) for _ in range(n_seeds)]

            eps = rng.normal(size=(half, n_params))
            candidates = [theta + sigma * e for e in eps] + [theta - sigma * e for e in eps]
            tasks = [(c, seeds, max_time, arch) for c in candidates]

            fitnesses = pool.map(_eval_theta, tasks) if pool else [_eval_theta(t) for t in tasks]
            fitnesses = np.array(fitnesses)

            utilities = _rank_utilities(fitnesses)
            grad = (utilities[:half, None] * eps).sum(axis=0) + (utilities[half:, None] * (-eps)).sum(axis=0)
            theta = theta + (lr / (pop_size * sigma)) * grad

            it_best = float(fitnesses.max())
            if it_best > best_fitness:
                best_fitness = it_best
            print(
                f"[iter {it:4d}] mean={fitnesses.mean():8.1f} best_this_iter={it_best:8.1f} "
                f"best_ever={best_fitness:8.1f} |theta|={np.linalg.norm(theta):7.2f}",
                flush=True,
            )

            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            np.save(out_path, theta)
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    return theta


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--pop", type=int, default=12, help="candidates per iteration (mirrored, so use an even number)")
    parser.add_argument("--n_seeds", type=int, default=3)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--sigma", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--max_time", type=float, default=1500.0)
    parser.add_argument("--init", default=None, help=".npy weight vector to start from (e.g. the cloned policy)")
    parser.add_argument("--out", default=BEST_PATH)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--arch", choices=list(ARCHS.keys()), default="mlp")
    args = parser.parse_args()

    init = np.load(args.init) if args.init else None
    evolve(
        iterations=args.iterations,
        pop_size=args.pop,
        n_seeds=args.n_seeds,
        workers=args.workers,
        sigma=args.sigma,
        lr=args.lr,
        max_time=args.max_time,
        init_theta=init,
        seed=args.seed,
        out_path=args.out,
        arch=args.arch,
    )
