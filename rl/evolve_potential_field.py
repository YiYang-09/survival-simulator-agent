"""
(mu, lambda) evolution strategy that tunes the potential-field controller's
parameters (rl/policies/potential_field.py::DEFAULT_PARAMS) directly against
the true game score -- no gradients, no reward shaping, just "which
parameter vector scores higher on average".

Each generation samples `pop` candidate parameter vectors around the current
parent (Gaussian mutation), scores each on a fresh batch of random seeds
(different seeds every generation, on purpose -- we already saw PPO's
checkpoint selection get fooled by reusing the same 3 seeds, so this avoids
tuning params that just happen to suit one fixed seed set), and keeps the
mean of the top performers as the next parent. Best-ever vector is
checkpointed every generation.

NOT run automatically. Kick it off yourself, e.g.:
    python -m rl.evolve_potential_field --generations 30 --pop 16 --workers 8

Pick --workers to fit whatever CPU headroom you have left after PPO training
(each candidate x each seed is one full episode run in its own process).

After it finishes, validate the result on held-out seeds before trusting it:
    python -m rl.eval potential_field_tuned --seeds 20
"""
import argparse
import json
import multiprocessing as mp
import os
import random

import numpy as np

from rl.eval import run_episode
from rl.policies.potential_field import DEFAULT_PARAMS, potential_field_action

PARAM_NAMES = list(DEFAULT_PARAMS.keys())

# (low, high) clip bounds per parameter -- keeps mutations in sane territory
BOUNDS = {
    "W_FRUIT":           (0.1, 15.0),
    "W_PREDATOR":        (0.1, 30.0),
    "W_WALL":            (0.0, 10.0),
    "W_TREE":            (0.0, 5.0),
    "W_TREE_NOFRUIT":    (0.0, 20.0),
    "DANGER_MULT":       (0.5, 4.0),
    "FACE_PREDATOR_DIST": (60.0, 400.0),  # <90 the predator charges anyway; >vision_range = disabled
    "LOW_ENERGY_FRAC":   (0.05, 0.6),
    "SPAWN_ENERGY_FRAC": (0.3, 0.9),
    "SPAWN_MIN_ABS":     (110.0, 400.0),
    "OLD_AGE_SPAWN":     (20.0, 200.0),
    "BREED_TRAIT_MIN":   (2.0, 5.5),  # 3.0 = baseline agent; 6.0 = fully maxed traits  # >120 effectively disables it (max_age caps at 120)  # spawning costs 100 flat, keep a margin above that
    "POP_TARGET_0":      (5.0, 80.0),
    "POP_HALFLIFE":      (150.0, 3000.0),
    "POP_TARGET_MIN":    (2.0, 20.0),     # floor: the generation chain must not break
    "FORAGE_SPEED_FRAC": (0.2, 1.0),
    "SEARCH_SPEED_FRAC": (0.0, 1.0),
    "IDLE_MOVE_FRAC":    (0.0, 0.5),
    "MAX_TURN":          (0.1, np.pi),
}

BEST_PARAMS_PATH = "rl/checkpoints/pf_params_best.json"


def params_to_vec(params):
    return np.array([params[k] for k in PARAM_NAMES], dtype=np.float64)


def vec_to_params(vec):
    return {k: float(v) for k, v in zip(PARAM_NAMES, vec)}


def clip_vec(vec):
    lo = np.array([BOUNDS[k][0] for k in PARAM_NAMES])
    hi = np.array([BOUNDS[k][1] for k in PARAM_NAMES])
    return np.clip(vec, lo, hi)


def _eval_candidate(args):
    """Module-level (picklable) fitness function: mean score of one parameter
    vector over a batch of seeds. Runs in a worker process."""
    vec, seeds, max_time = args
    params = vec_to_params(vec)

    def policy_fn(agent_status, rng):
        return potential_field_action(agent_status, rng, params=params)

    scores = [run_episode(policy_fn, seed, max_time=max_time)["score"] for seed in seeds]
    return float(np.mean(scores))


def load_init_params(path):
    """Warm-start point: saved params merged over DEFAULT_PARAMS, so a file
    saved before new parameters existed still loads (missing keys default)."""
    with open(path) as f:
        data = json.load(f)
    params = dict(DEFAULT_PARAMS)
    params.update(data.get("params", data))
    return params


def evolve(generations, pop_size, elite_frac, n_seeds, workers, sigma0, seed=0,
           init_params=None, out_path=BEST_PARAMS_PATH, max_time=3000.0):
    rng = np.random.default_rng(seed)
    seed_pool = random.Random(seed)

    parent = params_to_vec(init_params or DEFAULT_PARAMS)
    ranges = np.array([BOUNDS[k][1] - BOUNDS[k][0] for k in PARAM_NAMES])
    sigma = sigma0 * ranges  # per-parameter mutation scale, shrinks over generations

    n_elite = max(1, int(pop_size * elite_frac))
    best_vec, best_fitness = parent.copy(), -1e18

    pool = mp.Pool(workers) if workers > 1 else None

    try:
        for gen in range(1, generations + 1):
            # fresh training seeds every generation -- don't overfit to a fixed set
            train_seeds = [seed_pool.randint(0, 2**31 - 1) for _ in range(n_seeds)]

            offspring = [
                clip_vec(parent + rng.normal(size=len(PARAM_NAMES)) * sigma)
                for _ in range(pop_size)
            ]
            tasks = [(vec, train_seeds, max_time) for vec in offspring]

            if pool is not None:
                fitnesses = pool.map(_eval_candidate, tasks)
            else:
                fitnesses = [_eval_candidate(t) for t in tasks]

            order = np.argsort(fitnesses)[::-1]
            elite_vecs = [offspring[i] for i in order[:n_elite]]
            parent = np.mean(elite_vecs, axis=0)

            gen_best_fitness = fitnesses[order[0]]
            if gen_best_fitness > best_fitness:
                best_fitness = gen_best_fitness
                best_vec = offspring[order[0]].copy()

            sigma *= 0.97  # narrow the search as it converges

            print(
                f"[gen {gen:3d}] mean_fitness={float(np.mean(fitnesses)):7.1f} "
                f"best_this_gen={gen_best_fitness:7.1f} best_ever={best_fitness:7.1f}"
            )

            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "w") as f:
                json.dump({"params": vec_to_params(best_vec), "fitness": best_fitness}, f, indent=2)
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    return vec_to_params(best_vec), best_fitness


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=int, default=30)
    parser.add_argument("--pop", type=int, default=16)
    parser.add_argument("--elite_frac", type=float, default=0.25)
    parser.add_argument("--n_seeds", type=int, default=8, help="training seeds per fitness evaluation")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--sigma0", type=float, default=0.15, help="initial mutation scale as a fraction of each param's bound range")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--init", default=None, help="json file to warm-start from (default: DEFAULT_PARAMS)")
    parser.add_argument("--out", default=BEST_PARAMS_PATH)
    parser.add_argument("--max_time", type=float, default=3000.0,
                        help="truncate search episodes here (cheaper fitness); validate full length afterwards")
    args = parser.parse_args()

    best_params, best_fitness = evolve(
        generations=args.generations,
        pop_size=args.pop,
        elite_frac=args.elite_frac,
        n_seeds=args.n_seeds,
        workers=args.workers,
        sigma0=args.sigma0,
        seed=args.seed,
        init_params=load_init_params(args.init) if args.init else None,
        out_path=args.out,
        max_time=args.max_time,
    )

    print("\n=== DONE ===")
    print(f"best fitness (on training seeds, optimistic): {best_fitness:.2f}")
    print("best params:")
    for k, v in best_params.items():
        print(f"  {k} = {v}")
    print(f"\nSaved to {BEST_PARAMS_PATH}")
    print("Validate on held-out seeds before trusting this: python -m rl.eval potential_field_tuned --seeds 20")
