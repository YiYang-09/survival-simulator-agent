"""
Paired A/B harness for potential-field parameter sets.

Every config is run on the SAME seeds, and since src/elements/environment.py
now iterates its spatial-grid lookups in a stable order, a (config, seed) pair
is exactly reproducible. So differences between configs are caused by the
config, not by run-to-run luck -- before that fix the same config on the same
seed scored 692 / 684 / 956, which made small-sample comparisons worthless.

Reports mean, median, and the per-seed win rate against the first (baseline)
config, which is the statistic that actually matters when the per-episode
spread is this wide.

Configs are defined in CONFIGS below; edit and run:
    python -m rl.ab --seeds 12 --workers 16 --max_time 3000
"""
import argparse
import multiprocessing as mp
import statistics as stats

from rl.eval import run_episode
from rl.policies.potential_field import potential_field_action
from rl.evolve_potential_field import load_init_params

BASE = load_init_params("rl/checkpoints/pf_params_warmstart.json")


def _run(task):
    name, params, seed, max_time = task

    def policy_fn(agent_status, rng):
        return potential_field_action(agent_status, rng, params=params)

    r = run_episode(policy_fn, seed, max_time=max_time)
    return name, seed, r["score"], r["sim_time"]


def compare(configs, seeds, workers, max_time):
    tasks = [(name, params, seed, max_time)
             for name, params in configs.items()
             for seed in seeds]

    with mp.Pool(workers) as pool:
        results = pool.map(_run, tasks)

    by_name = {name: {} for name in configs}
    for name, seed, score, _ in results:
        by_name[name][seed] = score

    baseline = next(iter(configs))
    print(f"{'config':26s} {'mean':>9s} {'median':>9s} {'min':>8s} {'max':>8s}  vs-baseline")
    for name in configs:
        scores = [by_name[name][s] for s in seeds]
        wins = sum(1 for s in seeds if by_name[name][s] > by_name[baseline][s])
        tag = "(baseline)" if name == baseline else f"{wins}/{len(seeds)} seeds better"
        print(f"{name:26s} {stats.mean(scores):9.1f} {stats.median(scores):9.1f} "
              f"{min(scores):8.1f} {max(scores):8.1f}  {tag}")
    return by_name


# ---- the structural question: does heading for trees actually pay? ----------
# Fruit only ever spawns 20-60 units from a tree (Environment.spawn_fruit_around_tree),
# so trees are the only reliable food indicator. The evolved params set W_TREE=0,
# i.e. the agents ignore them and wander instead at ~5 energy/s.
CONFIGS = {
    "baseline (ignore trees)": BASE,
    "trees w=3":               {**BASE, "W_TREE": 1.0, "W_TREE_NOFRUIT": 3.0},
    "trees w=8":               {**BASE, "W_TREE": 2.0, "W_TREE_NOFRUIT": 8.0},
    "trees w=8 + park":        {**BASE, "W_TREE": 2.0, "W_TREE_NOFRUIT": 8.0,
                                "SEARCH_SPEED_FRAC": 0.15},
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=12)
    parser.add_argument("--base_seed", type=int, default=4000)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max_time", type=float, default=3000.0)
    args = parser.parse_args()

    seed_list = list(range(args.base_seed, args.base_seed + args.seeds))
    compare(CONFIGS, seed_list, args.workers, args.max_time)
