"""
Common evaluation harness: run any policy_fn(agent_status: dict, rng) -> dict
over N seeds using SimulationCore directly (true game mechanics, no reward
shaping), so every algorithm we try (rule-based, PPO, evolutionary, ...) is
scored the same way and numbers are comparable.

Usage:
    python -m rl.eval potential_field --seeds 5
    python -m rl.eval random --seeds 5
"""
import argparse
import random
import statistics as stats
import time

from src.core import SimulationCore
from rl.env_wrapper import Action


def run_episode(policy_fn, seed, dt=1 / 10, max_time=3000.0, core_kwargs=None):
    sim = SimulationCore(seed=seed, dt=dt, **(core_kwargs or {}))
    rng = random.Random(seed)

    actions = []
    while True:
        state = sim.step(actions)

        if state["num_agents"] == 0 or sim.env.time > max_time:
            return {
                "seed": seed,
                "score": state["score"],
                "sim_time": state["sim_time"],
                "final_agents": state["num_agents"],
            }

        actions = []
        for agent_status in state["observations"]:
            # global context the real API also provides on StepResponse --
            # injected here so training/eval and agent_server see the same thing
            agent_status["sim_time"] = state["sim_time"]
            agent_status["n_agents"] = state["num_agents"]
            raw = policy_fn(agent_status, rng)
            actions.append((agent_status["agent_id"], Action(**raw)))


def evaluate(policy_fn, seeds, **kwargs):
    print(f"{'seed':>10} {'score':>10} {'sim_time':>10} {'final_agents':>13} {'wall_s':>8}")
    results = []
    for seed in seeds:
        t0 = time.time()
        r = run_episode(policy_fn, seed, **kwargs)
        r["wall_s"] = time.time() - t0
        results.append(r)
        # print (and flush) as each episode finishes, so a long run shows live progress
        print(f"{r['seed']:>10} {r['score']:>10.2f} {r['sim_time']:>10.1f} {r['final_agents']:>13} {r['wall_s']:>8.1f}", flush=True)

    scores = [r["score"] for r in results]
    times = [r["sim_time"] for r in results]
    print("-" * 55)
    print(f"mean score = {stats.mean(scores):.2f}  (stdev {stats.pstdev(scores):.2f})")
    print(f"mean sim_time = {stats.mean(times):.1f}")
    return results


def _random_policy(agent_status, rng):
    return {
        "move_distance": rng.uniform(0.0, agent_status["sprint_speed"]),
        "move_direction": 0.0,
        "turn_angle": rng.uniform(-3.14159 / 4, 3.14159 / 4),
        "spawn_agent": True,
    }


POLICIES = {
    "random": _random_policy,
}

try:
    from rl.policies.potential_field import potential_field_action
    POLICIES["potential_field"] = potential_field_action
except ImportError:
    pass

try:
    from rl.policies.potential_field_tuned import make_tuned_policy
    POLICIES["potential_field_tuned"] = make_tuned_policy()  # needs pf_params_best.json to exist
except FileNotFoundError:
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("policy", choices=list(POLICIES.keys()))
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--base_seed", type=int, default=0)
    args = parser.parse_args()

    seeds = list(range(args.base_seed, args.base_seed + args.seeds))
    evaluate(POLICIES[args.policy], seeds)
