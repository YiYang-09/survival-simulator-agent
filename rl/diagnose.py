"""
Diagnostic run: track population size / mean energy / predator count over
time for a policy, to see what's actually cutting episodes short before the
3000s cap (predation, starvation, old-age attrition, no reproduction, ...).

    python -m rl.diagnose potential_field_tuned --seeds 3
"""
import argparse
import random

from src.core import SimulationCore
from rl.env_wrapper import Action


def diagnose_episode(policy_fn, seed, dt=1 / 10, max_time=3000.0, log_every=100.0):
    sim = SimulationCore(seed=seed, dt=dt)
    rng = random.Random(seed)

    actions = []
    next_log = 0.0
    ages_at_death = []
    last_n_agents = sim.env.starting_agents if hasattr(sim.env, "starting_agents") else None

    while True:
        state = sim.step(actions)

        n_agents = state["num_agents"]
        n_predators = len(sim.env.predators)

        if sim.env.time >= next_log:
            energies = [a.energy for a in sim.env.agents]
            ages = [a.age for a in sim.env.agents]
            mean_e = sum(energies) / len(energies) if energies else 0
            mean_age = sum(ages) / len(ages) if ages else 0
            n_fruits = len(sim.env.fruits)
            n_trees = len(sim.env.trees)
            print(f"  t={sim.env.time:7.1f}  agents={n_agents:3d}  predators={n_predators:2d}  "
                  f"fruits={n_fruits:3d}  trees={n_trees:3d}  mean_energy={mean_e:6.1f}  mean_age={mean_age:5.1f}")
            next_log += log_every

        if n_agents == 0 or sim.env.time > max_time:
            print(f"  ENDED at t={sim.env.time:.1f}  score={state['score']:.1f}  "
                  f"reason={'time cap reached' if sim.env.time > max_time else 'all agents dead'}")
            return state

        actions = []
        for agent_status in state["observations"]:
            agent_status["sim_time"] = state["sim_time"]
            agent_status["n_agents"] = state["num_agents"]
            raw = policy_fn(agent_status, rng)
            actions.append((agent_status["agent_id"], Action(**raw)))


if __name__ == "__main__":
    from rl.eval import POLICIES

    parser = argparse.ArgumentParser()
    parser.add_argument("policy", choices=list(POLICIES.keys()))
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--base_seed", type=int, default=2000)
    args = parser.parse_args()

    for seed in range(args.base_seed, args.base_seed + args.seeds):
        print(f"=== seed {seed} ===")
        diagnose_episode(POLICIES[args.policy], seed)
        print()
