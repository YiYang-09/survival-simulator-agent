"""
Loads the best parameter vector found by rl/evolve_potential_field.py
(rl/checkpoints/pf_params_best.json) and exposes it as a ready-to-use
policy_fn, same interface as potential_field_action.
"""
import json

from rl.policies.potential_field import DEFAULT_PARAMS, potential_field_action

BEST_PARAMS_PATH = "rl/checkpoints/pf_params_best.json"


def load_tuned_params(path: str = BEST_PARAMS_PATH) -> dict:
    with open(path) as f:
        data = json.load(f)
    params = dict(DEFAULT_PARAMS)
    params.update(data["params"])
    return params


def make_tuned_policy(path: str = BEST_PARAMS_PATH):
    params = load_tuned_params(path)

    def policy_fn(agent_status, rng):
        return potential_field_action(agent_status, rng, params=params)

    return policy_fn
