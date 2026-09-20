"""
Linear policy (single weight matrix, NumPy only -- no torch, so ES workers
stay light and fast to fork).

    obs (OBS_DIM) -> [move_logit, dir_raw, turn_raw, spawn_logit]

Decoding is chosen so that the energy-cheap action is easy to express:
    move_distance  = sprint_speed * sigmoid(move_logit - MOVE_BIAS)
    move_direction = tanh(dir_raw) * pi
    turn_angle     = tanh(turn_raw) * pi
    spawn_agent    = spawn_logit > 0

Linear policies are a deliberate choice, not a placeholder: with episodes this
expensive, a ~400-parameter policy is worth far more ES iterations than a deep
net, and linear policies are known to be competitive on continuous control
(Mania et al. 2018, "Simple random search...").
"""
import math

import numpy as np

from rl.obs_encoder import OBS_DIM, encode_agent_observation

N_OUT = 4
MOVE_BIAS = 1.0  # sigmoid(0 - 1.0) ~= 0.27, so "mostly idle" is the neutral action
N_PARAMS = (OBS_DIM + 1) * N_OUT  # +1 for the bias row


def unpack(theta: np.ndarray) -> np.ndarray:
    return np.asarray(theta, dtype=np.float64).reshape(OBS_DIM + 1, N_OUT)


def pack(W: np.ndarray) -> np.ndarray:
    return np.asarray(W, dtype=np.float64).reshape(-1)


def forward(W: np.ndarray, obs: np.ndarray) -> np.ndarray:
    return obs @ W[:-1] + W[-1]


def decode(out: np.ndarray, sprint_speed: float) -> dict:
    move_frac = 1.0 / (1.0 + math.exp(-(float(out[0]) - MOVE_BIAS)))
    return {
        "move_distance": float(move_frac * sprint_speed),
        "move_direction": float(math.tanh(float(out[1])) * math.pi),
        "turn_angle": float(math.tanh(float(out[2])) * math.pi),
        "spawn_agent": bool(out[3] > 0.0),
    }


def encode_targets(raw_action: dict, sprint_speed: float) -> np.ndarray:
    """Inverse of decode(): teacher action -> pre-activation targets, for
    least-squares behavior cloning."""
    eps = 1e-3
    frac = min(max(raw_action["move_distance"] / max(sprint_speed, 1e-6), eps), 1 - eps)
    move_logit = math.log(frac / (1 - frac)) + MOVE_BIAS

    d = min(max(raw_action["move_direction"] / math.pi, -0.999), 0.999)
    t = min(max(raw_action["turn_angle"] / math.pi, -0.999), 0.999)

    return np.array([
        move_logit,
        math.atanh(d),
        math.atanh(t),
        2.0 if raw_action["spawn_agent"] else -2.0,
    ], dtype=np.float64)


def make_linear_policy(theta: np.ndarray):
    """policy_fn(agent_status, rng) -> raw action dict, same interface as the
    rule-based policies, so rl/eval.py and agent_server.py can use it as-is."""
    W = unpack(theta)

    def policy_fn(agent_status, rng=None):
        obs = encode_agent_observation(agent_status).astype(np.float64)
        return decode(forward(W, obs), agent_status["sprint_speed"])

    return policy_fn
