"""
Small MLP policy: obs -> tanh hidden layer -> 4 outputs. NumPy-only forward
pass so ES workers stay light (no torch import per process).

A linear policy was tried first and topped out at ~184 vs a ~578 teacher: the
teacher's steering is atan2 of an inverse-distance-weighted vector sum, which
a linear map fundamentally cannot represent. One hidden layer fixes the
capacity problem at the cost of more ES dimensions.

Same decode()/encode_targets() contract as linear_policy, so cloning,
rl/eval.py and agent_server.py all work unchanged.
"""
import math

import numpy as np

from rl.obs_encoder import OBS_DIM, encode_agent_observation

HIDDEN = 24
N_OUT = 4
MOVE_BIAS = 1.0

# W1 (OBS_DIM x HIDDEN), b1 (HIDDEN), W2 (HIDDEN x N_OUT), b2 (N_OUT)
N_PARAMS = OBS_DIM * HIDDEN + HIDDEN + HIDDEN * N_OUT + N_OUT


def unpack(theta: np.ndarray):
    theta = np.asarray(theta, dtype=np.float64)
    i = 0
    W1 = theta[i:i + OBS_DIM * HIDDEN].reshape(OBS_DIM, HIDDEN); i += OBS_DIM * HIDDEN
    b1 = theta[i:i + HIDDEN]; i += HIDDEN
    W2 = theta[i:i + HIDDEN * N_OUT].reshape(HIDDEN, N_OUT); i += HIDDEN * N_OUT
    b2 = theta[i:i + N_OUT]
    return W1, b1, W2, b2


def pack(W1, b1, W2, b2) -> np.ndarray:
    return np.concatenate([
        np.asarray(W1, dtype=np.float64).reshape(-1),
        np.asarray(b1, dtype=np.float64).reshape(-1),
        np.asarray(W2, dtype=np.float64).reshape(-1),
        np.asarray(b2, dtype=np.float64).reshape(-1),
    ])


def forward(params, obs: np.ndarray) -> np.ndarray:
    W1, b1, W2, b2 = params
    h = np.tanh(obs @ W1 + b1)
    return h @ W2 + b2


def decode(out: np.ndarray, sprint_speed: float) -> dict:
    move_frac = 1.0 / (1.0 + math.exp(-(float(out[0]) - MOVE_BIAS)))
    return {
        "move_distance": float(move_frac * sprint_speed),
        "move_direction": float(math.tanh(float(out[1])) * math.pi),
        "turn_angle": float(math.tanh(float(out[2])) * math.pi),
        "spawn_agent": bool(out[3] > 0.0),
    }


def encode_targets(raw_action: dict, sprint_speed: float) -> np.ndarray:
    eps = 1e-3
    frac = min(max(raw_action["move_distance"] / max(sprint_speed, 1e-6), eps), 1 - eps)
    d = min(max(raw_action["move_direction"] / math.pi, -0.999), 0.999)
    t = min(max(raw_action["turn_angle"] / math.pi, -0.999), 0.999)
    return np.array([
        math.log(frac / (1 - frac)) + MOVE_BIAS,
        math.atanh(d),
        math.atanh(t),
        2.0 if raw_action["spawn_agent"] else -2.0,
    ], dtype=np.float64)


def make_mlp_policy(theta: np.ndarray):
    params = unpack(theta)

    def policy_fn(agent_status, rng=None):
        obs = encode_agent_observation(agent_status).astype(np.float64)
        return decode(forward(params, obs), agent_status["sprint_speed"])

    return policy_fn
