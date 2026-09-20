"""
Observation encoder shared by training (rl/env_wrapper.py) and live inference
(agent_server.py). Only reads fields that exist on the real ObservationResponse
DTO (src/utils/DTOs.py) -- nothing from the simulator's internal Agent object
that wouldn't be available over the API. This keeps train/inference identical.
"""
import numpy as np

BIOME_TYPES = ["forest", "grassland", "swamp", "desert", "river"]

# Nearest-K slots kept per observation type (padded with zeros if fewer seen)
K_FRUIT = 4
K_AGENT = 4
K_PREDATOR = 3
K_TREE = 2
K_EDGE = 4

# Normalization constants, taken from the mutation caps in
# src/elements/environment.py::spawn_agent (chunk_size=400 default)
MAX_SPEED = 20.0
MAX_SPRINT_SPEED = 40.0
MAX_HEARING_RADIUS = 100.0
MAX_VISION_RANGE = 400.0
MAX_VISION_ANGLE = np.pi / 2
MAX_MAX_ENERGY = 1000.0
MAX_AGE_NORM = 150.0  # max_age is randomized in [60,120]; this just bounds the feature
DIST_NORM = 400.0     # normalize all distances by the max possible vision range

SELF_DIM = 8
GLOBAL_DIM = 3   # sim_time fraction, food-decay factor, population count
BIOME_DIM = len(BIOME_TYPES)
OBS_DIM = (
    SELF_DIM + GLOBAL_DIM + BIOME_DIM
    + 4 * K_FRUIT
    + 6 * K_AGENT
    + 6 * K_PREDATOR
    + 4 * K_TREE
    + 4 * K_EDGE
)

MAX_SIM_TIME = 3000.0
POP_NORM = 50.0


def _closest_point_on_segment_to_origin(p1, p2):
    """Closest point to (0,0) on segment p1-p2. Edge coords are already given
    in the creature-local rotated frame (origin = creature, +x = forward)."""
    x1, y1 = p1
    x2, y2 = p2
    dx, dy = x2 - x1, y2 - y1
    denom = dx * dx + dy * dy
    if denom == 0:
        return p1
    t = -(x1 * dx + y1 * dy) / denom
    t = max(0.0, min(1.0, t))
    return (x1 + t * dx, y1 + t * dy)


def _slots_for_type(items, k, include_dir):
    """items: list of dicts with 'distance'/'angle' (+ 'rel_dir' if include_dir).
    Returns a flat list of features, nearest-first, zero-padded to k slots."""
    items_sorted = sorted(items, key=lambda o: o["distance"])[:k]
    feats = []
    for i in range(k):
        if i < len(items_sorted):
            o = items_sorted[i]
            dist_n = min(o["distance"] / DIST_NORM, 1.0)
            ang = o["angle"]
            f = [1.0, dist_n, float(np.sin(ang)), float(np.cos(ang))]
            if include_dir:
                rd = o.get("rel_dir", 0.0)
                f += [float(np.sin(rd)), float(np.cos(rd))]
        else:
            f = [0.0, 0.0, 0.0, 0.0] + ([0.0, 0.0] if include_dir else [])
        feats.extend(f)
    return feats


def encode_agent_observation(agent_status: dict) -> np.ndarray:
    """
    agent_status: dict with the same fields as ObservationResponse
    (agent_id, energy, biome, age, speed, sprint_speed, hearing_radius,
     vision_angle, vision_range, max_energy, observations).

    Returns a fixed-size float32 vector of length OBS_DIM.
    """
    energy = agent_status["energy"]
    max_energy = agent_status["max_energy"]
    age = agent_status["age"]
    speed = agent_status["speed"]
    sprint_speed = agent_status["sprint_speed"]
    hearing_radius = agent_status["hearing_radius"]
    vision_angle = agent_status["vision_angle"]
    vision_range = agent_status["vision_range"]
    biome = agent_status["biome"]
    observations = agent_status["observations"]

    self_feats = [
        energy / max(max_energy, 1e-6),
        min(age / MAX_AGE_NORM, 1.0),
        speed / MAX_SPEED,
        sprint_speed / MAX_SPRINT_SPEED,
        hearing_radius / MAX_HEARING_RADIUS,
        vision_angle / MAX_VISION_ANGLE,
        vision_range / MAX_VISION_RANGE,
        max_energy / MAX_MAX_ENERGY,
    ]

    # Global context (from StepResponse; injected by the caller). The food
    # supply decays as 0.5**(t/300), so hand that curve to the policy directly
    # rather than making it learn the exponential from raw time.
    sim_time = float(agent_status.get("sim_time", 0.0))
    n_agents = float(agent_status.get("n_agents", 1))
    global_feats = [
        min(sim_time / MAX_SIM_TIME, 1.0),
        0.5 ** (sim_time / 300.0),
        min(n_agents / POP_NORM, 2.0),
    ]

    biome_onehot = [1.0 if biome == b else 0.0 for b in BIOME_TYPES]

    fruits = [o for o in observations if o.get("type") == "Fruit"]
    agents = [o for o in observations if o.get("type") == "Agent"]
    predators = [o for o in observations if o.get("type") == "Predator"]
    trees = [o for o in observations if o.get("type") == "Tree"]
    edges_raw = [o for o in observations if o.get("type") == "Edge"]

    edges = []
    for o in edges_raw:
        p1, p2 = o["coords"]
        cx, cy = _closest_point_on_segment_to_origin(p1, p2)
        edges.append({
            "distance": float(np.hypot(cx, cy)),
            "angle": float(np.arctan2(cy, cx)),
        })

    feats = []
    feats += self_feats
    feats += global_feats
    feats += biome_onehot
    feats += _slots_for_type(fruits, K_FRUIT, include_dir=False)
    feats += _slots_for_type(agents, K_AGENT, include_dir=True)
    feats += _slots_for_type(predators, K_PREDATOR, include_dir=True)
    feats += _slots_for_type(trees, K_TREE, include_dir=False)
    feats += _slots_for_type(edges, K_EDGE, include_dir=False)

    vec = np.array(feats, dtype=np.float32)
    assert vec.shape[0] == OBS_DIM, f"expected {OBS_DIM}, got {vec.shape[0]}"
    return vec
