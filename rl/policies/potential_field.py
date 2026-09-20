"""
Rule-based controller: potential-field steering + a phase-aware state machine.

Grounded in the actual game economy (src/elements/*):
  * score += dt every tick regardless of population -> survival time is ~the
    entire objective (3000 max); a fruit is worth at most 0.06.
  * walking at full speed costs 0.05/unit = ~5 energy/s, vs ~1 energy/s just
    being alive -> moving is the dominant cost, so standing still when there
    is nothing to chase is the single biggest saving available.
  * agents die of old age (max_age 60-120s, then -0.01*age per TICK), so the
    species only survives via an unbroken chain of ~25-40 generations.
    Reproduction can never be switched off completely.
  * tree spawn rate decays as 0.5**(t/300), so food supply collapses over
    time; equilibrium tree count ~122*sqrt(0.5**(t/300)) (~22 at t=1500,
    ~4 at t=3000). Carrying capacity therefore falls with time, which is why
    the population target here decays too.

Works on the raw ObservationResponse-shaped dict. Callers additionally inject
"sim_time" and "n_agents" (both available on StepResponse) -- both are read
with safe fallbacks so an older caller still works.

All observation angles are already relative to the agent's own facing
direction (0 = straight ahead) -- see Creature.relative_distance_angle.
"""
import math
import random

DEFAULT_PARAMS = {
    # steering weights
    "W_FRUIT": 3.0,
    "W_PREDATOR": 8.0,
    "W_WALL": 2.0,
    "W_TREE": 0.4,          # tree pull while fruit is in sight
    "W_TREE_NOFRUIT": 3.0,  # tree pull when no fruit is visible (camp by the fruit factory)

    # danger / energy thresholds
    "DANGER_MULT": 1.5,
    "LOW_ENERGY_FRAC": 0.25,

    # Predator.step() only charges when the agent is facing AWAY from it
    # ("Only chase when behind"), or when closer than its hearing_radius*1.5
    # (=90). So outside that range, keep facing the predator and it just
    # circles instead of attacking -- retreat while looking at it.
    "FACE_PREDATOR_DIST": 100.0,

    # reproduction: gated by energy AND by a population target that decays
    # with time, but never below POP_TARGET_MIN (an unbroken chain matters
    # more than avoiding overshoot -- under-reproducing wipes the run out).
    "SPAWN_ENERGY_FRAC": 0.6,
    "SPAWN_MIN_ABS": 150.0,
    # max_age is 60-120s and past it energy drains by 0.01*age per TICK, so an
    # old agent is about to lose everything it is carrying anyway (and if a
    # predator takes it, the leftover energy is subtracted from the score).
    # Past this age, dump energy into a child as soon as it is affordable.
    "OLD_AGE_SPAWN": 55.0,
    # Artificial selection. Children inherit traits with a 10% chance of a
    # +-50% mutation per trait, capped at 2x the starting values (vision
    # 200->400, hearing 50->100, max_energy 500->1000). Better senses cost
    # NOTHING to run -- energy burn depends only on distance moved and time
    # alive -- so letting only above-threshold agents breed ratchets the
    # lineage upward over the ~25 generations a long run needs. Bypassed when
    # the population is at its floor, because an unbroken chain beats a
    # better-eyed extinction.
    "BREED_TRAIT_MIN": 2.9,
    "POP_TARGET_0": 40.0,
    "POP_HALFLIFE": 600.0,
    "POP_TARGET_MIN": 6.0,

    # movement throttling (fractions of the agent's walk speed)
    "FORAGE_SPEED_FRAC": 1.0,   # heading for visible food
    "SEARCH_SPEED_FRAC": 0.6,   # nothing visible, still looking
    "IDLE_MOVE_FRAC": 0.1,      # low energy: conserve
    "MAX_TURN": math.pi / 6,
}


def _closest_point_on_segment_to_origin(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    dx, dy = x2 - x1, y2 - y1
    denom = dx * dx + dy * dy
    if denom == 0:
        return p1
    t = max(0.0, min(1.0, -(x1 * dx + y1 * dy) / denom))
    return (x1 + t * dx, y1 + t * dy)


def potential_field_action(agent_status: dict, rng: random.Random = None, params: dict = None) -> dict:
    """Returns {'move_distance', 'move_direction', 'turn_angle', 'spawn_agent'}."""
    rng = rng or random.Random()
    p = params or DEFAULT_PARAMS

    energy = agent_status["energy"]
    max_energy = agent_status["max_energy"]
    speed = agent_status["speed"]
    sprint_speed = agent_status["sprint_speed"]
    hearing_radius = agent_status["hearing_radius"]
    age = agent_status["age"]
    vision_range = agent_status["vision_range"]
    observations = agent_status["observations"]
    # injected global context (present on StepResponse); safe fallbacks
    sim_time = agent_status.get("sim_time", 0.0)
    n_agents = agent_status.get("n_agents", 1)

    energy_frac = energy / max(max_energy, 1e-6)

    fruits = [o for o in observations if o.get("type") == "Fruit"]
    predators = [o for o in observations if o.get("type") == "Predator"]
    trees = [o for o in observations if o.get("type") == "Tree"]
    edges = [o for o in observations if o.get("type") == "Edge"]

    vx, vy = 0.0, 0.0
    danger = False
    danger_dist = max(hearing_radius * p["DANGER_MULT"], 60.0)

    nearest_pred = None
    for pr in predators:
        d = max(pr["distance"], 1e-3)
        if d < danger_dist:
            danger = True
        if nearest_pred is None or d < nearest_pred["distance"]:
            nearest_pred = pr
        f = p["W_PREDATOR"] / (d ** 1.5)
        vx -= f * math.cos(pr["angle"])
        vy -= f * math.sin(pr["angle"])

    for fr in fruits:
        d = max(fr["distance"], 1e-3)
        f = p["W_FRUIT"] / d
        vx += f * math.cos(fr["angle"])
        vy += f * math.sin(fr["angle"])

    # trees pull harder when there is no fruit in sight: fruit only spawns
    # around trees, so waiting next to one beats wandering (wandering costs
    # ~5 energy/s, a fruit is only worth 20-60)
    w_tree = p["W_TREE"] if fruits else p["W_TREE_NOFRUIT"]
    for t in trees:
        d = max(t["distance"], 1e-3)
        f = w_tree / d
        vx += f * math.cos(t["angle"])
        vy += f * math.sin(t["angle"])

    for e in edges:
        p1, p2 = e["coords"]
        cx, cy = _closest_point_on_segment_to_origin(p1, p2)
        d = max(math.hypot(cx, cy), 1e-3)
        angle = math.atan2(cy, cx)
        f = p["W_WALL"] / (d ** 2)
        vx -= f * math.cos(angle)
        vy -= f * math.sin(angle)

    mag = math.hypot(vx, vy)
    has_target = mag > 1e-6
    desired_heading = math.atan2(vy, vx) if has_target else rng.uniform(-0.3, 0.3)

    max_turn = p["MAX_TURN"]
    turn_angle = max(-max_turn, min(max_turn, desired_heading))
    move_direction = desired_heading

    # Retreat while watching: move away (desired_heading already points away
    # from the predator) but turn TOWARDS it, because a predator that is being
    # looked at circles instead of charging. Pointless once it is inside its
    # charge radius, where it attacks regardless.
    if nearest_pred is not None and nearest_pred["distance"] > p["FACE_PREDATOR_DIST"]:
        turn_angle = max(-max_turn, min(max_turn, nearest_pred["angle"]))

    # Speed: flee at full sprint, chase visible food, otherwise throttle hard.
    if danger:
        move_distance = sprint_speed
    elif energy_frac < p["LOW_ENERGY_FRAC"]:
        move_distance = speed * p["IDLE_MOVE_FRAC"]
    elif fruits:
        move_distance = speed * p["FORAGE_SPEED_FRAC"]
    else:
        move_distance = speed * p["SEARCH_SPEED_FRAC"]

    # Only skip the turn cost when effectively parked. Suppressing the turn
    # while still walking is a trap: with no wall/tree force the agent walks
    # in a straight line forever, jams against an obstacle and starves.
    if not has_target and move_distance < 0.05 * speed:
        turn_angle = 0.0
        move_direction = 0.0

    # Population target shrinks with the food supply, floored so the
    # generation chain can't break.
    pop_target = p["POP_TARGET_0"] * (0.5 ** (sim_time / max(p["POP_HALFLIFE"], 1e-6)))
    pop_target = max(p["POP_TARGET_MIN"], pop_target)

    # trait score, normalized so a freshly spawned baseline agent scores 3.0
    trait_score = (
        vision_range / 200.0
        + hearing_radius / 50.0
        + max_energy / 500.0
    )
    good_stock = trait_score >= p["BREED_TRAIT_MIN"] or n_agents <= p["POP_TARGET_MIN"]

    old_enough_to_cash_out = age > p["OLD_AGE_SPAWN"]
    spawn_agent = (
        not danger
        and energy > p["SPAWN_MIN_ABS"]
        and n_agents < pop_target
        and good_stock
        and (energy_frac > p["SPAWN_ENERGY_FRAC"] or old_enough_to_cash_out)
    )

    return {
        "move_distance": move_distance,
        "move_direction": move_direction,
        "turn_angle": turn_angle,
        "spawn_agent": spawn_agent,
    }
