"""
Gym-style wrapper around SimulationCore for multi-agent training with a single
shared (parameter-sharing) policy. One wrapper .step() call advances the whole
simulation by one tick and returns per-agent (obs, reward, done) for every
agent that was alive at the *start* of the tick.

Not a pydantic ActionRequest on purpose: pydantic validation is overhead we
don't want inside a hot training loop. agent_server.py is the only place that
needs to speak ActionRequest (the actual API boundary).
"""
from collections import namedtuple

from src.core import SimulationCore
from rl.obs_encoder import encode_agent_observation, OBS_DIM

Action = namedtuple("Action", ["move_distance", "move_direction", "turn_angle", "spawn_agent"])

# Reward shaping constants (reward_mode="energy" only).
ENERGY_REWARD_SCALE = 1.0 / 100.0   # net energy delta this tick, scaled down
TICK_BONUS = 0.01                   # small constant reward for staying alive
DEATH_PENALTY_SCALE = 2.0           # multiplied by (last_energy / max_energy)

# reward_mode="team" gives every living agent the change in the REAL game score
# (dt per tick + fruit/1000 - eaten_energy/100). That is the competition
# objective verbatim, so there is no shaping to misalign. Scaled up because a
# tick is only worth 0.1.
TEAM_REWARD_SCALE = 1.0

# reward_mode="survive": the formulation that survived the post-mortems.
#   +ALIVE_REWARD per tick while alive, 0 bootstrap at death.
#   +SPAWN_BONUS when the agent actually produces a child.
#
# Why not the two that failed:
#   * per-agent energy shaping with gamma=0.99 could only see 10 seconds, so
#     reproduction (which pays back over ~90s) was invisible -> died at ~200s.
#   * team reward with "bootstrap death with the team's value" made dying cost
#     exactly as much as living, so the policy stopped avoiding death at all
#     -> collapsed to ~108.
# Keeping death terminal restores the cost of dying; SPAWN_BONUS pays the
# parent for the part of a child's value it can never collect itself. A child
# lives ~90s = ~90 points of team reward; the 100 energy a birth costs shortens
# the parent's own life by roughly 25s. 30 sits between those.
ALIVE_REWARD = 0.1
SPAWN_BONUS = 30.0


class SurvivalEnv:
    """
    reset() -> obs: {agent_id: np.ndarray[OBS_DIM]}
    step(actions: {agent_id: Action}) -> obs, rewards, dones, info
        obs, rewards, dones: dicts keyed by agent_id (only agents alive at the
            START of this tick get an entry; newly spawned agents appear in the
            NEXT tick's obs dict once they've taken their first action)
        info: {"score": float, "sim_time": float, "num_agents": int,
               "spawned_ids": [...], "died_ids": [...]}
    """

    def __init__(self, seed=None, dt=1 / 10, max_time=3000.0, core_kwargs=None, reward_mode="team"):
        self.reward_mode = reward_mode
        self.seed = seed
        self.dt = dt
        self.max_time = max_time
        self.core_kwargs = core_kwargs or {}
        self.sim = None

    def reset(self, seed=None):
        if seed is not None:
            self.seed = seed
        self.sim = SimulationCore(seed=self.seed, dt=self.dt, **self.core_kwargs)
        obs = {}
        for agent in self.sim.env.agents:
            status = self.sim.env.get_agent_state(agent.agent_id)
            status["sim_time"] = self.sim.env.time
            status["n_agents"] = len(self.sim.env.agents)
            obs[agent.agent_id] = encode_agent_observation(status)
        return obs

    def step(self, actions: dict):
        env = self.sim.env

        # Snapshot pre-step energy/max_energy for reward + death-penalty calc
        energy_before = {aid: a.energy for aid, a in env.agents_dict.items()}
        max_energy_before = {aid: a.max_energy for aid, a in env.agents_dict.items()}
        score_before = env.score

        action_list = [
            (aid, act) for aid, act in actions.items() if aid in env.agents_dict
        ]
        state = self.sim.step(action_list)

        new_agents_dict = env.agents_dict

        obs, rewards, dones = {}, {}, {}

        # Agents alive at the start of this tick: reward + (obs if still alive)
        team_reward = (state["score"] - score_before) * TEAM_REWARD_SCALE

        for aid in energy_before:
            if aid in new_agents_dict:
                agent = new_agents_dict[aid]
                if self.reward_mode == "survive":
                    r = ALIVE_REWARD
                    act = actions.get(aid)
                    # agent_step() spawns iff the action asked for it and the
                    # agent still had >100 energy, which shows up as a ~100
                    # energy drop beyond ordinary movement costs
                    if act is not None and act.spawn_agent and                             (energy_before[aid] - agent.energy) > 95.0:
                        r += SPAWN_BONUS
                    rewards[aid] = r
                elif self.reward_mode == "team":
                    rewards[aid] = team_reward
                else:
                    energy_delta = agent.energy - energy_before[aid]
                    rewards[aid] = energy_delta * ENERGY_REWARD_SCALE + TICK_BONUS
                dones[aid] = False
                status = env.get_agent_state(aid)
                status["sim_time"] = state["sim_time"]
                status["n_agents"] = state["num_agents"]
                obs[aid] = encode_agent_observation(status)
            else:
                if self.reward_mode == "survive":
                    rewards[aid] = 0.0   # death is terminal; the loss is the
                                         # stream of ALIVE_REWARD it forfeits
                elif self.reward_mode == "team":
                    # the score penalty for being eaten is already inside
                    # team_reward; no extra hand-made death term
                    rewards[aid] = team_reward
                else:
                    frac = energy_before[aid] / max(max_energy_before[aid], 1e-6)
                    rewards[aid] = -DEATH_PENALTY_SCALE * max(frac, 0.0)
                dones[aid] = True

        spawned_ids = [aid for aid in new_agents_dict if aid not in energy_before]
        died_ids = [aid for aid in energy_before if aid not in new_agents_dict]

        # Newborns: give them an obs entry so the caller can start acting on
        # them next tick, but no reward yet (they had no "before" state).
        for aid in spawned_ids:
            status = env.get_agent_state(aid)
            status["sim_time"] = state["sim_time"]
            status["n_agents"] = state["num_agents"]
            obs[aid] = encode_agent_observation(status)

        info = {
            "score": state["score"],
            "sim_time": state["sim_time"],
            "num_agents": state["num_agents"],
            "spawned_ids": spawned_ids,
            "died_ids": died_ids,
        }
        episode_done = state["num_agents"] == 0 or state["sim_time"] > self.max_time

        return obs, rewards, dones, episode_done, info
