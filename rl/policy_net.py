"""
Actor-critic network + action encode/decode.

Action layout (3 continuous dims + 1 discrete):
    raw continuous action a in R^3 (unbounded Gaussian, standard PPO setup --
    same as stable-baselines3's default for Box action spaces: don't squash
    the distribution itself, just clip when applying to the env).
    a[0] -> move_distance  = (clip(a[0],-1,1)+1)/2 * agent.sprint_speed
    a[1] -> move_direction = clip(a[1],-1,1) * pi
    a[2] -> turn_angle     = clip(a[2],-1,1) * pi
    spawn ~ Bernoulli(sigmoid(spawn_logit))

move_distance is scaled by the ACTING agent's own sprint_speed (read from its
raw status, not the network), because sprint_speed differs per agent due to
trait mutation -- the network only ever outputs a fraction in [0,1].
"""
import math
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal, Bernoulli


class ActorCritic(nn.Module):
    """Actor and critic keep SEPARATE trunks on purpose.

    They used to share one, and that single fact destroyed four PPO runs: the
    value head starts random, team returns are O(100-900), so its loss is huge
    and its gradients flowed back through the shared layers and wrecked the
    behavior-cloned features. Measured directly -- with the actor's own weights
    frozen (policy_coef=0), the policy still fell from 669.6 to ~140 during
    "critic-only" warmup. Separate trunks make the critic unable to touch the
    actor's representation.
    """

    def __init__(self, obs_dim, hidden=128):
        super().__init__()
        self.actor_trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.critic_trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.mean_head = nn.Linear(hidden, 3)
        self.log_std = nn.Parameter(torch.full((3,), -0.5))  # std ~ 0.6 initially
        self.spawn_head = nn.Linear(hidden, 1)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, obs):
        ha = self.actor_trunk(obs)
        hc = self.critic_trunk(obs)
        mean = self.mean_head(ha)
        std = self.log_std.exp().expand_as(mean)
        spawn_logit = self.spawn_head(ha).squeeze(-1)
        value = self.value_head(hc).squeeze(-1)
        return mean, std, spawn_logit, value

    def act(self, obs, deterministic=False):
        """obs: (B, obs_dim) tensor. Returns cont_action, spawn, logp, value (each (B,...))."""
        mean, std, spawn_logit, value = self.forward(obs)
        if deterministic:
            cont_action = mean
            spawn = (torch.sigmoid(spawn_logit) > 0.5).float()
            return cont_action, spawn, None, value
        dist = Normal(mean, std)
        cont_action = dist.sample()
        spawn_dist = Bernoulli(logits=spawn_logit)
        spawn = spawn_dist.sample()
        logp = dist.log_prob(cont_action).sum(-1) + spawn_dist.log_prob(spawn)
        return cont_action, spawn, logp, value

    def evaluate_actions(self, obs, cont_action, spawn):
        """Recompute log-prob/entropy/value for PPO update (obs requires_grad through net)."""
        mean, std, spawn_logit, value = self.forward(obs)
        dist = Normal(mean, std)
        spawn_dist = Bernoulli(logits=spawn_logit)
        logp = dist.log_prob(cont_action).sum(-1) + spawn_dist.log_prob(spawn)
        entropy = dist.entropy().sum(-1) + spawn_dist.entropy()
        return logp, entropy, value


def decode_action(cont_action: np.ndarray, spawn: float, sprint_speed: float) -> dict:
    a = np.clip(cont_action, -1.0, 1.0)
    # squared mapping: a=0 -> 25% of sprint, a=-1 -> stationary. Standing still
    # costs 1 energy/s against ~5/s for walking, so the neutral action must be
    # cheap or the policy bleeds energy before it has learned anything.
    move_frac = ((a[0] + 1.0) / 2.0) ** 2
    return {
        "move_distance": float(move_frac * sprint_speed),
        "move_direction": float(a[1] * math.pi),
        "turn_angle": float(a[2] * math.pi),
        "spawn_agent": bool(spawn > 0.5),
    }


def encode_action_targets(raw_action: dict, sprint_speed: float) -> np.ndarray:
    """Inverse of decode_action(): a teacher action -> the pre-clip action
    vector that would produce it. Used to behavior-clone the actor so PPO
    starts from a policy that can already forage, instead of from noise."""
    frac = min(max(raw_action["move_distance"] / max(sprint_speed, 1e-6), 0.0), 1.0)
    return np.array([
        2.0 * math.sqrt(frac) - 1.0,
        min(max(raw_action["move_direction"] / math.pi, -1.0), 1.0),
        min(max(raw_action["turn_angle"] / math.pi, -1.0), 1.0),
    ], dtype=np.float32)
