"""
PPO with parameter sharing across a variable number of agents. Every agent's
(obs, action, reward) at every tick is one training sample for the same
network; population size changing over time is not a special case, it just
changes how many samples a given tick contributes.

Core idea (for readers new to RL):
- The network outputs an action distribution given an observation, plus a
  value estimate V(s) (expected future reward from that state).
- We roll out the current policy for `n_steps` ticks, recording what it did,
  what reward it got, and what V(s) it predicted.
- GAE turns those rewards into an "advantage": was this action better or
  worse than what the critic expected on average from this state?
- PPO nudges the policy towards actions with positive advantage and away
  from negative ones, but *clips* how far it moves in one update so a single
  batch of (possibly noisy) rollouts can't wreck the policy.
"""
from collections import defaultdict

import numpy as np
import torch

from rl.env_wrapper import SurvivalEnv, Action
from rl.policy_net import decode_action


def collect_rollout(env: SurvivalEnv, net, current_obs, n_steps, device):
    """
    Runs the (already-reset) env forward n_steps ticks, resetting internally
    whenever an episode ends. Returns per-agent trajectories (each a list of
    step-dicts in time order) plus the obs to resume from next call.
    """
    trajectories = defaultdict(list)
    episode_idx = 0
    tick = 0  # ids restart at 0 on reset; namespace them per episode

    for _ in range(n_steps):
        agent_ids = list(current_obs.keys())
        obs_batch = torch.as_tensor(
            np.stack([current_obs[aid] for aid in agent_ids]), dtype=torch.float32, device=device
        )
        with torch.no_grad():
            cont_action, spawn, logp, value = net.act(obs_batch)
        cont_action_np = cont_action.cpu().numpy()
        spawn_np = spawn.cpu().numpy()
        logp_np = logp.cpu().numpy()
        value_np = value.cpu().numpy()

        team_value = float(value_np.mean()) if len(value_np) else 0.0

        actions = {}
        for i, aid in enumerate(agent_ids):
            sprint_speed = env.sim.env.agents_dict[aid].sprint_speed
            raw = decode_action(cont_action_np[i], spawn_np[i], sprint_speed)
            actions[aid] = Action(**raw)
            trajectories[(episode_idx, aid)].append({
                "obs": current_obs[aid],
                "cont_action": cont_action_np[i],
                "spawn": spawn_np[i],
                "logp": float(logp_np[i]),
                "value": float(value_np[i]),
                "team_value": team_value,
                "tick": tick,
                "episode": episode_idx,
                "reward": 0.0,
                "done": False,
                "episode_end": False,
            })

        next_obs, rewards, dones, episode_done, info = env.step(actions)

        for aid in agent_ids:
            trajectories[(episode_idx, aid)][-1]["reward"] = rewards[aid]
            trajectories[(episode_idx, aid)][-1]["done"] = dones[aid]

        current_obs = next_obs
        if episode_done:
            # species extinct or clock expired: nothing follows this tick
            for aid in agent_ids:
                trajectories[(episode_idx, aid)][-1]["episode_end"] = True
            current_obs = env.reset()
            episode_idx += 1
        tick += 1

    return trajectories, current_obs


def compute_gae(traj, bootstrap_value, gamma=0.999, lam=0.95):
    """Fills each step dict with 'advantage' and 'return'.

    The last transition bootstraps with, in order of precedence:
      * 0 if the whole episode ended (species extinct / clock expired),
      * the tick's TEAM mean value if this agent died but the species lives on
        -- an agent that spends 100 energy on a child dies a little sooner and
        collects nothing more itself, so bootstrapping its death with 0 makes
        reproduction pure loss and the policy simply never reproduces (our
        first PPO run peaked at 5 agents and died of old age at ~200s),
      * V(next_obs) if the rollout window merely cut the trajectory short.
    """
    T = len(traj)
    last_gae = 0.0
    for t in reversed(range(T)):
        if t == T - 1:
            if traj[t]["episode_end"]:
                next_value, next_non_terminal = 0.0, 0.0
            elif traj[t]["done"]:
                # terminal: bootstrapping a death with the team's value made
                # dying as good as living and the policy stopped avoiding it
                next_value, next_non_terminal = 0.0, 0.0
            else:
                next_value, next_non_terminal = bootstrap_value, 1.0
        else:
            next_value, next_non_terminal = traj[t + 1]["value"], 1.0
        delta = traj[t]["reward"] + gamma * next_value * next_non_terminal - traj[t]["value"]
        last_gae = delta + gamma * lam * next_non_terminal * last_gae
        traj[t]["advantage"] = last_gae
    for t in range(T):
        traj[t]["return"] = traj[t]["advantage"] + traj[t]["value"]
    return traj


def build_batch(trajectories, current_obs, net, device, gamma=0.999, lam=0.95):
    """GAE per agent trajectory, then flatten everything into tensors."""
    with torch.no_grad():
        for key, traj in trajectories.items():
            aid = key[1] if isinstance(key, tuple) else key
            if aid in current_obs and not traj[-1]["episode_end"]:
                obs_t = torch.as_tensor(current_obs[aid], dtype=torch.float32, device=device).unsqueeze(0)
                bootstrap = net.forward(obs_t)[3].item()
            else:
                bootstrap = 0.0
            compute_gae(traj, bootstrap, gamma=gamma, lam=lam)

    obs, cont_action, spawn, logp_old, value_old, advantage, ret = [], [], [], [], [], [], []
    for traj in trajectories.values():
        for step in traj:
            obs.append(step["obs"])
            cont_action.append(step["cont_action"])
            spawn.append(step["spawn"])
            logp_old.append(step["logp"])
            value_old.append(step["value"])
            advantage.append(step["advantage"])
            ret.append(step["return"])

    def T(x, dtype=torch.float32):
        return torch.as_tensor(np.array(x), dtype=dtype, device=device)

    return {
        "obs": T(obs), "cont_action": T(cont_action), "spawn": T(spawn),
        "logp_old": T(logp_old), "value_old": T(value_old),
        "advantage": T(advantage), "return": T(ret),
    }


def ppo_update(net, optimizer, batch, n_epochs=4, mb_size=256, clip_eps=0.2,
                vf_coef=0.5, ent_coef=0.0, max_grad_norm=0.5, policy_coef=1.0):
    """policy_coef=0 trains the critic only.

    Warm-starting the actor from behavior cloning leaves the value head at its
    random init, so GAE advantages are pure noise and PPO walks a good policy
    straight off a cliff (we measured 957 -> 114 in ~300 updates). Fit the
    critic first, then let policy gradients flow.

    ent_coef defaults to 0 for the same reason: an entropy bonus rewards
    becoming more random, which is exactly how the cloned behavior got erased
    (entropy climbed 2.93 -> 3.24 while the score collapsed).
    """
    n = batch["obs"].shape[0]
    adv = batch["advantage"]
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    stats = defaultdict(list)
    idx = np.arange(n)
    for _ in range(n_epochs):
        np.random.shuffle(idx)
        for start in range(0, n, mb_size):
            mb = idx[start:start + mb_size]
            mb = torch.as_tensor(mb, dtype=torch.long, device=batch["obs"].device)

            logp, entropy, value = net.evaluate_actions(
                batch["obs"][mb], batch["cont_action"][mb], batch["spawn"][mb]
            )
            ratio = torch.exp(logp - batch["logp_old"][mb])
            surr1 = ratio * adv[mb]
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv[mb]
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = ((value - batch["return"][mb]) ** 2).mean()
            entropy_loss = -entropy.mean()

            loss = policy_coef * policy_loss + vf_coef * value_loss + ent_coef * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
            optimizer.step()

            stats["policy_loss"].append(policy_loss.item())
            stats["value_loss"].append(value_loss.item())
            stats["entropy"].append(-entropy_loss.item())

    return {k: float(np.mean(v)) for k, v in stats.items()} | {"n_samples": n}


def build_batch_team(trajectories, current_obs, net, device, gamma=0.999, lam=0.95):
    """Team-return credit assignment.

    Every agent alive at tick t is given the TEAM's discounted future score
    from t onwards -- computed on a single per-episode timeline that keeps
    running after any individual agent dies.

    This is what makes reproduction learnable without inventing a bonus for
    it. A birth is worth exactly however much it extends the episode, and a
    death costs exactly however much it shortens it (a lot when the
    population is small, almost nothing when it is large) -- which is the
    real structure of the game, since score accrues at dt per tick no matter
    how many agents are alive.

    The three earlier attempts all failed on hand-made stand-ins for this:
    per-agent energy shaping (too myopic to see a birth pay off), death
    bootstrapped with the team's value (made dying free), and a flat +30 per
    birth (got farmed until the population starved).
    """
    # --- rebuild the per-episode team timeline from the tagged samples ------
    ticks = {}  # (episode, tick) -> {"r": team reward, "v": mean value, "end": bool}
    for traj in trajectories.values():
        for step in traj:
            key = (step["episode"], step["tick"])
            slot = ticks.setdefault(key, {"r": step["reward"], "v": step["team_value"], "end": False})
            slot["end"] = slot["end"] or step["episode_end"]

    with torch.no_grad():
        if current_obs:
            obs_t = torch.as_tensor(np.stack(list(current_obs.values())),
                                    dtype=torch.float32, device=device)
            tail_value = float(net.forward(obs_t)[3].mean().item())
        else:
            tail_value = 0.0

    # --- GAE along that timeline, one sequence per episode -----------------
    adv, ret = {}, {}
    for episode in sorted({k[0] for k in ticks}):
        seq = sorted(k for k in ticks if k[0] == episode)
        last_gae = 0.0
        for i in range(len(seq) - 1, -1, -1):
            cur = ticks[seq[i]]
            if i == len(seq) - 1:
                next_v = 0.0 if cur["end"] else tail_value
                non_terminal = 0.0 if cur["end"] else 1.0
            else:
                next_v, non_terminal = ticks[seq[i + 1]]["v"], 1.0
            delta = cur["r"] + gamma * next_v * non_terminal - cur["v"]
            last_gae = delta + gamma * lam * non_terminal * last_gae
            adv[seq[i]] = last_gae
            ret[seq[i]] = last_gae + cur["v"]

    obs, cont_action, spawn, logp_old, value_old, advantage, returns = [], [], [], [], [], [], []
    for traj in trajectories.values():
        for step in traj:
            key = (step["episode"], step["tick"])
            obs.append(step["obs"])
            cont_action.append(step["cont_action"])
            spawn.append(step["spawn"])
            logp_old.append(step["logp"])
            value_old.append(step["value"])
            advantage.append(adv[key])
            returns.append(ret[key])

    def T(x):
        return torch.as_tensor(np.array(x), dtype=torch.float32, device=device)

    return {
        "obs": T(obs), "cont_action": T(cont_action), "spawn": T(spawn),
        "logp_old": T(logp_old), "value_old": T(value_old),
        "advantage": T(advantage), "return": T(returns),
    }
