"""
Wraps a trained ActorCritic into the common policy_fn(agent_status, rng) -> dict
interface, so it drops into rl/eval.py the same way potential_field_action
does, and later into agent_server.py for the real submission.
"""
import torch

from rl.obs_encoder import encode_agent_observation
from rl.policy_net import decode_action


def make_rl_policy(net, device="cpu", deterministic=True):
    """Continuous actions are taken at the distribution mean, but SPAWN is
    always sampled from its Bernoulli.

    Thresholding spawn at p>0.5 silently disables reproduction: the teacher
    only spawns on ~0.2% of decisions, so a well-fit policy outputs p far
    below 0.5 almost everywhere. The species then ages out at ~190s even
    though the same weights reproduce fine when sampled during training.
    """
    net.eval()

    def policy_fn(agent_status, rng):
        obs = encode_agent_observation(agent_status)
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            mean, _std, spawn_logit, _value = net.forward(obs_t)
            if deterministic:
                cont_action = mean
            else:
                cont_action, _, _, _ = net.act(obs_t, deterministic=False)[:4]
            spawn_p = torch.sigmoid(spawn_logit)[0].item()

        spawn = 1.0 if (rng.random() if rng else 0.5) < spawn_p else 0.0
        return decode_action(cont_action[0].cpu().numpy(), spawn, agent_status["sprint_speed"])

    return policy_fn


def load_policy(checkpoint_path, obs_dim, device="cpu", deterministic=True):
    from rl.policy_net import ActorCritic
    net = ActorCritic(obs_dim).to(device)
    net.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return make_rl_policy(net, device=device, deterministic=deterministic)
