"""
Multiprocessing rollout collection: N worker processes each run their own
SurvivalEnv, using a CPU copy of the current policy for action selection.
The main process only holds the network+optimizer and does the PPO update.

This targets the real bottleneck (env stepping is sequential Python/numpy)
rather than the network compute, which is tiny and not worth putting on GPU
at this scale.

Windows uses the 'spawn' start method, so the worker function must be a
plain module-level function (picklable) -- no closures/lambdas.
"""
import multiprocessing as mp
import time

import torch

from rl.env_wrapper import SurvivalEnv
from rl.obs_encoder import OBS_DIM
from rl.policy_net import ActorCritic
from rl.ppo import collect_rollout


def _worker_loop(worker_id, obs_dim, n_steps, conn):
    torch.set_num_threads(1)  # avoid oversubscription: many workers, each single-threaded
    # Stagger startup: env creation briefly spikes memory (map generation),
    # so avoid every worker hitting that peak in the same instant.
    time.sleep(worker_id * 0.15)
    net = ActorCritic(obs_dim)
    net.eval()
    env = SurvivalEnv(seed=None)
    obs = env.reset()

    while True:
        msg = conn.recv()
        if msg is None:  # shutdown
            break
        net.load_state_dict(msg)
        trajectories, obs = collect_rollout(env, net, obs, n_steps, device="cpu")
        # namespace agent ids by worker so different workers' ids never collide
        trajectories = {(worker_id, aid): traj for aid, traj in trajectories.items()}
        obs_out = {(worker_id, aid): o for aid, o in obs.items()}
        conn.send((trajectories, obs_out))
    conn.close()


class ParallelRolloutCollector:
    def __init__(self, n_workers, n_steps, obs_dim=OBS_DIM):
        self.parent_conns = []
        self.procs = []
        ctx = mp.get_context("spawn")
        for wid in range(n_workers):
            parent_conn, child_conn = ctx.Pipe()
            p = ctx.Process(
                target=_worker_loop, args=(wid, obs_dim, n_steps, child_conn), daemon=True
            )
            p.start()
            self.parent_conns.append(parent_conn)
            self.procs.append(p)

    def collect(self, net):
        state_dict = {k: v.detach().cpu() for k, v in net.state_dict().items()}
        for conn in self.parent_conns:
            conn.send(state_dict)

        all_trajectories, all_obs = {}, {}
        for conn in self.parent_conns:
            trajectories, obs = conn.recv()
            all_trajectories.update(trajectories)
            all_obs.update(obs)
        return all_trajectories, all_obs

    def close(self):
        for conn in self.parent_conns:
            try:
                conn.send(None)
            except (BrokenPipeError, OSError):
                pass
        for p in self.procs:
            p.join(timeout=5)
