"""
PPO training entry point.

    python -m rl.train --updates 2000 --n_steps 256 --device cuda
    python -m rl.train --updates 2000 --n_steps 256 --workers 8   # parallel envs

Every `eval_every` updates it runs a few fixed-seed episodes through the true
game (rl/eval.py's run_episode, no reward shaping) so progress is measured on
the real objective and is directly comparable to the potential-field baseline
(same seeds 0,1,2 -> baseline mean_score ~464).

--workers N: collect rollouts from N environments in parallel processes
instead of one. This is the actual speed lever -- env stepping is sequential
Python/numpy, not GPU work, so this scales much better than --device cuda
for a network this small. Leave --workers 1 (default) to keep the simple
single-process path.
"""
import argparse
import os
import time

import torch

from rl.env_wrapper import SurvivalEnv
from rl.obs_encoder import OBS_DIM
from rl.policy_net import ActorCritic
from rl.ppo import collect_rollout, build_batch, build_batch_team, ppo_update
from rl.parallel_rollout import ParallelRolloutCollector
from rl.eval import run_episode
from rl.policies.rl_policy import make_rl_policy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--n_steps", type=int, default=256, help="ticks collected per rollout/update")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=1, help="parallel env processes (1 = no parallelism)")
    parser.add_argument("--eval_every", type=int, default=20)
    parser.add_argument("--eval_seeds", type=int, default=3)
    parser.add_argument("--ckpt", default="rl/checkpoints/policy.pt")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--critic_warmup", type=int, default=0,
                        help="updates that train only the value head (use when resuming from a cloned actor)")
    parser.add_argument("--ent_coef", type=float, default=0.0)
    parser.add_argument("--clip_eps", type=float, default=0.1)
    parser.add_argument("--team_return", action="store_true",
                        help="credit every agent with the team's remaining episode score")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.ckpt) or ".", exist_ok=True)

    if args.workers > 1:
        # Leave the CPU to the worker processes; the main process's own net
        # is tiny, it doesn't need multiple BLAS threads and would otherwise
        # contend with the workers for cores.
        torch.set_num_threads(1)

    net = ActorCritic(OBS_DIM).to(args.device)
    if args.resume:
        net.load_state_dict(torch.load(args.resume, map_location=args.device))
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    collector = None
    env = None
    obs = None
    if args.workers > 1:
        collector = ParallelRolloutCollector(args.workers, n_steps=args.n_steps)
        print(f"Started {args.workers} parallel rollout workers.")
    else:
        env = SurvivalEnv(seed=None)
        obs = env.reset()

    best_eval_score = -1e18
    t0 = time.time()
    for update in range(1, args.updates + 1):
        if collector is not None:
            trajectories, obs_after = collector.collect(net)
        else:
            trajectories, obs = collect_rollout(env, net, obs, args.n_steps, args.device)
            obs_after = obs
        builder = build_batch_team if args.team_return else build_batch
        batch = builder(trajectories, obs_after, net, args.device)
        policy_coef = 0.0 if update <= args.critic_warmup else 1.0
        stats = ppo_update(net, optimizer, batch, ent_coef=args.ent_coef,
                           clip_eps=args.clip_eps, policy_coef=policy_coef)

        elapsed = time.time() - t0
        print(
            f"[update {update:5d}] samples={stats['n_samples']:5d} "
            f"policy_loss={stats['policy_loss']:.4f} value_loss={stats['value_loss']:.4f} "
            f"entropy={stats['entropy']:.4f} pc={policy_coef:.0f} elapsed={elapsed:.0f}s"
        )

        if update % args.eval_every == 0:
            policy_fn = make_rl_policy(net, device=args.device, deterministic=True)
            results = [run_episode(policy_fn, seed) for seed in range(args.eval_seeds)]
            mean_score = sum(r["score"] for r in results) / len(results)
            mean_time = sum(r["sim_time"] for r in results) / len(results)
            print(f"  eval @ update {update}: mean_score={mean_score:.2f} mean_sim_time={mean_time:.1f}")

            torch.save(net.state_dict(), args.ckpt)
            if mean_score > best_eval_score:
                best_eval_score = mean_score
                best_path = args.ckpt.replace(".pt", "_best.pt")
                torch.save(net.state_dict(), best_path)
                print(f"  new best ({mean_score:.2f}) -> saved {best_path}")

    if collector is not None:
        collector.close()


if __name__ == "__main__":
    main()
