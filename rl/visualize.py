"""
Watch a policy play, rendered with pygame (same renderer as local_playground.py).

    python -m rl.visualize --policy potential_field
    python -m rl.visualize --policy rl --ckpt rl/checkpoints/policy_best.pt
    python -m rl.visualize --policy random
"""
import argparse
import random

import pygame

from src.core import SimulationCore
from rl.env_wrapper import Action
from rl.policies.potential_field import potential_field_action


def _random_policy(agent_status, rng):
    return {
        "move_distance": rng.uniform(0.0, agent_status["sprint_speed"]),
        "move_direction": 0.0,
        "turn_angle": rng.uniform(-3.14159 / 4, 3.14159 / 4),
        "spawn_agent": True,
    }


def visualize(policy_fn, seed=None):
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    sim = SimulationCore(seed=seed)
    rng = random.Random(seed)

    pygame.init()
    info = pygame.display.Info()
    env_ratio = sim.env_width / sim.env_height
    screen_height = int(info.current_h * 0.9)
    screen_width = int(screen_height * env_ratio)
    screen = pygame.display.set_mode((screen_width, screen_height), pygame.SCALED)
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)

    running = True
    actions = []
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        state = sim.step(actions)

        actions = []
        for agent_status in state["observations"]:
            raw = policy_fn(agent_status, rng)
            actions.append((agent_status["agent_id"], Action(**raw)))

        sim.env.draw(screen)
        img = font.render(f'Score: {state["score"]:.2f}', True, (255, 255, 255))
        screen.blit(img, (20, 20))
        pygame.display.flip()
        clock.tick(60)

        print(f'Score: {state["score"]:.2f} | Agents alive: {state["num_agents"]:.0f} | Time: {sim.env.time:.2f}')

        if state["num_agents"] == 0 or sim.env.time > 3000:
            print(f"Game over! Final Score: {state['score']}")
            print(f"Seed: {seed}")
            running = False

    pygame.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=["potential_field", "rl", "random"], default="potential_field")
    parser.add_argument("--ckpt", default="rl/checkpoints/policy_best.pt")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.policy == "potential_field":
        fn = potential_field_action
    elif args.policy == "random":
        fn = _random_policy
    else:
        from rl.obs_encoder import OBS_DIM
        from rl.policies.rl_policy import load_policy
        fn = load_policy(args.ckpt, OBS_DIM, device="cpu", deterministic=True)

    visualize(fn, seed=args.seed)
