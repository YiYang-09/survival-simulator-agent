# Survival Simulator Agent — Nordic AI Cup 2026

My entry for the **survival-simulator** use case of the [Nordic AI Cup 2026](https://nordicaicup.com),
hosted by [Ambolt AI](https://ambolt.io/) (17–20 September 2026), with WASP as the Swedish partner.

**Final result: 1016.88 points, 42nd of 82 teams.**

The task: you control an entire species of herbivores in a partially observable 2D world.
Every tick the server sends each agent's local observations and you return its action —
move, turn, and whether to reproduce. The run ends when the species goes extinct or 3000
simulated seconds elapse. Your solution is served as a FastAPI `/predict` endpoint that the
organizers' evaluator calls over HTTP.

---

## What the final agent actually is

Not a neural network. A **potential-field controller with a phase-aware state machine**,
whose ~19 constants were partly tuned by an evolution strategy, plus three behavioural rules
derived by reading the simulator's source.

```
observations ──▶ force summation ──▶ desired heading
  fruit  (+)      state machine:  flee / conserve / forage ──▶ speed
  trees  (+)      reproduction gate ──▶ spawn?
  predators (−)
  walls  (−)
```

Reading the scoring code first turned out to matter more than any amount of tuning:

```python
self.score += dt                  # every tick, regardless of population size
self.score += fruit.energy / 1000 # a fruit is worth at most 0.06
self.score -= agent.energy / 100  # being eaten costs the energy you were carrying
```

So the objective is almost entirely **how long the species stays alive** — up to 3000 points
from survival, against tens of points from everything else. That reframes the problem from
"forage efficiently" to "never let the lineage break".

### The three rules that produced the real gains

Each came from reading `src/elements/`, not from search — the controller had no way to express
any of them, so no amount of parameter tuning could have found them.

| Mechanic found in the source | Rule |
|---|---|
| `Predator.step()` only charges when the agent faces **away** from it (`abs(agent_looking_dir) > π/2`), and otherwise circles | Retreat while **facing** the predator; sprint only inside its ~90-unit charge radius |
| `max_age` is 60–120s, after which energy drains by `0.01 * age` **per tick** | Old agents dump their remaining energy into a child before it evaporates |
| Traits are inherited with ±50% mutation, capped at 2× the base values, and better senses cost **nothing** to run | Only let agents above a trait threshold reproduce — artificial selection over ~25 generations |

Measured on 20 held-out seeds against the previously deployed version: **763.5 → 914.2 mean**,
and the worst case improved too (513 → 556), which matters because the final evaluation is
the average of only three runs.

---

## What did not work

Kept here deliberately — the failures were more instructive than the win.

### PPO: five attempts, five collapses

| Attempt | Setup | Result | Root cause |
|---|---|---|---|
| 1 | Per-agent energy shaping, γ=0.99 | dies at ~200s | γ=0.99 at 10 ticks/s is a **10-second horizon**; reproduction pays back over ~90s, so it was invisible. The policy never banked the 100 energy a birth costs and the population never grew past the 5 founders. |
| 2 | True Δscore reward, death bootstrapped with the team's value | 957 → 108 | Bootstrapping a death with the team's value made dying worth as much as living, so the policy stopped avoiding death. |
| 3 | Survive reward + flat +30 per birth | 736 → 85 | The bonus got farmed: +30 is worth 300s of survival, so the policy over-reproduced until the food ran out. |
| 4 | Team-return credit assignment (no hand-set constants) | 928 → 99 | Same collapse as 2 and 3 despite a completely different reward. |
| 5 | Separate actor/critic trunks | 911 → 564 → 20 | Collapse fixed, degradation not. |

Four different reward formulations converging on the *same* failure was the clue: the problem
could not be the reward. The common factor was the network.

**The actual bug:** actor and critic shared a trunk. The value head starts random and team
returns are O(100–900), so its loss dominates and its gradients flow back through the shared
layers, destroying the behavior-cloned features. Proven with a controlled experiment — with
the actor's own weights **frozen** (`policy_coef=0`), the policy still fell from **669.6 to ~140**
during "critic-only" warmup. Splitting the trunks stopped the collapse (911 during warmup
instead of 140), but policy-gradient updates still drifted the policy downwards, and with the
deadline approaching I stopped rather than spend the remaining budget on a sixth attempt.

### Neuroevolution: ruled out by arithmetic, not by trying harder

An episode takes 1–2 CPU-minutes, so the remaining budget was ~4,000 episodes. Black-box
search needs roughly 10–50 evaluations per dimension:

- 19-parameter controller → 200–1,000 evaluations → **fits**
- 2,476-parameter MLP → 25,000–125,000 evaluations → **off by ~20–100×**

Measured behaviour matched: 7 ES iterations moved the weight norm by 0.3% and the best score
not at all. Behavior cloning the same network from the rule-based controller scored 783 on 20
seeds — a real neural policy, just not a better one than the 914 controller.

---

## Methodology notes

Two things mattered more than any single experiment.

**The simulator is not reproducible across runs.** The same policy on the same seed scored
692 / 684 / 956. `Environment._get_local_objects()` returns `set`s, and CPython hashes plain
objects by `id()`, so iteration order changes between runs — which changes observation order,
float summation order, `min()` tie-breaks and which fruit gets eaten first, and trajectories
diverge. Sorting those lookups by a stable key made a seed exactly reproducible (465.21 twice
in a row), which is what made paired A/B testing possible. Patch kept local only: the deployed
agent never runs the simulator, the organizers' server does.

**Small-sample comparisons are worthless here.** Per-episode scores span 284–1334 for the same
policy. Early in the competition I drew conclusions from 3–5 seeds and was wrong repeatedly —
a "956.8" candidate measured on 5 seeds turned out to be 783.3 on 20. `rl/ab.py` runs every
configuration on the *same* seeds and reports per-seed win rate alongside the mean, which is
the statistic that survives this much variance.

Three mechanism-derived hypotheses that looked right on paper were killed by that harness:
capping the population, camping next to trees, and raising the reproduction threshold all made
the score worse. Tree attraction in particular cost 10× (897 → 77, losing on 12/12 seeds): fruit
really does only spawn within 20–60 units of a tree, but an inverse-distance attraction pins the
agent to the trunk and it starves.

---

## Layout

```
agent_server.py                      FastAPI /predict endpoint (the submission)
rl/policies/potential_field.py       the controller that was submitted
rl/checkpoints/pf_params_best.json   its 19 tuned constants
rl/evolve_potential_field.py         (μ,λ) evolution strategy over those constants
rl/ab.py                             paired A/B harness
rl/eval.py  rl/diagnose.py           scoring and population/energy traces
rl/ppo.py  rl/train.py  rl/policy_net.py   PPO (the five attempts above)
rl/clone_*.py  rl/policies/mlp_policy.py   behavior cloning into a neural policy
rl/evolve_nn.py                      OpenAI-ES style neuroevolution
rl/visualize.py                      watch a policy play (pygame)
```

## Running it

The simulator itself belongs to the organizers and is not redistributed here. Clone their
repository and drop this code into the use-case folder:

```bash
git clone https://github.com/amboltio/Nordic-AI-Cup-2026.git
cd Nordic-AI-Cup-2026/survival-simulator
python -m venv .venv && source .venv/bin/activate   # .venv\Scripts\activate on Windows
pip install -r requirements.txt

# copy this repo's agent_server.py and rl/ in, then:
python -m rl.eval potential_field_tuned --seeds 20   # score the submitted agent
python -m rl.visualize --policy potential_field      # watch it play
python agent_server.py                               # serve /predict on :9052
```

Serving needs only `fastapi`, `pydantic` and `uvicorn` — the controller uses nothing beyond
`math` and `random`. Training additionally needs `torch`.

## Credits

Simulator, task design and evaluation service by [Ambolt AI](https://ambolt.io/) —
[amboltio/Nordic-AI-Cup-2026](https://github.com/amboltio/Nordic-AI-Cup-2026).
Everything in this repository is my own work for the competition.
