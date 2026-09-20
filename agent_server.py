import random
from fastapi import FastAPI, Body
from src.utils.DTOs import StepResponse, ActionRequest
from rl.policies.potential_field_tuned import make_tuned_policy

HOST = "0.0.0.0"
PORT = 9052

app = FastAPI(title="Survival Simulator Agent Endpoint")

policy_fn = make_tuned_policy()  # loads rl/checkpoints/pf_params_best.json once at startup
rng = random.Random(1)

@app.post("/predict")
def predict(step: StepResponse = Body(...)):
    """
    Receives the current simulation state and returns actions for all agents.
    """
    actions = []
    for agent in step.agent_status:
        agent_status = agent.dict()
        # pass the global context down to the policy (same fields the training
        # harness injects, so deployed behavior matches what we evaluated)
        agent_status["sim_time"] = step.sim_time
        agent_status["n_agents"] = step.n_agents
        raw = policy_fn(agent_status, rng)
        actions.append(ActionRequest(agent_id=agent.agent_id, **raw).dict())

    # Must return {"actions": [...]} format
    return {"actions": actions}

@app.get("/")
def index():
    return {"message": "Agent endpoint running!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)