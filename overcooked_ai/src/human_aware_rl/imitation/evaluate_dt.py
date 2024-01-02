from human_aware_rl.rllib.rllib import (
    evaluate,
    get_base_ae
)
from overcooked_ai_py.agents.agent import Agent, AgentPair
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_env import DEFAULT_ENV_PARAMS
from human_aware_rl.imitation.transformer import DecisionTransformer
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer
import torch
import numpy as np
DEFAULT_EVALUATION_PARAMS = {
    "ep_length": 400,
    "num_games": 1,
    "display": False,
}

DEFAULT_BC_PARAMS = {
    "mdp_params": {"layout_name": "cramped_room", "old_dynamics": True},
    "env_params": DEFAULT_ENV_PARAMS,
    "mdp_fn_params": {},
    "evaluation_params": DEFAULT_EVALUATION_PARAMS,
    "action_shape": (len(Action.ALL_ACTIONS),),
}

class DTAgent(Agent):
    def __init__(self, model, agent_index, featurize_fn, max_len=10, target_return=100):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device)
        self.agent_index = agent_index
        self.featurize_fn = featurize_fn
        self.target_return = target_return
        self.history = {
            "ep_states": [],
            "ep_action0s": [],
            "ep_action1s": [],
            "ep_rtgs": [self.target_return],
            "ep_timesteps": [],
        }
        self.time = 1
        self.max_len = max_len
        self.load()
    
    def set_agent_index(self, agent_index):
        self.agent_index = agent_index
    
    def load(self):
        path = f"/Users/apple/Desktop/AI/CoRe/CoRe/overcook/model/dt_{self.max_len}.pth"
        self.model.load_state_dict(torch.load(path, map_location=self.device))

    def reset(self):
        self.history = {
            "ep_states": [],
            "ep_action0s": [],
            "ep_action1s": [],
            "ep_rtgs": [self.target_return],
            "ep_timesteps": [],
        }
        self.time = 1

    def action_probabilities(self, state):
        state = self.featurize_fn(state)[self.agent_index]
        self.history["ep_states"].append(state)
        self.history["ep_action0s"].append(6)
        self.history["ep_action1s"].append(6)
        self.history["ep_timesteps"].append(self.time)
        self.time += 1
        action0_preds, action1_preds, rtg_preds = self.model.get_action(
            self.history["ep_states"],
            self.history["ep_action0s"],
            self.history["ep_action1s"],
            self.history["ep_rtgs"],
            self.history["ep_timesteps"],
        )
        # print(rtg_preds)
        return torch.softmax(action0_preds, dim=0).cpu().detach().numpy()
    
    def action(self, state):
        action_probs = self.action_probabilities(state)
        # sample action from action_probs
        action = np.random.choice(
            np.array(Action.ALL_ACTIONS, dtype=object), p=action_probs
        )

        # choose argmax action
        # action = Action.ALL_ACTIONS[np.argmax(action_probs)]
        # print(action)

        agent_action_info = {"action_probs": action_probs}
        return action, agent_action_info

    def update(self, self_action, teammate_action, reward):
        self.history["ep_action0s"][-1] = Action.ACTION_TO_INDEX[self_action]
        self.history["ep_action1s"][-1] = Action.ACTION_TO_INDEX[teammate_action]
        self.history["ep_rtgs"].append(self.history["ep_rtgs"][-1] - reward)
        # print(f"History RTGS: {self.history['ep_rtgs'][-1]}")
        # print(f"Agent {self.agent_index}: {self.history}")
        # exit()



def _get_base_ae(bc_params):
    return get_base_ae(bc_params["mdp_params"], bc_params["env_params"])

def evaluate(
    eval_params,
    mdp_params,
    outer_shape,
    model,
    featurize_fn,
    verbose=False,
    max_len=10,
):
    """
    Used to visualize rollouts of trained policies

    eval_params (dict): Contains configurations such as the rollout length, number of games, and whether to display rollouts
    mdp_params (dict): OvercookedMDP compatible configuration used to create environment used for evaluation
    outer_shape (list): a list of 2 item specifying the outer shape of the evaluation layout
    agent_0_policy (rllib.Policy): Policy instance used to map states to action logits for agent 0
    agent_1_policy (rllib.Policy): Policy instance used to map states to action logits for agent 1
    agent_0_featurize_fn (func): Used to preprocess states for agent 0, defaults to lossless_state_encoding if 'None'
    agent_1_featurize_fn (func): Used to preprocess states for agent 1, defaults to lossless_state_encoding if 'None'
    """
    if verbose:
        print("eval mdp params", mdp_params)
    evaluator = get_base_ae(
        mdp_params,
        {"horizon": eval_params["ep_length"], "num_mdp": 1},
        outer_shape,
    )

    # Wrap rllib policies in overcooked agents to be compatible with Evaluator code
    agent0 = DTAgent(
        model, agent_index=0, featurize_fn=featurize_fn,max_len=max_len
    )
    agent1 = DTAgent(
        model, agent_index=1, featurize_fn=featurize_fn,max_len=max_len
    )

    # Compute rollouts
    if "store_dir" not in eval_params:
        eval_params["store_dir"] = None
    if "display_phi" not in eval_params:
        eval_params["display_phi"] = False
    results = evaluator.evaluate_agent_pair(
        AgentPair(agent0, agent1, dt=3),
        num_games=eval_params["num_games"],
        display=eval_params["display"],
        dir=eval_params["store_dir"],
        display_phi=eval_params["display_phi"],
        info=verbose,
    )
    return results

def evaluate_bc_model(model, bc_params, verbose=False, max_len=10):
    evaluation_params = bc_params["evaluation_params"]
    mdp_params = bc_params["mdp_params"]
    # Get reference to state encoding function used by bc agents, with compatible signature
    base_ae = _get_base_ae(bc_params)
    base_env = base_ae.env
    def featurize_fn(state):
        return base_env.featurize_state_mdp(state)

    # Compute the results of the rollout(s)
    results = evaluate(
        eval_params=evaluation_params,
        mdp_params=mdp_params,
        outer_shape=None,
        model=model,
        verbose=verbose,
        featurize_fn=featurize_fn,
        max_len=max_len,
    )
    print(results)
    StateVisualizer().display_rendered_trajectory(results, ipython_display=False)

if __name__=="__main__":
    state_dim = 96
    action_dim = 6
    # model = DecisionTransformer(state_dim=state_dim, act_dim=action_dim, 
                                # hidden_size=256, max_length=10,
                                # max_ep_len=1250,
                                # n_layer=4, n_head=2, n_inner=4*256, activation_function='relu',
                                # resid_pdrop=0.1, attn_pdrop=0.1, n_positions=1024)
    model = DecisionTransformer(state_dim=state_dim, act_dim=action_dim, 
                                hidden_size=128, max_length=10,
                                max_ep_len=1250,
                                n_layer=3, n_head=1, n_inner=4*256, activation_function='relu',
                                resid_pdrop=0.1, attn_pdrop=0.1, n_positions=1024)
    # for i in range(5):
    evaluate_bc_model(model, DEFAULT_BC_PARAMS, verbose=True, max_len=10)