import os
import warnings

import numpy as np

from human_aware_rl.imitation.evaluate_dt import DTAgent

from human_aware_rl.imitation.behavior_cloning_tf2 import (
    BehaviorCloningPolicy,
    _get_base_ae,
    evaluate_bc_model,
    load_bc_model,
)
from human_aware_rl.rllib.rllib import (
    AgentPair,
    RlLibAgent,
    evaluate,
    get_agent_from_trainer,
    load_agent,
    load_agent_pair,
)
from overcooked_ai_py.agents.agent import ToMAugmentedAgent, GreedyHumanModel, HumanAgent
from overcooked_ai_py.mdp.overcooked_env import DEFAULT_ENV_PARAMS
from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.mdp.actions import Action
from human_aware_rl.imitation.transformer import DecisionTransformer
# Ignore all warnings
warnings.filterwarnings("ignore")

# the order of layouts we want to evaluate
layouts = [
    "cramped_room",
    "asymmetric_advantages",
    "coordination_ring",
    "forced_coordination",
    "counter_circuit_o_1order",
]

file_dir = os.path.dirname(os.path.abspath(__file__))
bc_path = os.path.join(file_dir, "../imitation/bc_runs")
proxy_path = os.path.join(file_dir, "../imitation/proxy")
# directories where the BC agents are stored
bc = [
    os.path.join(bc_path, "train/cramped_room"),
    os.path.join(bc_path, "train/asymmetric_advantages"),
    os.path.join(bc_path, "train/coordination_ring"),
    os.path.join(bc_path, "train/random0"),
    os.path.join(bc_path, "train/random3"),
]

bc_all = os.path.join(bc_path, "train/all")

# directories where the human proxy agents are stored
hp = [
    os.path.join(proxy_path, "train/cramped_room"),
    os.path.join(proxy_path, "train/asymmetric_advantages"),
    os.path.join(proxy_path, "train/coordination_ring"),
    os.path.join(proxy_path, "train/random0"),
    os.path.join(proxy_path, "train/random3"),
]

# reproduced agents ppo agents trained with bc, change the comments to the path of your trained agents
# change this to one of the agents creatd after running run_ppo_bc_experiments.sh bash script
ppo_bc = [
    # ppo_bc_crammed_room,
    # ppo_bc_asymmetric_advantages,
    # ppo_bc_coordination_ring,
    # ppo_bc_forced_coordination,
    # ppo_bc_counter_circuit_o_1order,
]
# reproduced agents ppo agents trained with self-play, change the comments to the path of your trained agents
# change this to one of the agents creatd after running run_experiments.sh bash script
ppo_sp = [
    "reproduced_results/ppo_sp_cramped_room/PPO_cramped_room_False_nw=10_vf=0.009950_es=0.200000_en=0.100000_kl=0.197000_20_2023-12-31_05-57-0687als7lk",
    "reproduced_results/ppo_sp_asymmetric_advantages/PPO_asymmetric_advantages_False_nw=10_vf=0.022000_es=0.200000_en=0.100000_kl=0.185000_40_2023-12-31_02-25-44_gmmjkdh",
    "reproduced_results/ppo_sp_coordination_ring/PPO_coordination_ring_False_nw=10_vf=0.009330_es=0.200000_en=0.100000_kl=0.156000_20_2023-12-31_00-26-11qjux6rq0",
    "reproduced_results/ppo_sp_forced_coordination/PPO_forced_coordination_False_nw=10_vf=0.016000_es=0.200000_en=0.100000_kl=0.310000_40_2023-12-31_04-19-31thcecq9u",
    "reproduced_results/ppo_sp_counter_circuit/PPO_counter_circuit_o_1order_False_nw=10_vf=0.009920_es=0.200000_en=0.100000_kl=0.299000_40_2023-12-31_07-47-451rax_8q8"
]

dt_tg = [
    120,
    200,
    100,
    120,
    100
]

DEFAULT_EVALUATION_PARAMS = {
    "ep_length": 400,
    "num_games": 1,
    "display": False,
}

DEFAULT_PARAMS = {
    "mdp_params": {"layout_name": "cramped_room", "old_dynamics": True},
    "env_params": DEFAULT_ENV_PARAMS,
    "mdp_fn_params": {},
    "evaluation_params": DEFAULT_EVALUATION_PARAMS,
    "action_shape": (len(Action.ALL_ACTIONS),),
}
# Customized evaluation functions

def evaluate_hp_bc(bc_model_path, hp_model_path, layout, order=0):
    """
    This function evaluates the performance between a BC model (trained with the human training data) and a human proxy model (trained with the human testing data)
    The order parameter determines the placement of the agents
    """
    bc_model, bc_params = load_bc_model(bc_model_path)
    bc_params["mdp_params"]["layout_name"] = layout
    bc_policy = BehaviorCloningPolicy.from_model(
        bc_model, bc_params, stochastic=True
    )

    hp_model, hp_params = load_bc_model(hp_model_path)
    hp_policy = BehaviorCloningPolicy.from_model(
        hp_model, hp_params, stochastic=True
    )
    print(bc_params)
    base_ae = _get_base_ae(bc_params)
    base_env = base_ae.env

    bc_agent = RlLibAgent(bc_policy, 0, base_env.featurize_state_mdp)
    hp_agent = RlLibAgent(hp_policy, 1, base_env.featurize_state_mdp)
    ae = AgentEvaluator.from_layout_name(
        mdp_params={"layout_name": layout, "old_dynamics": True},
        env_params={"horizon": 400},
    )
    if order == 0:
        ap = AgentPair(hp_agent, bc_agent)
    else:
        ap = AgentPair(bc_agent, hp_agent)
    result = ae.evaluate_agent_pair(ap, 5, 400)
    hp_actions = []
    predicted_actions = []
    if order == 0:
        for i in range(len(result["ep_actions"])):
            for action in result["ep_actions"][i]:
                hp_actions.append(Action.ACTION_TO_INDEX[action[0]])
        for i in range(len(result["ep_infos"])):
            for info in result["ep_infos"][i]:
                predicted_actions.append(info['agent_infos'][1]['action1_predict'])
    else:
        for i in range(len(result["ep_actions"])):
            for action in result["ep_actions"][i]:
                hp_actions.append(Action.ACTION_TO_INDEX[action[1]])
        for i in range(len(result["ep_infos"])):
            for info in result["ep_infos"][i]:
                predicted_actions.append(info['agent_infos'][0]['action1_predict'])
    # mask all hp_actions=4
    idx = np.where(np.array(hp_actions) != 4)[0]
    hp_actions = np.array(hp_actions)[idx]
    predicted_actions = np.array(predicted_actions)[idx]
    acc = 0
    # print(hp_actions, predicted_actions)
    for i in range(len(hp_actions)):
        if hp_actions[i] == predicted_actions[i]:
            acc += 1
    acc = acc / len(hp_actions)
    return result, result["ep_returns"], acc

def evaluate_ppo_bc(path, layout, order=0):
    """
    This function loads and evaluates a PPO agent and a BC agent that was trained together, thus stored in the same trainer
    Order determines the starting position of the agents
    """
    ae = AgentEvaluator.from_layout_name(
        {"layout_name": layout, "old_dynamics": True}, {"horizon": 400}
    )
    if order == 0:
        ap = load_agent_pair(path, "ppo", "bc")
    else:
        ap = load_agent_pair(path, "bc", "ppo")
    result = ae.evaluate_agent_pair(ap, 1, 400)
    return result, result["ep_returns"]


def evaluate_ppo(path, layout):
    """
    This function loads and evaluates the performance of 2 PPO self-play agents
    Order doesn't matter here since the agents are self-play
    """
    ae = AgentEvaluator.from_layout_name(
        {"layout_name": layout, "old_dynamics": True}, {"horizon": 400}
    )
    ap = load_agent_pair(path, "ppo", "ppo")
    result = ae.evaluate_agent_pair(ap, 1, 400)
    return result, result["ep_returns"]

def evaluate_human_ppo(trainer_path, layout):
    """
    This function loads and evaluates the performance of a human agent and a PPO agent
    Order determines the starting position of the agents
    """
    ae = AgentEvaluator.from_layout_name(
        {"layout_name": layout, "old_dynamics": True}, {"horizon": 400}
    )
    ppo_agent = load_agent(trainer_path, policy_id="ppo", agent_index=1)
    human_agent = HumanAgent()
    print("Now you will be playing the game as agent 0, which is the person in blue.")
    print("Press UP, DOWN, LEFT, RIGHT, ENTER and SPACE to move, stay and interact.")
    ap = AgentPair(human_agent, ppo_agent)
    result = ae.evaluate_agent_pair(ap, 1, 400, display=False)
    return result, result["ep_returns"]

def evaluate_hp_ppo(hp_model_path, trainer_path, layout, order=0):
    """
    This function evaluates the performance between a PPO agent and a human proxy model (trained with the human testing data)
    The order parameter determines the placement of the agents
    """
    hp_model, hp_params = load_bc_model(hp_model_path)
    hp_policy = BehaviorCloningPolicy.from_model(
        hp_model, hp_params, stochastic=True
    )
    # print(hp_params)
    base_ae = _get_base_ae(hp_params)
    base_env = base_ae.env
    hp_agent = RlLibAgent(hp_policy, order, base_env.featurize_state_mdp)
    ppo_agent = load_agent(trainer_path, policy_id="ppo", agent_index=1-order)

    ae = AgentEvaluator.from_layout_name(
        mdp_params={"layout_name": layout, "old_dynamics": True},
        env_params={"horizon": 400},
    )
    if order == 0:
        ap = AgentPair(hp_agent, ppo_agent)
    else:
        ap = AgentPair(ppo_agent, hp_agent)
    result = ae.evaluate_agent_pair(ap, 5, 400)
    return result, result["ep_returns"]

def evaluate_hp_dt(model, params, hp_model_path, layout, max_len=10, target_return=100, order=0):
    """
    This function evaluates the performance between a BC model (trained with the human training data) and a human proxy model (trained with the human testing data)
    The order parameter determines the placement of the agents
    """
    hp_model, hp_params = load_bc_model(hp_model_path)
    hp_policy = BehaviorCloningPolicy.from_model(
        hp_model, hp_params, stochastic=True
    )
    base_ae = _get_base_ae(params)
    base_env = base_ae.env
    def featurize_fn(state):
        return base_env.featurize_state_mdp(state)
    dt = DTAgent(
        model, agent_index=1-order, featurize_fn=featurize_fn,max_len=max_len,target_return=target_return
    )
    hp_agent = RlLibAgent(hp_policy, order, base_env.featurize_state_mdp)
    ae = AgentEvaluator.from_layout_name(
        mdp_params={"layout_name": layout, "old_dynamics": True},
        env_params={"horizon": 400},
    )
    if order == 0:
        ap = AgentPair(hp_agent, dt, dt=2)
    else:
        ap = AgentPair(dt, hp_agent, dt=1)
    result = ae.evaluate_agent_pair(ap, 5, 400)
    hp_actions = []
    predicted_actions = []
    for i in range(len(result["ep_actions"])):
        for action in result["ep_actions"][i]:
            hp_actions.append(Action.ACTION_TO_INDEX[action[order]])
    for i in range(len(result["ep_infos"])):
        for info in result["ep_infos"][i]:
            predicted_actions.append(info['agent_infos'][1-order]['action1_predict'])
    idx = np.where(np.array(hp_actions) != 4)[0]
    hp_actions = np.array(hp_actions)[idx]
    predicted_actions = np.array(predicted_actions)[idx]
    acc = 0
    for i in range(len(hp_actions)):
        if hp_actions[i] == predicted_actions[i]:
            acc += 1
    acc = acc / len(hp_actions)
    return result, result["ep_returns"], acc

def evaluate_hp_tom(model, params, hp_model_path, trainer_path, layout, max_len=10, target_return=100, beta=2, order=0):
    base_ae = _get_base_ae(params)
    ppo_agent = load_agent(trainer_path, policy_id="ppo", agent_index=1)
    base_env = base_ae.env
    def featurize_fn(state):
        return base_env.featurize_state_mdp(state)
    tom = ToMAugmentedAgent(
        model, ppo_agent, agent_index=1-order, featurize_fn=featurize_fn, max_len=max_len, target_return=target_return,beta=beta
    )
    hp_model, hp_params = load_bc_model(hp_model_path)
    hp_policy = BehaviorCloningPolicy.from_model(
        hp_model, hp_params, stochastic=True
    )
    hp_agent = RlLibAgent(hp_policy, order, base_env.featurize_state_mdp)
    ae = AgentEvaluator.from_layout_name(
        mdp_params={"layout_name": layout, "old_dynamics": True},
        env_params={"horizon": 400},
    )
    if order == 0:
        ap = AgentPair(hp_agent,tom, dt=2)
    else:
        ap = AgentPair(tom, hp_agent, dt=1)
    result = ae.evaluate_agent_pair(ap, 5, 400)
    return result, result["ep_returns"]


def evaluate_dt_ppo(model, params, trainer_path, layout, order = 0, max_len = 10, target_return=100):
    base_ae = _get_base_ae(params)
    ppo_agent = load_agent(trainer_path, policy_id="ppo", agent_index=1)
    base_env = base_ae.env
    def featurize_fn(state):
        return base_env.featurize_state_mdp(state)
    dt = DTAgent(
        model, agent_index=1-order, featurize_fn=featurize_fn,max_len=max_len,target_return=target_return
    )
    ae = AgentEvaluator.from_layout_name(
        mdp_params={"layout_name": layout, "old_dynamics": True},
        env_params={"horizon": 400},
    )
    if order == 0:
        ap = AgentPair(ppo_agent ,dt) 
    else:
        ap = AgentPair(dt, ppo_agent)
    result = ae.evaluate_agent_pair(ap, 10, 400)
    hp_actions = []
    predicted_actions = []
    if order == 0:
        for i in range(len(result["ep_actions"])):
            for action in result["ep_actions"][i]:
                hp_actions.append(Action.ACTION_TO_INDEX[action[0]])
        for i in range(len(result["ep_infos"])):
            for info in result["ep_infos"][i]:
                predicted_actions.append(info['agent_infos'][1]['action1_predict'])
    else:
        for i in range(len(result["ep_actions"])):
            for action in result["ep_actions"][i]:
                hp_actions.append(Action.ACTION_TO_INDEX[action[1]])
        for i in range(len(result["ep_infos"])):
            for info in result["ep_infos"][i]:
                predicted_actions.append(info['agent_infos'][0]['action1_predict'])
    acc = 0
    for i in range(len(hp_actions)):
        if hp_actions[i] == predicted_actions[i]:
            acc += 1
    acc = acc / len(hp_actions)
    return result, result["ep_returns"], acc

def evaluate_human_dt(model, params, layout, max_len = 10, target_return=100):
    base_ae = _get_base_ae(params)
    human_agent = HumanAgent()
    print("Now you will be playing the game as agent 0, which is the person in blue.")
    print("Press UP, DOWN, LEFT, RIGHT, ENTER and SPACE to move, stay and interact.")
    base_env = base_ae.env
    def featurize_fn(state):
        return base_env.featurize_state_mdp(state)
    dt = DTAgent(
        model, agent_index=0, featurize_fn=featurize_fn,max_len=max_len,target_return=target_return
    )
    ae = AgentEvaluator.from_layout_name(
        mdp_params={"layout_name": layout, "old_dynamics": True},
        env_params={"horizon": 400},
    )
    ap = AgentPair(human_agent, dt, dt=2)
    result = ae.evaluate_agent_pair(ap, 1, 400)
    return result, result["ep_returns"]

def evaluate_dt_tom(model, params, trainer_path, layout, max_len=10, target_return=100, beta=2):
    base_ae = _get_base_ae(params)
    ppo_agent = load_agent(trainer_path, policy_id="ppo", agent_index=1)
    base_env = base_ae.env
    def featurize_fn(state):
        return base_env.featurize_state_mdp(state)
    tom = ToMAugmentedAgent(
        model, ppo_agent, agent_index=1, featurize_fn=featurize_fn, max_len=max_len, target_return=target_return,beta=beta
    )
    dt = DTAgent(
        model, agent_index=0, featurize_fn=featurize_fn,max_len=max_len, target_return=target_return
    )
    ae = AgentEvaluator.from_layout_name(
        mdp_params={"layout_name": layout, "old_dynamics": True},
        env_params={"horizon": 400},
    )
    ap = AgentPair(dt, tom, dt=3)
    result = ae.evaluate_agent_pair(ap, 10, 400)
    return result, result["ep_returns"]

def evaluate_human_tom(model, params, trainer_path, layout, max_len=10, target_return=100, beta=2):
    base_ae = _get_base_ae(params)
    ppo_agent = load_agent(trainer_path, policy_id="ppo", agent_index=1)
    base_env = base_ae.env
    def featurize_fn(state):
        return base_env.featurize_state_mdp(state)
    print("Now you will be playing the game as agent 0, which is the person in blue.")
    print("Press UP, DOWN, LEFT, RIGHT, ENTER and SPACE to move, stay and interact.")
    tom = ToMAugmentedAgent(
        model, ppo_agent, agent_index=1, featurize_fn=featurize_fn, max_len=max_len, target_return=target_return,beta=beta
    )
    human_agent = HumanAgent()
    ae = AgentEvaluator.from_layout_name(
        mdp_params={"layout_name": layout, "old_dynamics": True},
        env_params={"horizon": 400},
    )
    ap = AgentPair(human_agent, tom, dt=2)
    result = ae.evaluate_agent_pair(ap, 1, 400)
    return result, result["ep_returns"]


def evaluate_dt_greedy(model, params, layout, max_len=10, target_return=100, beta=2):
    base_ae = _get_base_ae(params)
    base_env = base_ae.env
    def featurize_fn(state):
        return base_env.featurize_state_mdp(state)
    greedy = GreedyHumanModel(base_env.mlam)
    dt = DTAgent(
        model, agent_index=0, featurize_fn=featurize_fn,max_len=max_len, target_return=target_return
    )
    ae = AgentEvaluator.from_layout_name(
        mdp_params={"layout_name": layout, "old_dynamics": True},
        env_params={"horizon": 400},
    )
    ap = AgentPair(dt, greedy, dt=1)
    result = ae.evaluate_agent_pair(ap, 5, 400)
    return result, result["ep_returns"]

def evaluate_ppo_greedy(params, trainer_path, layout):
    base_ae = _get_base_ae(params)
    base_env = base_ae.env
    ppo_agent = load_agent(trainer_path, policy_id="ppo", agent_index=1)
    def featurize_fn(state):
        return base_env.featurize_state_mdp(state)
    greedy = GreedyHumanModel(base_env.mlam)
    ae = AgentEvaluator.from_layout_name(
        mdp_params={"layout_name": layout, "old_dynamics": True},
        env_params={"horizon": 400},
    )
    ap = AgentPair(greedy, ppo_agent, dt=0)
    result = ae.evaluate_agent_pair(ap, 5, 400)
    return result, result["ep_returns"]

def evaluate_greedy_tom(model, params, trainer_path, layout, max_len=10, target_return=100, beta=2):
    base_ae = _get_base_ae(params)
    ppo_agent = load_agent(trainer_path, policy_id="ppo", agent_index=1)
    base_env = base_ae.env
    def featurize_fn(state):
        return base_env.featurize_state_mdp(state)
    tom = ToMAugmentedAgent(
        model, ppo_agent, agent_index=1, featurize_fn=featurize_fn, max_len=max_len, target_return=target_return,beta=beta
    )
    greedy = GreedyHumanModel(base_env.mlam)
    ae = AgentEvaluator.from_layout_name(
        mdp_params={"layout_name": layout, "old_dynamics": True},
        env_params={"horizon": 400},
    )
    ap = AgentPair(greedy, tom, dt=2)
    result = ae.evaluate_agent_pair(ap, 5, 400)
    return result, result["ep_returns"]


def eval_models(order):
    hp_PBC = {}
    hp_PSP = {}
    bc_PBC = {}
    PSP_PSP = {}
    hp_BC = {}

    for i in range(5):
        # hp vs ppo_bc
        _, res = evaluate_hp_ppo(hp[i], ppo_bc[i], layouts[i], order)
        hp_PBC[layouts[i]] = (np.mean(res), np.std(res) / len(res) ** 0.5)
        # hp vs ppo_sp
        _, res = evaluate_hp_ppo(hp[i], ppo_sp[i], layouts[i], order)
        hp_PSP[layouts[i]] = (np.mean(res), np.std(res) / len(res) ** 0.5)
        # bc vs ppo_bc
        _, res = evaluate_ppo_bc(ppo_bc[i], layouts[i], order)
        bc_PBC[layouts[i]] = (np.mean(res), np.std(res) / len(res) ** 0.5)
        # ppo_sp vs ppo_sp
        _, res = evaluate_ppo(ppo_sp[i], layouts[i])
        PSP_PSP[layouts[i]] = (np.mean(res), np.std(res) / len(res) ** 0.5)
        # bc vs hp
        _, res = evaluate_hp_bc(bc[i], hp[i], layouts[i], order)
        hp_BC[layouts[i]] = (np.mean(res), np.std(res) / len(res) ** 0.5)
    return PSP_PSP, hp_PSP, hp_PBC, hp_BC, bc_PBC

def dt_ppo_experiment():
    result = [{},{}]
    model = DecisionTransformer(state_dim=96, act_dim=6, 
                                hidden_size=128, max_length=10,
                                max_ep_len=1250,
                                n_layer=3, n_head=1, n_inner=4*256, activation_function='relu',
                                resid_pdrop=0.1, attn_pdrop=0.1, n_positions=1024)
    for order in [0,1]:
        for i in range(5):
            DEFAULT_PARAMS["mdp_params"]["layout_name"] = layouts[i]
            _, res, acc = evaluate_dt_ppo(model, DEFAULT_PARAMS, ppo_sp[i], layouts[i], target_return=dt_tg[i], order=order)
            print(acc)
            result[order][layouts[i]] = (np.mean(res), np.std(res) / len(res) ** 0.5, acc)
    
    print(result)
    # {'cramped_room': (104.0, 7.899367063252599), 'asymmetric_advantages': (116.0, 16.685322891691367), 'coordination_ring': (98.0, 6.603029607687671), 'forced_coordination': (20.0, 4.898979485566356), 'counter_circuit_o_1order': (26.0, 4.049691346263317)}

def dt_tom_experiment(beta=2):
    result = {}
    model = DecisionTransformer(state_dim=96, act_dim=6, 
                                hidden_size=128, max_length=10,
                                max_ep_len=1250,
                                n_layer=3, n_head=1, n_inner=4*256, activation_function='relu',
                                resid_pdrop=0.1, attn_pdrop=0.1, n_positions=1024)
    for i in range(5):
        DEFAULT_PARAMS["mdp_params"]["layout_name"] = layouts[i]
        _, res = evaluate_dt_tom(model, DEFAULT_PARAMS, ppo_sp[i], layouts[i], target_return=dt_tg[i], beta=beta)
        result[layouts[i]] = (np.mean(res), np.std(res) / len(res) ** 0.5)
    
    print(result)

# for beta in [1, 2, 5, 10]:
#     print(beta)
#     dt_tom_experiment(beta)


def ppo_greedy_experiment():
    result = {}
    for i in range(5):
        DEFAULT_PARAMS["mdp_params"]["layout_name"] = layouts[i]
        _, res = evaluate_ppo_greedy(DEFAULT_PARAMS, ppo_sp[i], layouts[i])
        result[layouts[i]] = (np.mean(res), np.std(res) / len(res) ** 0.5)
    print(result)
    # {'cramped_room': (212.0, 4.381780460041329), 'asymmetric_advantages': (128.0, 36.48561360317241), 'coordination_ring': (188.0, 7.155417527999327), 'forced_coordination': (0.0, 0.0), 'counter_circuit_o_1order': (84.0, 10.430723848324238)}

def dt_greedy_experiment():
    result = {}
    model = DecisionTransformer(state_dim=96, act_dim=6, 
                                hidden_size=128, max_length=10,
                                max_ep_len=1250,
                                n_layer=3, n_head=1, n_inner=4*256, activation_function='relu',
                                resid_pdrop=0.1, attn_pdrop=0.1, n_positions=1024)
    for i in range(5):
        DEFAULT_PARAMS["mdp_params"]["layout_name"] = layouts[i]
        _, res = evaluate_dt_greedy(model, DEFAULT_PARAMS, layouts[i], target_return=dt_tg[i])
        result[layouts[i]] = (np.mean(res), np.std(res) / len(res) ** 0.5)
    
    print(result)
    # {'cramped_room': (108.0, 21.614809737770074), 'asymmetric_advantages': (220.0, 9.797958971132712), 'coordination_ring': (132.0, 4.381780460041329), 'forced_coordination': (0.0, 0.0), 'counter_circuit_o_1order': (80.0, 8.0)

def tom_greedy_experiment():
    result = {}
    model = DecisionTransformer(state_dim=96, act_dim=6, 
                                hidden_size=128, max_length=10,
                                max_ep_len=1250,
                                n_layer=3, n_head=1, n_inner=4*256, activation_function='relu',
                                resid_pdrop=0.1, attn_pdrop=0.1, n_positions=1024)
    for i in range(5):
        DEFAULT_PARAMS["mdp_params"]["layout_name"] = layouts[i]
        _, res = evaluate_greedy_tom(model, DEFAULT_PARAMS, ppo_sp[i], layouts[i], target_return=dt_tg[i], beta=10)
        result[layouts[i]] = (np.mean(res), np.std(res) / len(res) ** 0.5)
    
    print(result)
    # {'cramped_room': (224.0, 3.5777087639996634), 'asymmetric_advantages': (108.0, 30.252272641902458), 'coordination_ring': (192.0, 12.13260071048248), 'forced_coordination': (0.0, 0.0), 'counter_circuit_o_1order': (36.0, 21.46625258399798)}

def human_ppo_experiment(x):
    result = {}
    for i in range(1):
        DEFAULT_PARAMS["mdp_params"]["layout_name"] = layouts[x]
        _, res = evaluate_human_ppo(ppo_sp[x], layouts[x])
        result[layouts[x]] = (np.mean(res), np.std(res) / len(res) ** 0.5)
    print(result)


def human_dt_experiment(x):
    result = {}
    model = DecisionTransformer(state_dim=96, act_dim=6, 
                                hidden_size=128, max_length=10,
                                max_ep_len=1250,
                                n_layer=3, n_head=1, n_inner=4*256, activation_function='relu',
                                resid_pdrop=0.1, attn_pdrop=0.1, n_positions=1024)
    for i in range(1):
        DEFAULT_PARAMS["mdp_params"]["layout_name"] = layouts[x]
        _, res = evaluate_human_dt(model, DEFAULT_PARAMS, layouts[x], target_return=dt_tg[x])
        result[layouts[x]] = (np.mean(res), np.std(res) / len(res) ** 0.5)
    
    print(result)

def human_tom_experiment(x):
    result = {}
    model = DecisionTransformer(state_dim=96, act_dim=6, 
                                hidden_size=128, max_length=10,
                                max_ep_len=1250,
                                n_layer=3, n_head=1, n_inner=4*256, activation_function='relu',
                                resid_pdrop=0.1, attn_pdrop=0.1, n_positions=1024)
    for i in range(1):
        DEFAULT_PARAMS["mdp_params"]["layout_name"] = layouts[x]
        _, res = evaluate_human_tom(model, DEFAULT_PARAMS, ppo_sp[x], layouts[x], target_return=dt_tg[x], beta=1)
        result[layouts[x]] = (np.mean(res), np.std(res) / len(res) ** 0.5)
    
    print(result)