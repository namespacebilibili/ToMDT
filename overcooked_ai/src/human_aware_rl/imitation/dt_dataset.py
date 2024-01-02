import copy
import os
import pickle

import numpy as np

from human_aware_rl.data_dir import DATA_DIR
from human_aware_rl.human.process_dataframes import (
    get_human_human_trajectories,
    get_trajs_from_data,
)
from human_aware_rl.rllib.rllib import get_base_ae
from human_aware_rl.static import *
from human_aware_rl.utils import get_flattened_keys, recursive_dict_update
from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.mdp.overcooked_mdp import (
    ObjectState,
    OvercookedGridworld,
    OvercookedState,
    PlayerState,
)
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_env import DEFAULT_ENV_PARAMS
import pandas as pd

from collections import defaultdict
from typing import DefaultDict
import json
import random
#################
# Configuration #
#################

# min_length = 1e5

def _get_data_path(layout, dataset_type, data_path):
    if data_path:
        return data_path
    if dataset_type == "train":
        return (
            CLEAN_2019_HUMAN_DATA_TRAIN
            if layout in LAYOUTS_WITH_DATA_2019
            else CLEAN_2020_HUMAN_DATA_TRAIN
        )
    if dataset_type == "test":
        return (
            CLEAN_2019_HUMAN_DATA_TEST
            if layout in LAYOUTS_WITH_DATA_2019
            else CLEAN_2020_HUMAN_DATA_TEST
        )
    if dataset_type == "all":
        return (
            CLEAN_2019_HUMAN_DATA_ALL
            if layout in LAYOUTS_WITH_DATA_2019
            else CLEAN_2020_HUMAN_DATA_ALL
        )
    
BC_SAVE_DIR = os.path.join(DATA_DIR, "dt_runs")

DEFAULT_DATA_PARAMS = {
    "layouts": ["cramped_room"],
    "check_trajectories": False,
    "featurize_states": True,
    "data_path": CLEAN_2019_HUMAN_DATA_TRAIN,
}

DEFAULT_MLP_PARAMS = {
    # Number of fully connected layers to use in our network
    "num_layers": 2,
    # Each int represents a layer of that hidden size
    "net_arch": [64, 64],
}

DEFAULT_TRAINING_PARAMS = {
    "epochs": 30,
    "validation_split": 0.15,
    "batch_size": 64,
    "learning_rate": 1e-3,
    "use_class_weights": False,
}

DEFAULT_EVALUATION_PARAMS = {
    "ep_length": 400,
    "num_games": 1,
    "display": False,
}

DEFAULT_BC_PARAMS = {
    "eager": True,
    "use_lstm": False,
    "cell_size": 256,
    "data_params": DEFAULT_DATA_PARAMS,
    "mdp_params": {"layout_name": "cramped_room", "old_dynamics": False},
    "env_params": DEFAULT_ENV_PARAMS,
    "mdp_fn_params": {},
    "mlp_params": DEFAULT_MLP_PARAMS,
    "training_params": DEFAULT_TRAINING_PARAMS,
    "evaluation_params": DEFAULT_EVALUATION_PARAMS,
    "action_shape": (len(Action.ALL_ACTIONS),),
}

# Boolean indicating whether all param dependencies have been loaded. Used to prevent re-loading unceccesarily
_params_initalized = False

def json_action_to_python_action(action):
    if type(action) is list:
        action = tuple(action)
    if type(action) is str:
        action = action.lower()
    assert action in Action.ALL_ACTIONS
    return action

def json_joint_action_to_python_action(json_joint_action):
    """Port format from javascript to python version of Overcooked"""
    if type(json_joint_action) is str:
        try:
            json_joint_action = json.loads(json_joint_action)
        except json.decoder.JSONDecodeError:
            # hacky fix to circumvent 'INTERACT' action being malformed json (because of single quotes)
            # Might need to find a more robust way around this in the future
            json_joint_action = eval(json_joint_action)
    return tuple(json_action_to_python_action(a) for a in json_joint_action)


def json_state_to_python_state(df_state):
    """Convert from a df cell format of a state to an Overcooked State"""
    if type(df_state) is str:
        df_state = json.loads(df_state)

    return OvercookedState.from_dict(df_state)

def _get_base_ae(bc_params):
    return get_base_ae(bc_params["mdp_params"], bc_params["env_params"])


def _get_observation_shape(bc_params):
    """
    Helper function for creating a dummy environment from "mdp_params" and "env_params" specified
    in bc_params and returning the shape of the observation space
    """
    base_ae = _get_base_ae(bc_params)
    base_env = base_ae.env
    dummy_state = base_env.mdp.get_standard_start_state()
    obs_shape = base_env.featurize_state_mdp(dummy_state)[0].shape
    return obs_shape


# For lazily loading the default params. Prevents loading on every import of this module
def get_bc_params(**args_to_override):
    """
    Loads default bc params defined globally. For each key in args_to_override, overrides the default with the
    value specified for that key. Recursively checks all children. If key not found, creates new top level parameter.

    Note: Even though children can share keys, for simplicity, we enforce the condition that all keys at all levels must be distict
    """
    global _params_initalized, DEFAULT_BC_PARAMS
    if not _params_initalized:
        DEFAULT_BC_PARAMS["observation_shape"] = _get_observation_shape(
            DEFAULT_BC_PARAMS
        )
        _params_initalized = False
    params = copy.deepcopy(DEFAULT_BC_PARAMS)

    for arg, val in args_to_override.items():
        updated = recursive_dict_update(params, arg, val)
        if not updated:
            print(
                "WARNING, no value for specified bc argument {} found in schema. Adding as top level parameter".format(
                    arg
                )
            )

    all_keys = get_flattened_keys(params)
    if len(all_keys) != len(set(all_keys)):
        raise ValueError(
            "Every key at every level must be distict for BC params!"
        )

    return params
    
def get_trajectories(
    layouts, dataset_type="train", data_path=None, **kwargs
):
    data = []
    # Determine which paths are needed for which layouts (according to hierarchical path resolution rules outlined in docstring)
    data_path_to_layouts = DefaultDict(list)
    for layout in layouts:
        curr_data_path = _get_data_path(layout, dataset_type, data_path)
        # print(curr_data_path)
        data_path_to_layouts[curr_data_path].append(layout)

    # For each data path, load data once and parse trajectories for all corresponding layouts
    for data_path in data_path_to_layouts:
        curr_data = get_trajs_from_data(
            data_path, layouts=data_path_to_layouts[data_path], **kwargs
        )
        # concate data
        data = data + curr_data 
    return data

def get_trajs_from_data(data_path, layouts, silent=False, **kwargs):
    if not silent:
        print("Loading data from {}".format(data_path))
    main_trials = pd.read_pickle(data_path)
    trajs = convert_joint_df_trajs_to_overcooked_single(
        main_trials, layouts, silent=silent, **kwargs
    )
    return trajs

def convert_joint_df_trajs_to_overcooked_single(
    main_trials, layouts, silent=False, **kwargs  
):
    trajectories = []
    num_trials_for_layout = {}
    max_returns = {}
    average_returns = {}
    for layout_name in layouts:
        trial_ids = np.unique(
            main_trials[main_trials["layout_name"] == layout_name]["trial_id"]
        )
        num_trials = len(trial_ids)
        num_trials_for_layout[layout_name] = num_trials
        if num_trials == 0:
            print(
                "WARNING: No trajectories found on {} layout!".format(
                    layout_name
                )
            )

        for trial_id in trial_ids:
            traj_df = main_trials[main_trials["trial_id"] == trial_id]
            traj0, traj1 = df_traj_to_python_traj(traj_df)
            trajectories.append(traj0)
            trajectories.append(traj1)
            max_returns[layout_name] = max(max_returns.get(layout_name, 0), traj0["ep_returns"][0])
            average_returns[layout_name] = average_returns.get(layout_name,0) + traj0["ep_returns"][0]
            max_returns[layout_name] = max(max_returns.get(layout_name, 0), traj1["ep_returns"][0])
            average_returns[layout_name] = average_returns.get(layout_name,0) + traj1["ep_returns"][0]
        average_returns[layout_name] = average_returns[layout_name] / (2 * num_trials)
    print(max_returns)
    print(average_returns)
    return trajectories
            
def df_traj_to_python_traj(
        traj_df, check_trajectories=True, silent=True, **kwargs
):
    global min_length
    if len(traj_df) == 0:
        return None
    datapoint = traj_df.iloc[0]
    layout_name = datapoint["layout_name"]
    agent_evaluator = AgentEvaluator.from_layout_name(
        mdp_params={"layout_name": layout_name},
        env_params={
            "horizon": 1250
        },  # Defining the horizon of the mdp of origin of the trajectories
    )
    mdp = agent_evaluator.env.mdp
    env = agent_evaluator.env

    overcooked_states = [json_state_to_python_state(s) for s in traj_df.state]
    overcooked_states_0 = []
    overcooked_states_1 = []

    for i in range(len(overcooked_states)):
        overcooked_states_0.append(env.featurize_state_mdp(overcooked_states[i])[0]) # from the perspective of player 0
        overcooked_states_1.append(env.featurize_state_mdp(overcooked_states[i])[1]) # from the perspective of player 1

    # min_length = min(len(overcooked_states), min_length)
    overcooked_actions_0 = [
        json_joint_action_to_python_action(joint_action)[0]
        for joint_action in traj_df.joint_action
    ]
    overcooked_actions_1 = [
        json_joint_action_to_python_action(joint_action)[1]
        for joint_action in traj_df.joint_action
    ]
    for i in range(len(overcooked_actions_0)):
        overcooked_actions_0[i] = np.array([Action.ACTION_TO_INDEX[overcooked_actions_0[i]]]).astype(int)
        overcooked_actions_1[i] = np.array([Action.ACTION_TO_INDEX[overcooked_actions_1[i]]]).astype(int)

    overcooked_rewards = list(traj_df.reward)
    # print(f"States: {len(overcooked_states)}")
    # print(f"Actions: {len(overcooked_actions)}")
    # print(f"Rewards: {len(overcooked_rewards)}")
    # print(overcooked_actions[100], overcooked_rewards[10], overcooked_states[100])
    assert (
        sum(overcooked_rewards) == datapoint.score_total
    ), "Rewards didn't sum up to cumulative rewards. Probably trajectory df is corrupted / not complete"

    trajectories_0 = {
        "ep_states": overcooked_states_0,
        "ep_actions_0": overcooked_actions_0,
        "ep_actions_1": overcooked_actions_1,
        "ep_rewards": overcooked_rewards,  # Individual (dense) reward values
        "ep_dones": [False] * len(overcooked_states),  # Individual done values
        "ep_infos": [{}] * len(overcooked_states),
        "ep_returns": [
            sum(overcooked_rewards)
        ],  # Sum of dense rewards across each episode
        "ep_lengths": [len(overcooked_states)],  # Lengths of each episode
    }
    trajectories_1 = {
        "ep_states": overcooked_states_1,
        "ep_actions_0": overcooked_actions_1,
        "ep_actions_1": overcooked_actions_0,
        "ep_rewards": overcooked_rewards,  # Individual (dense) reward values
        "ep_dones": [False] * len(overcooked_states),  # Individual done values
        "ep_infos": [{}] * len(overcooked_states),
        "ep_returns": [
            sum(overcooked_rewards)
        ],  # Sum of dense rewards across each episode
        "ep_lengths": [len(overcooked_states)],  # Lengths of each episode
    }
    trajectories_0 = {
        k: np.array(v) if k not in ["ep_actions", "metadatas"] else v
        for k, v in trajectories_0.items()
    }

    trajectories_1 = {
        k: np.array(v) if k not in ["ep_actions", "metadatas"] else v
        for k, v in trajectories_1.items()
    }
    # if check_trajectories:
    #     agent_evaluator.check_trajectories(trajectories, verbose=not silent)

    return trajectories_0, trajectories_1

def load_data(bc_params, verbose=False):
    processed_trajs = get_trajectories(
        **bc_params["data_params"], silent=not verbose
    )
    # print((processed_trajs['cramped_room'][0].keys()))
    return processed_trajs

def build_dataset():
    layouts = [
        [
        "random3",
        "coordination_ring",
        "cramped_room",
        "random0",
        "asymmetric_advantages"]
        # [
        # "asymmetric_advantages_tomato",
        # "counter_circuit",
        # "cramped_corridor",
        # "inverse_marshmallow_experiment",
        # "marshmallow_experiment",
        # "marshmallow_experiment_coordination",
        # "soup_coordination",
        # "you_shall_not_pass",
        # ]
    ]
    train_data_paths = [
        CLEAN_2019_HUMAN_DATA_ALL,
        # CLEAN_2020_HUMAN_DATA_ALL
    ]
    # test_data_paths = [
    #     CLEAN_2019_HUMAN_DATA_TEST,
    #     CLEAN_2020_HUMAN_DATA_TEST
    # ]
    bc_params = DEFAULT_BC_PARAMS
    train_dataset = []
    test_dataset = []
    for data_path, layout in zip(train_data_paths, layouts):
        bc_params["data_params"]["layouts"] = layout
        bc_params["data_params"]["data_path"] = data_path
        train_dataset += load_data(bc_params)
    # for data_path, layout in zip(test_data_paths, layouts):
    #     bc_params["data_params"]["layouts"] = layout
    #     bc_params["data_params"]["data_path"] = data_path
    #     test_dataset += load_data(bc_params)

    # transfer 55 trajectories from test to train
    train_dataset = train_dataset[:-20]
    test_dataset = train_dataset[-20:]
    train_save_path = os.path.join(DATA_DIR, "train.pickle")
    test_save_path = os.path.join(DATA_DIR, "test.pickle")
    with open(train_save_path, "wb") as f:
        pickle.dump(train_dataset, f)
    with open(test_save_path, "wb") as f:
        pickle.dump(test_dataset, f)
    print(f"Load {len(train_dataset)} train trajectories and {len(test_dataset)} test trajectories")
    return train_dataset, test_dataset

class traj_buffer():
    def __init__(self, data_path, max_length=30, pad=6):
        self.max_length = max_length # GPT2 capacity
        self.pad = pad
        with open(data_path, 'rb') as f:
            self.dataset = pickle.load(f)
    def buf_len(self):
        return len(self.dataset)

    def get_batch(self, batch_size):
        def discount_cumsum(x, gamma):
            discount_cumsum = np.zeros_like(x)
            discount_cumsum[-1] = x[-1]
            for t in reversed(range(x.shape[0]-1)):
                discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
            return discount_cumsum
        rtn_data = {
            'ep_states': [],
            'ep_actions_0': [],
            'ep_actions_1': [],
            'ep_rtgs': [],
            'ep_timesteps': [],
            'mask': []
        }
        while batch_size > 0:
            select_list = random.sample(range(0, self.buf_len()), min(batch_size, self.buf_len()))
            batch_size -= len(select_list)
            for idx in select_list:
                # rtn_traj = {
                #     "ep_states": [],
                #     "ep_actions_0": [],
                #     "ep_actions_1": [],
                #     "ep_rewards": [],
                #     "ep_timesteps": [],
                #     "mask": []
                # }
                traj = self.dataset[idx]
                seq_len = len(traj['ep_states'])
                traj_rtg = discount_cumsum(traj['ep_rewards'], 1.)
                st_idx = random.randint(-self.max_length + 1, seq_len - 1)
                if st_idx < 0:
                    ed_idx = st_idx + self.max_length
                    states_pad = np.zeros_like(traj['ep_states'][0])
                    tmp_state = np.concatenate(([states_pad] * (-st_idx), traj['ep_states'][:ed_idx]), axis=0)
                    tmp_action_0 = traj['ep_actions_0'][:ed_idx]
                    tmp_action_1 = traj['ep_actions_1'][:ed_idx]
                    tmp_action_0 = [action for action_ in tmp_action_0 for action in action_]
                    tmp_action_1 = [action for action_ in tmp_action_1 for action in action_] 
                    tmp_action_0 = [self.pad] * (-st_idx) + tmp_action_0
                    tmp_action_1 = [self.pad] * (-st_idx) + tmp_action_1
                    tmp_rtgs = [0] * (-st_idx) + traj_rtg[:ed_idx].tolist()
                    tmp_timestep = [0] * (-st_idx) + [i for i in range(1, ed_idx+1)]
                    mask = [0] * (-st_idx) + [1] * ed_idx
                    # print(tmp_action_0)
                elif st_idx + self.max_length <= seq_len:
                    ed_idx = st_idx + self.max_length
                    tmp_state = traj['ep_states'][st_idx:ed_idx]
                    tmp_action_0 = traj['ep_actions_0'][st_idx:ed_idx]
                    tmp_action_1 = traj['ep_actions_1'][st_idx:ed_idx]
                    tmp_action_0 = [action for action_ in tmp_action_0 for action in action_]
                    tmp_action_1 = [action for action_ in tmp_action_1 for action in action_] 
                    tmp_rtgs = traj_rtg[st_idx:ed_idx].tolist()
                    tmp_timestep = [i for i in range(st_idx+1, ed_idx+1)]
                    mask = [1] * self.max_length
                else:
                    ed_idx = seq_len
                    states_pad = np.zeros_like(traj['ep_states'][0])
                    # print(st_idx, ed_idx)
                    # print(traj['ep_states'][0])
                    # print(ed_idx, st_idx)
                    # print([states_pad] * (self.max_length - ed_idx + st_idx))
                    # print(traj['ep_states'][st_idx:ed_idx])
                    # tmp_state = [states_pad] * (self.max_length - ed_idx + st_idx) + traj['ep_states'][st_idx:ed_idx]
                    tmp_state = np.concatenate(([states_pad] * (self.max_length - ed_idx + st_idx), traj['ep_states'][st_idx:ed_idx]), axis=0)
                    tmp_action_0 = traj['ep_actions_0'][st_idx:ed_idx]
                    tmp_action_1 = traj['ep_actions_1'][st_idx:ed_idx]
                    tmp_action_0 = [action for action_ in tmp_action_0 for action in action_]
                    tmp_action_1 = [action for action_ in tmp_action_1 for action in action_] 
                    tmp_action_0 = [self.pad] * (self.max_length - ed_idx + st_idx) + tmp_action_0
                    tmp_action_1 = [self.pad] * (self.max_length - ed_idx + st_idx) + tmp_action_1
                    tmp_rtgs = [0] * (self.max_length - ed_idx + st_idx) + traj_rtg[st_idx:ed_idx].tolist()
                    tmp_timestep = [0] * (self.max_length - ed_idx + st_idx) + [i for i in range(st_idx+1, ed_idx+1)]
                    mask = [0] * (self.max_length - ed_idx + st_idx) + [1] * (ed_idx - st_idx)
                rtn_data['ep_states'].append(tmp_state)
                rtn_data['ep_actions_0'].append(tmp_action_0)
                rtn_data['ep_actions_1'].append(tmp_action_1)
                rtn_data["ep_rtgs"].append(tmp_rtgs)
                rtn_data['ep_timesteps'].append(tmp_timestep)
                rtn_data['mask'].append(mask)
        return rtn_data

if __name__ == "__main__":
    build_dataset()
    # # train_dataset, test_dataset = build_dataset()
    buffer = traj_buffer(os.path.join(DATA_DIR, "train.pickle"))
    data = buffer.get_batch(1)
    print(data)
