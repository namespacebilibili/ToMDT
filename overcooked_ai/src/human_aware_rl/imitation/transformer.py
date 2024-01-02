import torch
import random
import torch.nn as nn
import numpy as np
import transformers

from human_aware_rl.imitation.gpt2 import GPT2Model

class TrajectoryModel(nn.Module):

    def __init__(self, state_dim, act_dim, max_length=None):
        super().__init__()

        self.state_dim = state_dim
        self.act_dim = act_dim
        self.max_length = max_length

    def forward(self, states, action0s, action1s, timesteps, attention_mask=None):
        # "masked" tokens or unspecified inputs can be passed in as None
        return None, None, None

    def get_action(self, states, action0s, action1s, timesteps, **kwargs):
        # these will come as tensors on the correct device
        return torch.zeros_like(action0s[-1])

class StateEmbedding(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU()
        )
    def forward(self, states):
        return self.mlp(states)
    
class DecisionTransformer(TrajectoryModel):
    def __init__(
            self, 
            state_dim, 
            act_dim, 
            hidden_size,
            max_length=None,
            max_ep_len=1250,
            **kwargs
    ):
        super().__init__(state_dim, act_dim, max_length=max_length)
        self.hidden_size = hidden_size
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        config = transformers.GPT2Config(
            vocab_size=1, # doesn't matter -- we don't use the vocab
            n_embd=hidden_size,
            n_ctx=400,
            **kwargs
        )

        self.transformer = GPT2Model(config)
        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_return = torch.nn.Linear(1, hidden_size)
        self.embed_state = StateEmbedding(state_dim, hidden_size)
        self.embed_action0 = nn.Embedding(act_dim+1, hidden_size)
        self.embed_action1 = nn.Embedding(act_dim+1, hidden_size)
        self.embed_role = nn.Embedding(2, hidden_size)
        self.embed_ln = nn.LayerNorm(hidden_size)

        self.predict_action0 = nn.Sequential(
            *([nn.Linear(hidden_size, self.act_dim)])
        )

        self.predict_action1 = nn.Sequential(
            *([nn.Linear(hidden_size, self.act_dim)])
        )

        self.predict_return = torch.nn.Linear(hidden_size, 1)
        # self.predict_state = nn.Sequential(
        #     *([nn.Linear(hidden_size, self.state_dim)])
        # )

    def forward(self, states, action0s, action1s, returns_to_go, timesteps, attention_mask=None):
        batch_size, seq_length = len(states), len(states[0])
        
        if attention_mask is None:
            attention_mask = np.ones((batch_size, seq_length))

        state_embeddings = self.embed_state(torch.tensor(states, dtype = torch.float).to(self.device))
        action0s = torch.tensor(action0s, dtype = torch.long).to(self.device)
        action1s = torch.tensor(action0s, dtype = torch.long).to(self.device)
        action0_embeddings = self.embed_action0(action0s)
        action1_embeddings = self.embed_action1(action1s)
        timestep_embeddings = self.embed_timestep(torch.tensor(timesteps, dtype = torch.long).to(self.device))
        returns_embeddings = self.embed_return(torch.tensor(returns_to_go, dtype = torch.float).unsqueeze(-1).to(self.device))
        self_embeddings = self.embed_role(torch.zeros_like(action0s).to(self.device))
        teammate_embeddings = self.embed_role(torch.ones_like(action1s).to(self.device))

        # shape: (batch_size, seq_length, hidden_size)
        state_embeddings = state_embeddings + timestep_embeddings 
        action0_embeddings = action0_embeddings + self_embeddings + timestep_embeddings
        action1_embeddings = action1_embeddings + teammate_embeddings + timestep_embeddings
        returns_embeddings = returns_embeddings + timestep_embeddings

        stacked_inputs = torch.stack((
            returns_embeddings, state_embeddings, action0_embeddings, action1_embeddings
        ), dim=1).to(self.device) # shape: (batch_size, 3, seq_length, hidden_size) 

        # (S, a0, a1)
        # stacked_inputs = stacked_inputs.permute(0, 2, 1, 3).reshape(batch_size, 3*seq_length, self.hidden_size)
        stacked_inputs = stacked_inputs.permute(0, 2, 1, 3).reshape(batch_size, 4*seq_length, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)

        attention_mask = torch.from_numpy(attention_mask).to(self.device)
        stacked_attention_mask = torch.stack((
            attention_mask, attention_mask, attention_mask, attention_mask
        ), dim=1).to(self.device)
        # stacked_attention_mask = stacked_attention_mask.permute(0, 2, 1).reshape(batch_size, 3*seq_length)
        stacked_attention_mask = stacked_attention_mask.permute(0, 2, 1).reshape(batch_size, 4*seq_length)
        transformers_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask
        )

        x = transformers_outputs['last_hidden_state']
        # x = x.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)
        x = x.reshape(batch_size, seq_length, 4, self.hidden_size).permute(0, 2, 1, 3)
        # 0 for rtg, 1 for state, 2 for action0, 3 for action1
        # predict action given state
        action0_preds = self.predict_action0(x[:, 1])
        action1_preds = self.predict_action1(x[:, 1])
        return_preds = self.predict_return(x[:, 3])

        return action0_preds, action1_preds, return_preds

    def get_action(self, states, action0s, action1s, returns_to_go, timesteps, **kwargs):
        if self.max_length is not None:
            states = states[-self.max_length:]
            action0s = action0s[-self.max_length:]
            action1s = action1s[-self.max_length:]
            timesteps = timesteps[-self.max_length:]
            returns_to_go = returns_to_go[-self.max_length:]
            attention_mask = np.concatenate((
                np.zeros((1, self.max_length - len(states))),
                np.ones((1, len(states)))
            ), axis=1)
            states_pad = np.zeros_like(states[0])
            states = [states_pad] * (self.max_length - len(states)) + states
            action_pad = self.act_dim
            action0s = [action_pad] * (self.max_length - len(action0s)) + action0s
            action1s = [action_pad] * (self.max_length - len(action1s)) + action1s
            returns_to_go = [0] * (self.max_length - len(returns_to_go)) + returns_to_go
            timesteps = [0] * (self.max_length - len(timesteps)) + timesteps
        else:
            attention_mask = None
        
        states = np.array(states).reshape(1, -1, self.state_dim)
        action0s = np.array(action0s).reshape(1, -1)
        action1s = np.array(action1s).reshape(1, -1)
        returns_to_go = np.array(returns_to_go).reshape(1, -1)
        timesteps = np.array(timesteps).reshape(1, -1)

        action0_preds, action1_preds, return_preds = self.forward(
            states, action0s, action1s, returns_to_go, timesteps, attention_mask=attention_mask, **kwargs
        )
        return action0_preds[0, -1, :], action1_preds[0, -1, :], return_preds[0, -1]




