from __future__ import annotations

import torch
from torch import nn, cat
import torch.nn.functional as F
from torch.nn import Sequential, Linear, Module, ModuleList, LSTM, RMSNorm

from x_mlps_pytorch.normed_mlp import create_mlp, MLP
from x_transformers import Decoder, Encoder

# functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# sampling

def log(t, eps = 1e-20):
    return t.clamp_min(eps).log()

def gumbel_noise(t):
    return -log(-log(torch.rand_like(t)))

def gumbel_sample(t, dim = -1, keepdim = False):
    t = t + gumbel_noise(t)
    return t.argmax(dim = dim, keepdim = keepdim)

# classes

class Policy(Module):
    def __init__(
        self,
        dim,
        *,
        dim_state,
        num_actions,
        depth,
        dim_abstract_observation = None,
        dim_abstract_action = None
    ):
        super().__init__()
        dim_abstract_observation = default(dim_abstract_observation, dim)
        dim_abstract_action = default(dim_abstract_action, dim)

        self.num_actions = num_actions

        self.to_embed = create_mlp(
            dim_in = dim_state,
            dim = dim,
            depth = depth
        )

        self.norms = ModuleList([RMSNorm(dim), RMSNorm(dim), RMSNorm(dim)])

        # heads
        # 1. actions, main policy
        # 2. encoding for observation y(s)
        # 3. encoding for observation + actions z(s, a)

        self.to_action_logits = Linear(dim, num_actions, bias = False)

        self.to_encoded_observation = MLP(dim, dim * 2, dim_abstract_observation)

        self.to_encoded_action = MLP(dim + num_actions, dim * 2, dim_abstract_observation)

    def forward(
        self,
        state,
        sample = False
    ):
        embed = self.to_embed(state)

        action_logit_input, encoded_observation_input, encoded_action_input = (norm(embed) for norm in self.norms)

        action_logits = self.to_action_logits(action_logit_input)

        encoded_observations = self.to_encoded_observation(encoded_observation_input)

        def get_encoded_action(action):
            action_one_hot = F.one_hot(action, num_classes = self.num_actions).float()
            return self.to_encoded_action((encoded_action_input, action_one_hot))

        if not sample:
            return action_logits, encoded_observations, get_encoded_action

        sampled_action = gumbel_sample(action_logits)

        encoded_actions = get_encoded_action(sampled_action)

        return sampled_action, action_logits, encoded_observations, encoded_actions

# main class

class DiscoRL(Module):
    def __init__(
        self,
        policy: Module,
        policy_optimizer: Module,
        meta_network: Module,
        meta_value_network: Module,
    ):
        super().__init__()

        self.meta_network = meta_network
        self.meta_value_network = meta_value_network

        self.policy = policy
        self.policy_optimizer = policy_optimizer

    def forward(
        self,
        state,
        actions,
        rewards,
        terminated,
        lens = None  # (b,)
    ):
        return state
