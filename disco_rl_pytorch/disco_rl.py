from __future__ import annotations
from collections import namedtuple

import torch
from torch import nn, cat
import torch.nn.functional as F
from torch.nn import Sequential, Linear, Module, ModuleList, LSTM, RMSNorm

from einops import pack

from x_mlps_pytorch.normed_mlp import create_mlp, MLP
from x_transformers import Decoder, Encoder

# constants

PolicyOutput = namedtuple('PolicyOutput', (
    'action_logits',
    'encoded_observations',
    'actions',
    'sampled_actions',
    'pred_action_value',
    'pred_next_action_logits'
), defaults = (None,) * 4)

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

        self.meta_head_norms = ModuleList([RMSNorm(dim), RMSNorm(dim), RMSNorm(dim)])

        # meta learned output heads
        # 1. actions, main policy
        # 2. encoding for observation y(s)
        # 3. encoding for observation + actions z(s, a)

        self.to_action_logits = Linear(dim, num_actions, bias = False)

        self.to_encoded_observation = MLP(dim, dim * 2, dim_abstract_observation)

        self.to_encoded_action = MLP(dim + num_actions, dim * 2, dim_abstract_observation)

        # prediction heads

        self.output_norms = ModuleList([RMSNorm(dim), RMSNorm(dim)])

        self.to_action_value = MLP(dim + num_actions, dim * 2, 1)

        self.to_next_action_pred = MLP(dim + num_actions, dim * 2, num_actions)

    def forward(
        self,
        state,
        sample = False
    ):
        embed = self.to_embed(state)

        action_logit_input, encoded_observation_input, encoded_action_input = (norm(embed) for norm in self.meta_head_norms)

        action_logits = self.to_action_logits(action_logit_input)

        encoded_observations = self.to_encoded_observation(encoded_observation_input)

        def get_encoded_action(action):
            action_one_hot = F.one_hot(action, num_classes = self.num_actions).float()
            return self.to_encoded_action((encoded_action_input, action_one_hot))

        def get_pred(action):
            action_value_input, next_action_pred_input = (norm(embed) for norm in self.output_norms)
            action_one_hot = F.one_hot(action, num_classes = self.num_actions).float()

            pred_action_value = self.to_action_value((action_value_input, action_one_hot))
            pred_next_action_logits = self.to_next_action_pred((next_action_pred_input, action_one_hot))

            return pred_action_value, pred_next_action_logits

        if not sample:
            return PolicyOutput(action_logits, encoded_observations, get_encoded_action)

        # sample action

        sampled_action = gumbel_sample(action_logits)

        # get the heads that depend on sampled action

        encoded_actions = get_encoded_action(sampled_action)

        pred_action_value, pred_next_action_logits = get_pred(sampled_action)

        return PolicyOutput(action_logits, encoded_observations, sampled_action, encoded_actions, pred_action_value, pred_next_action_logits)

# meta network(s) related

class SharedMetaEmbed(Module):
    def __init__(
        self,
        dim,
        num_actions,
        dim_abstract_observation,
        dim_abstract_action,
        mlp_depth = 2,
        mlp_expansion = 2.,
    ):
        super().__init__()

        dim_in = (
            num_actions + 2 +          # one hot actions, rewards, terminated
            num_actions +              # action dist
            dim_abstract_action +      # encoded actions
            dim_abstract_observation + # encoded observation
            + 1                        # pred q value
        )

        self.to_embed = create_mlp(dim, dim_in = dim_in, depth = mlp_depth)
        self.num_actions = num_actions

    def forward(
        self,
        actions,
        rewards,
        terminated,
        action_logits,
        encoded_observations,
        encoded_actions,
        pred_action_value
    ):

        actions_one_hot = F.one_hot(actions, self.num_actions)

        action_dist = action_logits.softmax(dim = -1)

        concatted_inputs, _ = pack((actions_one_hot, rewards, terminated.float(), action_dist, encoded_actions, encoded_observations, pred_action_value), 'b t *')

        embeds = self.to_embed(concatted_inputs)

        return embeds

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
