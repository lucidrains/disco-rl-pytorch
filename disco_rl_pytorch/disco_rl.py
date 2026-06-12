from __future__ import annotations

import math
from functools import partial
from collections import namedtuple

import einx
import torch
import torch.nn.functional as F
from torch import nn, cat, is_tensor
from torch.nn import Sequential, Linear, Module, ModuleList, LSTM, RMSNorm

from torch.autograd import grad as torch_grad
from torch.func import vmap, grad, functional_call

from einops import pack, rearrange, repeat
from einops.layers.torch import Reduce

from torch_einops_utils import tree_map_tensor, shift_left

from x_mlps_pytorch.normed_mlp import create_mlp, MLP
from x_transformers import Decoder, Encoder

from assoc_scan import AssocScan

# constants

PolicyOutput = namedtuple('PolicyOutput', (
    'action_logits',
    'encoded_observations',
    'actions',
    'encoded_actions',
    'pred_action_value',
    'pred_next_action_logits'
), defaults = (None,) * 4)

MetaNetworkOutput = namedtuple('MetaNetworkOutput', (
    'target_action_logits',
    'target_encoded_observations',
    'target_encoded_actions',
    'loss_weight'
), defaults = (None,))

DiscoRLOutput = namedtuple('DiscoRLOutput', (
    'target_action_logits',
    'values',
    'loss_weights'
), defaults = (None,))

AdamState = namedtuple('AdamState', (
    'time',
    'moments',
    'variances'
))

LinearNoBias = partial(Linear, bias = False)

# functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0

# sampling

def log(t, eps = 1e-20):
    if not is_tensor(t):
        return math.log(max(t, eps))

    return t.clamp_min(eps).log()

def gumbel_noise(t):
    return -log(-log(torch.rand_like(t)))

def gumbel_sample(t, dim = -1, keepdim = False):
    t = t + gumbel_noise(t)
    return t.argmax(dim = dim, keepdim = keepdim)

# tensor helpers

def detach_tree(t):
    return tree_map_tensor(lambda t: t.detach(), t)

def rescale(t, from_range, to_range, eps = 1e-6):
    from_min, from_max = from_range
    to_min, to_max = to_range
    return (t - from_min) / max(from_max - from_min, eps) * (to_max - to_min) + to_min

def forward_kl(logits, target_logits, weight = None):
    log_probs = logits.log_softmax(dim = -1)
    target_prob = target_logits.softmax(dim = -1)

    kl = F.kl_div(log_probs, target_prob, reduction = 'none').sum(dim = -1)

    if exists(weight):
        kl = kl * weight

    return kl.mean()

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

        self.to_action_logits = LinearNoBias(dim, num_actions)

        self.to_encoded_observation = MLP(dim, dim * 2, dim_abstract_observation)

        self.to_encoded_action = MLP(dim + num_actions, dim * 2, dim_abstract_observation)

        # prediction heads

        self.output_norms = ModuleList([RMSNorm(dim), RMSNorm(dim)])

        self.to_action_value = MLP(dim + num_actions, dim * 2, 1)

        self.to_next_action_pred = MLP(dim + num_actions, dim * 2, num_actions)

    def get_encoded_action(
        self,
        embed,
        action
    ):
        encoded_action_input = self.meta_head_norms[-1](embed)
        action_one_hot = F.one_hot(action, num_classes = self.num_actions).float()
        return self.to_encoded_action((encoded_action_input, action_one_hot))

    def get_pred(
        self,
        embed,
        action
    ):
        action_value_input, next_action_pred_input = (norm(embed) for norm in self.output_norms)
        action_one_hot = F.one_hot(action, num_classes = self.num_actions).float()

        pred_action_value = self.to_action_value((action_value_input, action_one_hot))
        pred_next_action_logits = self.to_next_action_pred((next_action_pred_input, action_one_hot))

        return pred_action_value, pred_next_action_logits

    def forward(
        self,
        state,
        actions = None,
        sample = False
    ):
        assert not (sample and exists(actions))

        embed = self.to_embed(state)

        action_logit_input, encoded_observation_input = (norm(embed) for norm in self.meta_head_norms[:2])

        action_logits = self.to_action_logits(action_logit_input)

        encoded_observations = self.to_encoded_observation(encoded_observation_input)

        if not sample and not exists(actions):
            return action_logits, encoded_observations

        # action given or sample action

        if exists(actions):
            action = actions
        else:
            action = gumbel_sample(action_logits)

        # get the heads that depend on sampled action

        encoded_actions = self.get_encoded_action(embed, action)

        pred_action_value, pred_next_action_logits = self.get_pred(embed, action)

        return PolicyOutput(action_logits, encoded_observations, action, encoded_actions, pred_action_value, pred_next_action_logits)

# film

class FiLM(Module):
    def __init__(
        self,
        dim,
        dim_cond
    ):
        super().__init__()
        self.norm = RMSNorm(dim, elementwise_affine = False)

        self.to_gamma_beta = Linear(dim_cond, dim * 2, bias = False)
        torch.nn.init.zeros_(self.to_gamma_beta.weight)

    def forward(
        self,
        tokens,
        cond
    ):
        normed = self.norm(tokens)

        gamma, beta = self.to_gamma_beta(cond).chunk(2, dim = -1)
        gamma, beta = (t.expand_as(normed) for t in (gamma, beta))

        scaled = einx.multiply('b n d, b n d', normed, gamma + 1.)
        return einx.add('b n d, b n d', scaled, beta)

# v-trace / retrace

class RLTrace(Module):
    def __init__(
        self,
        gamma = 0.99,
        lam = 1.,
        clip_rhos = 1.,
        clip_trace_weights = 1.,
        is_retrace = False
    ):
        super().__init__()
        self.gamma = gamma
        self.lam = lam
        self.clip_rhos = clip_rhos
        self.clip_trace_weights = clip_trace_weights
        self.scan = AssocScan(reverse = True)

        self.is_retrace = is_retrace

    def forward(
        self,
        values,
        rewards,
        log_probs,
        old_log_probs,
        terminated,
        next_values = None,
        next_value = 0.
    ):
        not_terminated = 1. - terminated.float()

        log_rhos = log_probs - old_log_probs
        rhos = log_rhos.exp()

        # trace weights

        trace_weights = rhos.clamp(max = self.clip_trace_weights)

        if self.is_retrace:
            trace_weights = trace_weights * self.lam
            trace_weights = shift_left(trace_weights, dim = 1)

        gates = self.gamma * trace_weights * not_terminated

        # values

        assert not (self.is_retrace and not exists(next_values)), 'next_values must be explicitly provided for Retrace'

        if not exists(next_values):
            next_values = shift_left(values, dim = 1, pad_value = next_value)

        next_values = next_values * not_terminated

        # td errors

        delta_values = rewards + self.gamma * next_values - values

        if not self.is_retrace:
            delta_values = delta_values * rhos.clamp(max = self.clip_rhos)

        # scan

        return self.scan(gates, delta_values) + values

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
            1                          # pred q value
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

class MetaNetwork(Module):
    def __init__(
        self,
        dim,
        num_actions,
        dim_abstract_observation,
        dim_abstract_action,
        lstm_kwargs: dict = dict(),
        adaptive_loss_weight = False,
        loss_weight_range = (1e-2, 10.),
        dim_condition = None
    ):
        super().__init__()

        self.rnn = LSTM(dim, dim, batch_first = True, **lstm_kwargs)

        # condition from forward meta-rnn

        dim_condition = default(dim_condition, dim)
        self.condition_film = FiLM(dim, dim_condition)

        # norms and output heads

        num_norms = 4 if adaptive_loss_weight else 3
        self.norms = ModuleList([nn.RMSNorm(dim) for _ in range(num_norms)])

        self.to_target_action_logits = LinearNoBias(dim, num_actions)

        self.to_target_encoded_observation = LinearNoBias(dim, dim_abstract_observation)

        self.to_target_encoded_action = LinearNoBias(dim, dim_abstract_action)

        # adaptive loss weight

        self.adaptive_loss_weight = adaptive_loss_weight
        self.loss_weight_range = loss_weight_range

        if adaptive_loss_weight:
            self.to_loss_weight_logits = LinearNoBias(dim, 3)

    def forward(
        self,
        shared_meta_embed,
        hidden = None,
        condition = None
    ):
        time_backwards_shared_embed = shared_meta_embed.flip(dims = (1,))

        rnn_encoded, _ = self.rnn(time_backwards_shared_embed, hidden)

        rnn_encoded = rnn_encoded.flip(dims = (1,))

        if exists(condition):
            rnn_encoded = self.condition_film(rnn_encoded, condition)

        target_action_logits, target_encoded_observation, target_encoded_action = (fn(norm(rnn_encoded)) for norm, fn in zip(self.norms[:3], (self.to_target_action_logits, self.to_target_encoded_observation, self.to_target_encoded_action)))

        loss_weight = None

        if self.adaptive_loss_weight:
            weight_logits = self.to_loss_weight_logits(self.norms[3](rnn_encoded))

            log_loss_weight = rescale(
                weight_logits.sigmoid(),
                (0., 1.),
                tuple(map(log, self.loss_weight_range))
            )

            loss_weight = log_loss_weight.exp()

        output = MetaNetworkOutput(target_action_logits, target_encoded_observation, target_encoded_action, loss_weight)

        return output

    def loss(
        self,
        preds: PolicyOutput,
        targets: MetaNetworkOutput
    ):
        weight = targets.loss_weight

        weight_action, weight_obs, weight_encoded_action = (None, None, None)

        if exists(weight):
            weight_action, weight_obs, weight_encoded_action = weight.unbind(dim = -1)

        loss = (
            forward_kl(preds.action_logits, targets.target_action_logits, weight = weight_action) +
            forward_kl(preds.encoded_observations, targets.target_encoded_observations, weight = weight_obs) +
            forward_kl(preds.encoded_actions, targets.target_encoded_actions, weight = weight_encoded_action)
        )

        return loss

class MetaRNN(Module):
    def __init__(
        self,
        dim,
        lstm_kwargs: dict = dict(),
        encoder_pool_kwargs: dict = dict()
    ):
        super().__init__()

        self.experience_pool = Sequential(
            Encoder(dim = dim, depth = 2, **encoder_pool_kwargs),
            Reduce('b t d -> 1 1 d', 'mean')
        )

        self.rnn = LSTM(dim, dim, batch_first = True, **lstm_kwargs)

    def forward(
        self,
        shared_meta_embed,
        hiddens = None
    ):
        shared_meta_embed = self.experience_pool(shared_meta_embed)

        return self.rnn(shared_meta_embed, hiddens)

class MetaValue(Module):
    def __init__(
        self,
        dim,
        dim_state,
        depth,
    ):
        super().__init__()

        self.to_value = create_mlp(
            dim,
            dim_in = dim_state,
            dim_out = 1,
            depth = depth
        )

    def forward(
        self,
        state
    ):
        value = self.to_value(state)
        return rearrange(value, '... 1 -> ...')

# vectorized

class Population(Module):
    def __init__(
        self,
        model: Module
    ):
        super().__init__()

        self.model = model

        def forward(params, state, kwargs):
            return functional_call(model, params, state, kwargs = kwargs)

        def forward_with_actions(params, state, actions, kwargs):
            return functional_call(model, params, state, kwargs = {**kwargs, 'actions': actions})

        self.vmap_forward = vmap(forward, in_dims = (0, 0, None), out_dims = 0, randomness = 'different')
        self.vmap_forward_with_actions = vmap(forward_with_actions, in_dims = (0, 0, 0, None), out_dims = 0, randomness = 'different')

    def init_params(self, batch):
        params = self.model.named_parameters()
        return {name: repeat(t, '... -> b ...', b = batch) for name, t in params}

    def forward(
        self,
        state,
        params = None,
        actions = None,
        **kwargs
    ):
        batch = state.shape[0]

        params = params if exists(params) else self.init_params(batch)

        if exists(actions):
            return self.vmap_forward_with_actions(params, state, actions, kwargs)

        return self.vmap_forward(params, state, kwargs)

# vectorized adam

class Adam(Module):
    def __init__(
        self,
        lr = 5e-4,
        betas = (0.9, 0.999),
        eps = 1e-8
    ):
        super().__init__()

        self.betas = betas
        self.eps = eps
        self.lr = lr

    def init_optim_states(
        self,
        params
    ):
        moment = {name: torch.zeros_like(t) for name, t in params.items()}
        variance = {name: torch.zeros_like(t) for name, t in params.items()}
        return AdamState(0, moment, variance)

    def forward(
        self,
        optim_state,
        loss,
        params,
        detach_grads = False
    ):
        time, moments, variances = optim_state
        time += 1

        beta1, beta2 = self.betas
        eps, lr = self.eps, self.lr

        # handle dictionary structure for grad

        param_names = params.keys()

        grad_values = torch_grad(
            loss,
            tuple(params.values()),
            only_inputs = True,
            create_graph= True,
            retain_graph = True,
            allow_unused = True
        )

        # maybe detach, for TBPTT

        if detach_grads:
            grad_values = detach_tree(grad_values)

        # back to dict[str, Tensor]

        grads = dict(zip(param_names, grad_values))

        # doing the gradient step, and meta learning through it

        next_params = dict()
        next_moments = dict()
        next_variances = dict()

        for name, grad in grads.items():
            param = params[name]

            if not exists(grad):
                next_params[name] = param
                next_moments[name] = moments[name]
                next_variances[name] = variances[name]
                continue

            moment, variance = moments[name], variances[name]

            # ema

            next_moment = moment.lerp(grad, 1. - beta1)
            next_variance = variance.lerp(grad ** 2, 1. - beta2)

            # correction

            unbiased_moment = next_moment / (1. - beta1 ** time)
            unbiased_variance = next_variance / (1. - beta2 ** time)

            # update params

            update = unbiased_moment * unbiased_variance.add(eps).rsqrt()

            next_params[name] = param - update * lr

            # save next moment and variance

            next_moments[name] = next_moment
            next_variances[name] = next_variance

        next_optim_states = AdamState(time, next_moments, next_variances)
        return next_optim_states, next_params

# main class

class DiscoRL(Module):
    def __init__(
        self,
        policy: Policy | Module,
        policy_optimizer: Adam | Module,
        shared_meta_embed: SharedMetaEmbed | Module,
        meta_rnn: MetaRNN | Module,
        meta_network: MetaNetwork | Module,
        meta_value_network: MetaValue | Module,
        update_steps = 10,
        detach_every = 0,
    ):
        super().__init__()

        self.policy = policy
        self.policy_optimizer = policy_optimizer
        self.shared_meta_embed = shared_meta_embed

        self.meta_rnn = meta_rnn
        self.meta_network = meta_network
        self.meta_value_network = meta_value_network

        self.update_steps = update_steps
        self.detach_every = detach_every
        self.should_detach = detach_every > 0

    def forward(
        self,
        state,
        actions,
        rewards,
        terminated,
        params = None,
        optim_states = None,
        lens = None  # (b,)
    ):
        batch, time = state.shape[:2]
        steps = self.update_steps

        if not exists(params):
            params = self.policy.init_params(batch)

        if not exists(optim_states):
            optim_states = self.policy_optimizer.init_optim_states(params)

        chunks = zip(
            state.split(steps, dim = 1),
            actions.split(steps, dim = 1),
            rewards.split(steps, dim = 1),
            terminated.split(steps, dim = 1)
        )

        all_target_action_logits = []
        all_loss_weights = []

        hiddens = None

        for ind, (state_chunk, actions_chunk, rewards_chunk, terminated_chunk) in enumerate(chunks, start = 1):

            policy_output = self.policy(state_chunk, params = params, actions = actions_chunk)

            embeds = self.shared_meta_embed(
                actions_chunk,
                rewards_chunk,
                terminated_chunk,
                policy_output.action_logits,
                policy_output.encoded_observations,
                policy_output.encoded_actions,
                policy_output.pred_action_value
            )

            condition, hiddens = self.meta_rnn(embeds, hiddens)

            meta_network_output = self.meta_network(embeds, condition = condition)

            all_target_action_logits.append(meta_network_output.target_action_logits)

            if exists(meta_network_output.loss_weight):
                all_loss_weights.append(meta_network_output.loss_weight)

            loss = self.meta_network.loss(policy_output, meta_network_output)

            detach_grads = self.should_detach and divisible_by(ind, self.detach_every)

            optim_states, params = self.policy_optimizer(optim_states, loss, params, detach_grads = detach_grads)

            if detach_grads:
                hiddens, params, optim_states = detach_tree((hiddens, params, optim_states))

        values = self.meta_value_network(state)

        target_action_logits = cat(all_target_action_logits, dim = 1)
        loss_weights = cat(all_loss_weights, dim = 1) if all_loss_weights else None

        return DiscoRLOutput(target_action_logits, values, loss_weights)
