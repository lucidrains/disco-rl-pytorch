# /// script
# requires-python = ">= 3.10"
# dependencies = [
#     "accelerate",
#     "assoc-scan",
#     "einx",
#     "einops",
#     "fire",
#     "gymnasium[box2d]",
#     "memmap-replay-buffer",
#     "moviepy",
#     "torch",
#     "torch-einops-utils",
#     "tqdm",
#     "wandb",
#     "x-mlps-pytorch",
#     "x-transformers"
# ]
# ///

import os
import shutil
from collections import deque

import fire
import gymnasium as gym
import numpy as np
import wandb
from einops import rearrange
from memmap_replay_buffer import ReplayBuffer
from tqdm import tqdm

from accelerate import Accelerator

import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.optim import Adam
from torch.utils._pytree import tree_map

from disco_rl_pytorch.disco_rl import (
    Policy,
    PolicyAdam,
    SharedMetaEmbed,
    MetaRNN,
    MetaNetwork,
    MetaValue,
    DiscoRL,
    Population,
    divisible_by,
    detach_tree
)

def cycle_dl(buffer, batch_size, seq_len):
    while True:
        yield from buffer.dataloader(
            batch_size = batch_size,
            n_steps = seq_len,
            sequence_fields = ('state', 'action', 'reward', 'terminated', 'log_prob'),
            shuffle = True,
            drop_last = True
        )

def main(
    total_steps: int = 100_000,
    batch_size: int = 16,
    seq_len: int = 16,
    train_every: int = 4,
    buffer_capacity: int = 50_000,
    buffer_max_episodes: int = 5000,
    dim: int = 64,
    num_actions: int = 4,
    dim_state: int = 8,
    depth: int = 2,
    lr_meta: float = 3e-4,
    lr_inner: float = 3e-4,
    gamma: float = 0.99,
    max_grad_norm: float = 1.0,
    record_every: int = 100,
    avg_cum_reward_episodes: int = 20,
    recordings_dir: str = './recordings',
    replay_buffer_dir: str = './replay_buffer',
    wandb_project: str = 'discorl-lunarlander',
    use_wandb: bool = False,
    cpu: bool = False,
    ema_beta: float = 0.99,
):
    accelerator = Accelerator(cpu = cpu)
    device = accelerator.device

    buffer_warmup = batch_size * seq_len * 2

    if use_wandb:
        wandb.init(project = wandb_project)

    shutil.rmtree(recordings_dir, ignore_errors = True)
    os.makedirs(recordings_dir, exist_ok = True)

    env = gym.make('LunarLander-v3', render_mode = 'rgb_array')
    env = gym.wrappers.RecordVideo(env, recordings_dir, episode_trigger = lambda ep: divisible_by(ep, record_every))

    policy = Population(Policy(dim = dim, dim_state = dim_state, num_actions = num_actions, depth = depth))
    shared_meta_embed = SharedMetaEmbed(dim = dim, num_actions = num_actions, dim_abstract_observation = dim, dim_abstract_action = dim)
    meta_rnn = MetaRNN(dim = dim)
    meta_network = MetaNetwork(dim = dim, num_actions = num_actions, dim_abstract_observation = dim, dim_abstract_action = dim, dim_condition = dim)
    meta_value = MetaValue(dim = dim, dim_state = dim_state, depth = depth)

    discorl = DiscoRL(
        policy = policy,
        policy_optimizer = PolicyAdam(lr = lr_inner),
        shared_meta_embed = shared_meta_embed,
        meta_rnn = meta_rnn,
        meta_network = meta_network,
        meta_value_network = meta_value,
        update_steps = seq_len,
        gamma = gamma
    ).to(device)

    meta_params = [
        *shared_meta_embed.parameters(),
        *meta_rnn.parameters(),
        *meta_network.parameters(),
        *meta_value.parameters(),
    ]

    meta_optimizer = Adam(meta_params, lr = lr_meta)

    buffer = ReplayBuffer(
        folder = replay_buffer_dir,
        max_episodes = buffer_max_episodes,
        max_timesteps = buffer_capacity,
        fields = dict(
            state = ('float', (dim_state,)),
            action = 'int',
            reward = 'float',
            terminated = 'bool',
            log_prob = 'float',
        )
    )

    ep_rewards = deque(maxlen = avg_cum_reward_episodes)
    dl = cycle_dl(buffer, batch_size, seq_len)

    obs, _ = env.reset()
    ep_ret = 0.

    pbar = tqdm(total = total_steps, desc = 'training')

    params = tree_map(lambda t: t.clone(), policy.init_params(batch_size))
    ema_params = tree_map(lambda t: t.clone(), params)
    optim_states = discorl.policy_optimizer.init_optim_states(params)

    with torch.no_grad():
        acting_params = tree_map(lambda t: t.clone(), policy.init_params(1))

    for ind in range(total_steps):
        step = ind + 1
        with torch.no_grad():
            t_obs = torch.tensor(obs, dtype = torch.float32, device = device)
            t_obs = rearrange(t_obs, 'd -> 1 1 d')
            out = policy(t_obs, params = acting_params, sample = True)

            action = out.actions[0, 0]
            log_prob = Categorical(logits = out.action_logits[0, 0]).log_prob(action)

        action_np = action.item()
        next_obs, reward, terminated, truncated, _ = env.step(action_np)
        ep_ret += reward

        buffer.store(
            state = obs,
            action = action_np,
            reward = reward,
            terminated = terminated,
            log_prob = log_prob.item(),
        )

        obs = next_obs
        pbar.update(1)

        if terminated or truncated:
            ep_rewards.append(ep_ret)
            avg = np.mean(ep_rewards)

            if use_wandb:
                wandb.log(dict(episode_reward = ep_ret, avg_cum_reward = avg, step = step))

            pbar.set_postfix(dict(avg_cum_reward = f'{avg:.1f}'))
            obs, _ = env.reset()
            ep_ret = 0.

        if buffer.timestep_index < buffer_warmup or not divisible_by(step, train_every):
            continue

        batch = next(dl)

        states = batch['seq_state'].float().to(device)
        actions = batch['seq_action'].long().to(device)
        rewards = batch['seq_reward'].float().to(device)
        terminals = batch['seq_terminated'].bool().to(device)
        old_log_probs = batch['seq_log_prob'].float().to(device)

        meta_optimizer.zero_grad()

        loss_return = discorl.loss(
            state = states,
            actions = actions,
            rewards = rewards,
            terminated = terminals,
            old_log_probs = old_log_probs,
            params = params,
            ema_params = ema_params,
            optim_states = optim_states,
        )

        meta_loss = loss_return.meta_value_loss + loss_return.meta_policy_loss
        meta_loss.backward()

        torch.nn.utils.clip_grad_norm_(meta_params, max_grad_norm)
        meta_optimizer.step()

        # detach agent parameters to truncate backprop between outer loops
        params, optim_states = detach_tree((loss_return.out.params, loss_return.out.optim_states))

        with torch.no_grad():

            # update ema params
            for e, p in zip(ema_params.values(), params.values()):
                e.lerp_(p, 1. - ema_beta)

            # update behavior policy with the population average

            for p, agent_p in zip(acting_params.values(), params.values()):
                p.copy_(agent_p.mean(dim = 0, keepdim = True))

        if use_wandb:
            wandb.log(dict(
                meta_value_loss = loss_return.meta_value_loss.item(),
                meta_policy_loss = loss_return.meta_policy_loss.item(),
                meta_kl_loss = loss_return.meta_regularization_kl_loss.item(),
                meta_loss = meta_loss.item(),
            ))

    pbar.close()
    env.close()

    if use_wandb:
        wandb.finish()

if __name__ == '__main__':
    fire.Fire(main)
