import pytest
parametrize = pytest.mark.parametrize

import torch
import torch.nn.functional as F

@parametrize('adaptive_loss_weight', (True, False))
def test_components(adaptive_loss_weight):
    from disco_rl_pytorch.disco_rl import (
        SharedMetaEmbed,
        MetaNetwork,
        MetaRNN,
        MetaValue,
        Policy,
        Population,
        Adam,
        forward_kl,
        PolicyOutput
    )

    model = Policy(dim = 32, dim_state = 8, num_actions = 4, depth = 2)

    population = Population(model)

    params = population.init_params(7)

    states = torch.randn(7, 20, 8)

    action_logits, encoded_observations, actions, encoded_actions, pred_action_value, pred_next_action = population(states, params = params, sample = True)

    assert actions.shape == (7, 20)

    rewards = torch.randn(7, 20)
    terminated = torch.zeros(7, 20).bool()

    embedder = SharedMetaEmbed(dim = 32, num_actions = 4, dim_abstract_observation = 32, dim_abstract_action = 32)

    embeds = embedder(actions, rewards, terminated, action_logits, encoded_observations, encoded_actions, pred_action_value)

    assert embeds.shape == (7, 20, 32)

    # meta rnn

    meta_rnn = MetaRNN(dim = 32)

    condition, hiddens = meta_rnn(embeds)

    # meta network

    meta_network = MetaNetwork(
        dim = 32,
        num_actions = 4,
        dim_abstract_action = 32,
        dim_abstract_observation = 32,
        adaptive_loss_weight = adaptive_loss_weight
    )

    meta_network_outputs = meta_network(embeds, condition = condition)

    policy_outputs = PolicyOutput(action_logits, encoded_observations, actions, encoded_actions, pred_action_value, pred_next_action)

    loss = meta_network.loss(policy_outputs, meta_network_outputs)

    population_optimizer = Adam()

    optim_states = population_optimizer.init_optim_states(params)

    next_optim_states, next_params = population_optimizer(optim_states, loss, params)

    assert next_params.keys() == params.keys()
    assert loss.numel() == 1

    value_network = MetaValue(dim = 32, dim_state = 8, depth = 4)

    values = value_network(states)

    assert values.shape == (7, 20)

@parametrize('adaptive_loss_weight', (True, False))
def test_disco_rl(adaptive_loss_weight):
    from disco_rl_pytorch.disco_rl import (
        SharedMetaEmbed,
        MetaNetwork,
        MetaRNN,
        MetaValue,
        Policy,
        Population,
        Adam,
        DiscoRL
    )

    model = Policy(dim = 32, dim_state = 8, num_actions = 4, depth = 2)
    population = Population(model)

    embedder = SharedMetaEmbed(dim = 32, num_actions = 4, dim_abstract_observation = 32, dim_abstract_action = 32)
    meta_rnn = MetaRNN(dim = 32)
    meta_network = MetaNetwork(
        dim = 32,
        num_actions = 4,
        dim_abstract_action = 32,
        dim_abstract_observation = 32,
        adaptive_loss_weight = adaptive_loss_weight
    )
    population_optimizer = Adam()
    value_network = MetaValue(dim = 32, dim_state = 8, depth = 4)

    disco_rl = DiscoRL(
        policy = population,
        policy_optimizer = population_optimizer,
        shared_meta_embed = embedder,
        meta_rnn = meta_rnn,
        meta_network = meta_network,
        meta_value_network = value_network,
        update_steps = 10
    )

    states = torch.randn(7, 20, 8)
    actions = torch.randint(0, 4, (7, 20))
    rewards = torch.randn(7, 20)
    terminated = torch.zeros(7, 20).bool()

    output = disco_rl(states, actions, rewards, terminated)

    assert output.target_action_logits.shape == (7, 20, 4)
    assert output.values.shape == (7, 20)

    if adaptive_loss_weight:
        assert output.loss_weights.shape == (7, 20, 3)
    else:
        assert output.loss_weights is None
