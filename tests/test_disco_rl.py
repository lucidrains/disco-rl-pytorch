
import torch
import torch.nn.functional as F

def test_disco_rl():
    from disco_rl_pytorch.disco_rl import Policy, SharedMetaEmbed, MetaNetwork, MetaRNN, forward_kl

    model = Policy(dim = 32, dim_state = 8, num_actions = 4, depth = 2)

    states = torch.randn(7, 20, 8)

    action_logits, encoded_observations, actions, encoded_actions, pred_action_value, pred_next_action = model(states, sample = True)

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

    meta_network = MetaNetwork(dim = 32, num_actions = 4, dim_abstract_action = 32, dim_abstract_observation = 32)

    target_action_logits, target_encoded_observations, target_encoded_actions = meta_network(embeds, condition = condition)

    loss = (
        forward_kl(action_logits, target_action_logits) +
        forward_kl(encoded_observations, target_encoded_observations) +
        forward_kl(encoded_actions, target_encoded_actions)
    )

    assert loss.numel() == 1
