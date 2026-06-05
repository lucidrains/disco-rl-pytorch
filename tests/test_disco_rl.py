
def test_disco_rl():
    import torch
    from disco_rl_pytorch.disco_rl import Policy, SharedMetaEmbed

    model = Policy(dim = 32, dim_state = 8, num_actions = 4, depth = 2)

    states = torch.randn(7, 20, 8)

    action_logits, encoded_observations, actions, encoded_actions, pred_action_value, pred_next_action = model(states, sample = True)

    assert actions.shape == (7, 20)

    rewards = torch.randn(7, 20)
    terminated = torch.zeros(7, 20).bool()

    embedder = SharedMetaEmbed(dim = 32, num_actions = 4, dim_abstract_observation = 32, dim_abstract_action = 32)

    embed = embedder(actions, rewards, terminated, action_logits, encoded_observations, encoded_actions, pred_action_value)
