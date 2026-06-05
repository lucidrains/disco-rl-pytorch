
def test_disco_rl():
    import torch
    from disco_rl_pytorch.disco_rl import Policy

    model = Policy(dim = 32, dim_state = 8, num_actions = 4, depth = 2)

    states = torch.randn(7, 8)

    actions, action_logits, encoded_observations, encoded_actions = model(states, sample = True)

    assert actions.shape == (7,)
