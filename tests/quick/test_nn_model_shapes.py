'''Quick shape tests for BasketballLSTM and MultiHeadAttention.'''

import pytest
import torch


pytestmark = pytest.mark.quick


@pytest.fixture(scope='module')
def model_mod():
    """Import the nn.model module once for all tests in this module."""
    from nn import model as _m  # pylint: disable=import-outside-toplevel
    return _m


def test_basketball_lstm_forward_shape(model_mod):
    """BasketballLSTM.forward returns (batch, 1) output and correct hidden shape."""
    torch.manual_seed(0)
    net = model_mod.BasketballLSTM(
        input_dim=8,
        hidden_dim=16,
        output_dim=1,
        num_layers=1,
        dropout=0.0,
        recurrent_dropout=0.0,
        bidirectional=True,
        num_heads=4,
    )
    net.eval()
    x = torch.randn(2, 10, 8)
    out, hidden = net(x)
    assert out.shape == (2, 1)
    h, c = hidden
    assert h.shape == (2, 2, 16)
    assert c.shape == (2, 2, 16)


def test_basketball_lstm_outputs_in_sigmoid_range(model_mod):
    """BasketballLSTM outputs values constrained to [0, 1] by the final sigmoid."""
    torch.manual_seed(0)
    net = model_mod.BasketballLSTM(input_dim=4, hidden_dim=8, num_layers=1,
                                   dropout=0.0, recurrent_dropout=0.0,
                                   num_heads=2, bidirectional=False)
    net.eval()
    out, _ = net(torch.randn(3, 5, 4))
    assert torch.all(out >= 0.0) and torch.all(out <= 1.0)


def test_unidirectional_halves_hidden_state(model_mod):
    """Unidirectional BasketballLSTM exposes num_directions == 1."""
    net = model_mod.BasketballLSTM(input_dim=4, hidden_dim=8, num_layers=1,
                                   dropout=0.0, recurrent_dropout=0.0,
                                   num_heads=2, bidirectional=False)
    assert net.num_directions == 1


def test_multihead_attention_requires_divisible_hidden_dim(model_mod):
    """MultiHeadAttention asserts hidden_dim is divisible by num_heads."""
    with pytest.raises(AssertionError):
        model_mod.MultiHeadAttention(hidden_dim=7, num_heads=4)


def test_multihead_attention_forward_returns_correct_shapes(model_mod):
    """MultiHeadAttention.forward returns attended output + per-head attention weights."""
    torch.manual_seed(0)
    attn = model_mod.MultiHeadAttention(hidden_dim=16, num_heads=4)
    x = torch.randn(2, 5, 16)
    out, weights = attn(x)
    assert out.shape == (2, 5, 16)
    assert weights.shape == (2, 4, 5, 5)
