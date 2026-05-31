"""Behavioral tests for the nn package's pruning and LR scheduling (P0-D)."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import nn as bnn


def _tiny_loader(n=8, seq=4, feat=10):
    """A minimal loader yielding (padded_features, labels, lengths) batches via
    the project's own collate function."""
    data = [(torch.randn(seq, feat), torch.FloatTensor([i % 2])) for i in range(n)]
    return DataLoader(
        data, batch_size=4, collate_fn=bnn.collate_variable_length_sequences
    )


def test_pruning_materially_reduces_params_and_runs():
    """P0-D: pruning must drop the materialized parameter count (not just zero a
    mask) and the model must still run a forward pass at the reduced head count."""
    model = bnn.BasketballLSTM(input_dim=10, hidden_dim=16, num_heads=8)
    params_before = sum(p.numel() for p in model.parameters())
    heads_before = model.attention.num_heads

    pruned = bnn.prune_attention_heads(model, prune_amount=0.5)

    params_after = sum(p.numel() for p in pruned.parameters())
    assert params_after < params_before, "pruning did not materially reduce params"
    assert pruned.attention.num_heads < heads_before

    # Forward pass still works at the reduced head count
    out, _ = pruned(torch.randn(2, 5, 10))
    assert out.shape == (2, 1)


def test_pruned_head_count_matches_prune_amount():
    model = bnn.BasketballLSTM(input_dim=10, hidden_dim=16, num_heads=8)
    bnn.prune_attention_heads(model, prune_amount=0.25)
    assert model.attention.num_heads == 6  # 8 - round(8*0.25)


def test_scheduler_lr_changes_across_epochs(tmp_path, monkeypatch):
    """train_model must step the scheduler each epoch so the LR changes."""
    import config
    # train_model writes history/checkpoints under config.MODEL_DIR; redirect to tmp
    monkeypatch.setattr(config, 'MODEL_DIR', str(tmp_path))

    model = bnn.BasketballLSTM(input_dim=10, hidden_dim=16, num_heads=8)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3)
    loader = _tiny_loader()

    lr_before = optimizer.param_groups[0]['lr']
    history = bnn.train_model(
        model, loader, loader, nn.BCELoss(), optimizer,
        num_epochs=2, device='cpu', early_stopping_patience=5, scheduler=scheduler,
    )
    lr_after = optimizer.param_groups[0]['lr']

    assert lr_after != lr_before
    assert len(history['learning_rate']) == 2
    assert history['learning_rate'][0] != history['learning_rate'][1]
