'''
Smoke test for the LSTM training loop.

Creates a tiny synthetic dataset directory layout, trains BasketballLSTM for
2 epochs, and asserts the loss decreases and the checkpoint is written.
'''

import os

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader


pytestmark = pytest.mark.extensive


SEQ_LEN = 20
FEATURE_DIM = 6


def _make_synthetic_csv(path, label, seed):
    rng = np.random.default_rng(seed)
    # Pass samples (label=1) follow a sinusoidal pattern; not-pass samples are noise.
    if label == 1:
        signal = np.sin(np.linspace(0, 4 * np.pi, SEQ_LEN)).reshape(-1, 1)
        features = signal + 0.1 * rng.standard_normal((SEQ_LEN, FEATURE_DIM))
    else:
        features = rng.standard_normal((SEQ_LEN, FEATURE_DIM))
    df = pd.DataFrame(features, columns=[f'feat_{i}' for i in range(FEATURE_DIM)])
    df['Frame'] = np.arange(SEQ_LEN)
    df.to_csv(path, index=False)


def _build_dataset_dir(root, n_per_class=4):
    for split in ('train', 'validation', 'test'):
        for label_name, label in (('pass', 1), ('not-pass', 0)):
            split_dir = root / split / label_name
            split_dir.mkdir(parents=True, exist_ok=True)
            for i in range(n_per_class):
                seed = hash((split, label_name, i)) & 0xFFFF
                _make_synthetic_csv(split_dir / f'sample_{i}.csv', label, seed)


def test_lstm_training_loss_decreases_and_saves_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setenv('MODEL_DIR', str(tmp_path / 'models'))
    (tmp_path / 'models').mkdir()

    import importlib
    import config
    importlib.reload(config)
    from nn import dataset as nn_dataset, model as nn_model, io as nn_io

    data_root = tmp_path / 'data'
    _build_dataset_dir(data_root, n_per_class=4)

    train_ds = nn_dataset.BasketballPlayDataset(str(data_root), split='train')
    val_ds = nn_dataset.BasketballPlayDataset(str(data_root), split='validation')

    # Each CSV has FEATURE_DIM + 1 (Frame) columns -> input_dim = FEATURE_DIM + 1
    input_dim = FEATURE_DIM + 1

    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True,
                              collate_fn=nn_dataset.collate_variable_length_sequences)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False,
                            collate_fn=nn_dataset.collate_variable_length_sequences)

    torch.manual_seed(0)
    model = nn_model.BasketballLSTM(
        input_dim=input_dim, hidden_dim=16, output_dim=1, num_layers=1,
        dropout=0.0, recurrent_dropout=0.0, bidirectional=True, num_heads=4,
    )
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    # Measure loss before training
    model.eval()
    with torch.no_grad():
        feats, labels, _ = next(iter(train_loader))
        out, _ = model(feats)
        initial_loss = criterion(out, labels).item()

    # Train 5 epochs
    model.train()
    for _ in range(5):
        for feats, labels, _lengths in train_loader:
            optimizer.zero_grad()
            out, _ = model(feats)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

    # Measure loss after training (on same batch)
    model.eval()
    with torch.no_grad():
        feats, labels, _ = next(iter(train_loader))
        out, _ = model(feats)
        final_loss = criterion(out, labels).item()

    assert final_loss < initial_loss, (
        f'expected loss to decrease, but got initial={initial_loss:.4f} final={final_loss:.4f}'
    )

    # Save/load roundtrip
    ckpt = tmp_path / 'models' / 'basketball_lstm_model.pt'
    nn_io.save_model(model, str(ckpt))
    assert ckpt.exists() and ckpt.stat().st_size > 0

    fresh = nn_model.BasketballLSTM(
        input_dim=input_dim, hidden_dim=16, output_dim=1, num_layers=1,
        dropout=0.0, recurrent_dropout=0.0, bidirectional=True, num_heads=4,
    )
    nn_io.load_model(fresh, str(ckpt), device='cpu')
    fresh.eval()
    with torch.no_grad():
        out_fresh, _ = fresh(feats)
    torch.testing.assert_close(out, out_fresh)


def test_prune_attention_heads_zeros_out_lowest_importance():
    from nn import model as nn_model, pruning as nn_pruning

    torch.manual_seed(0)
    model = nn_model.BasketballLSTM(
        input_dim=4, hidden_dim=8, output_dim=1, num_layers=1,
        dropout=0.0, recurrent_dropout=0.0, bidirectional=True, num_heads=4,
    )

    # Set head importance to known values so pruning is deterministic
    with torch.no_grad():
        model.attention.head_importance.data = torch.tensor(
            [[1.0], [0.1], [0.9], [0.05]]
        )

    pruned = nn_pruning.prune_attention_heads(model, prune_amount=0.5)
    nonzero = (pruned.attention.head_importance.data != 0).sum().item()
    assert nonzero == 2  # 4 heads, prune 50% -> 2 remaining
