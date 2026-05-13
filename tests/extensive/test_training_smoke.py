'''
Smoke test for the LSTM training loop.

Creates a tiny synthetic dataset directory layout, trains BasketballLSTM for
a few epochs, and asserts the loss decreases and the checkpoint round-trips.
'''

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader


pytestmark = pytest.mark.extensive


SEQ_LEN = 20
FEATURE_DIM = 6


def _make_synthetic_csv(path, label, seed):
    """Write a synthetic per-sample CSV: label=1 follows a sinusoid; label=0 is noise."""
    rng = np.random.default_rng(seed)
    if label == 1:
        signal = np.sin(np.linspace(0, 4 * np.pi, SEQ_LEN)).reshape(-1, 1)
        features = signal + 0.1 * rng.standard_normal((SEQ_LEN, FEATURE_DIM))
    else:
        features = rng.standard_normal((SEQ_LEN, FEATURE_DIM))
    df = pd.DataFrame(features, columns=[f'feat_{i}' for i in range(FEATURE_DIM)])
    df['Frame'] = np.arange(SEQ_LEN)
    df.to_csv(path, index=False)


def _build_dataset_dir(root, n_per_class=4):
    """Build a tmp dataset root with train/validation/test x pass/not-pass splits."""
    for split in ('train', 'validation', 'test'):
        for label_name, label in (('pass', 1), ('not-pass', 0)):
            split_dir = root / split / label_name
            split_dir.mkdir(parents=True, exist_ok=True)
            for i in range(n_per_class):
                seed = hash((split, label_name, i)) & 0xFFFF
                _make_synthetic_csv(split_dir / f'sample_{i}.csv', label, seed)


def test_lstm_training_loss_decreases_and_saves_checkpoint(tmp_path, monkeypatch):
    """A few epochs of training reduce loss, and save/load round-trips parameters."""
    monkeypatch.setenv('MODEL_DIR', str(tmp_path / 'models'))
    (tmp_path / 'models').mkdir()

    import importlib  # pylint: disable=import-outside-toplevel
    import config  # pylint: disable=import-outside-toplevel
    importlib.reload(config)
    from nn import dataset as nn_dataset, model as nn_model, io as nn_io  # pylint: disable=import-outside-toplevel

    data_root = tmp_path / 'data'
    _build_dataset_dir(data_root, n_per_class=4)

    train_ds = nn_dataset.BasketballPlayDataset(str(data_root), split='train')

    input_dim = FEATURE_DIM + 1

    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True,
                              collate_fn=nn_dataset.collate_variable_length_sequences)
    # Deterministic eval loader -- shuffle=False so initial/final losses
    # are measured on the same batches and the comparison is meaningful.
    eval_loader = DataLoader(train_ds, batch_size=2, shuffle=False,
                             collate_fn=nn_dataset.collate_variable_length_sequences)

    torch.manual_seed(0)
    model = nn_model.BasketballLSTM(
        input_dim=input_dim, hidden_dim=16, output_dim=1, num_layers=1,
        dropout=0.0, recurrent_dropout=0.0, bidirectional=True, num_heads=4,
    )
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    def _avg_loss():
        """Mean BCE loss across the deterministic eval loader."""
        model.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for batch_feats, batch_labels, _ in eval_loader:
                out, _ = model(batch_feats)
                total += criterion(out, batch_labels).item()
                count += 1
        return total / max(count, 1)

    initial_loss = _avg_loss()

    model.train()
    for _ in range(10):
        for feats, labels, _lengths in train_loader:
            optimizer.zero_grad()
            out, _ = model(feats)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

    final_loss = _avg_loss()

    assert final_loss < initial_loss, (
        f'expected loss to decrease, but got initial={initial_loss:.4f} final={final_loss:.4f}'
    )

    # Capture deterministic output for the save/load roundtrip assertion.
    model.eval()
    with torch.no_grad():
        feats, labels, _ = next(iter(eval_loader))
        out, _ = model(feats)

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
    """prune_attention_heads with prune_amount=0.5 zeros the lowest-importance heads."""
    from nn import model as nn_model, pruning as nn_pruning  # pylint: disable=import-outside-toplevel

    torch.manual_seed(0)
    model = nn_model.BasketballLSTM(
        input_dim=4, hidden_dim=8, output_dim=1, num_layers=1,
        dropout=0.0, recurrent_dropout=0.0, bidirectional=True, num_heads=4,
    )

    with torch.no_grad():
        model.attention.head_importance.data = torch.tensor(
            [[1.0], [0.1], [0.9], [0.05]]
        )

    pruned = nn_pruning.prune_attention_heads(model, prune_amount=0.5)
    nonzero = (pruned.attention.head_importance.data != 0).sum().item()
    assert nonzero == 2
