'''Quick unit tests for utils.export/read_dataframe_to_csv and configure_logger.'''

import logging

import pandas as pd
import pytest

import utils


pytestmark = pytest.mark.quick


def test_export_then_read_roundtrips_dataframe(tmp_path):
    """export_dataframe_to_csv -> read_dataframe_to_csv preserves columns and values."""
    df = pd.DataFrame({'a': [1, 2, 3], 'b': ['x', 'y', 'z']})
    out_path = tmp_path / 'sub' / 'out.csv'
    logger = logging.getLogger('test_export_then_read')

    utils.export_dataframe_to_csv(df, str(out_path), logger)

    assert out_path.exists()
    loaded = utils.read_dataframe_to_csv(str(out_path), logger)
    assert list(loaded['a']) == [1, 2, 3]
    assert list(loaded['b']) == ['x', 'y', 'z']


def test_export_creates_missing_parent_directory(tmp_path):
    """export_dataframe_to_csv creates missing parent directories."""
    df = pd.DataFrame({'x': [42]})
    nested = tmp_path / 'deep' / 'nested' / 'tree' / 'file.csv'
    utils.export_dataframe_to_csv(df, str(nested), logging.getLogger('t'))
    assert nested.exists()


def test_configure_logger_returns_named_logger_with_handler(tmp_path, monkeypatch):
    """configure_logger returns a DEBUG-level logger with a FileHandler attached."""
    monkeypatch.setattr(utils, 'log_dir', str(tmp_path))
    logger = utils.configure_logger('quick_test_logger')

    assert logger.name == 'quick_test_loggerLogger'
    assert logger.level == logging.DEBUG
    assert any(isinstance(h, logging.FileHandler) for h in logger.handlers)
    assert (tmp_path / 'quick_test_logger.log').exists()


def test_read_dataframe_raises_on_missing_file(tmp_path):
    """read_dataframe_to_csv raises FileNotFoundError on a missing path."""
    logger = logging.getLogger('t')
    with pytest.raises(FileNotFoundError):
        utils.read_dataframe_to_csv(str(tmp_path / 'does_not_exist.csv'), logger)
