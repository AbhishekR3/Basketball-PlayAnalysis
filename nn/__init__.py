'''
nn package

Modular split of the original Neural_Network.py:

    _logger     - shared logger instance ('neural_network')
    dataset     - BasketballPlayDataset, ScaleTransform, fit_scaler, collate
    transforms  - time warp / jitter / horizontal flip / Compose
    model       - MultiHeadAttention, BasketballLSTM
    training    - train_model, evaluate_model
    pruning     - prune_attention_heads, evaluate_pruned_model, analyze_and_visualize_pruning
    plots       - plot_training_history, plot_confusion_matrix
    io          - save_model, load_model

Importing this package also seeds torch + numpy from config.SEED so behavior
matches the original module-level setup of Neural_Network.py.
'''


#%% Import libraries

import numpy as np
import torch

import config
from ._logger import logger

# Re-export the public API
from .dataset import (
    BasketballPlayDataset,
    ScaleTransform,
    fit_scaler,
    collate_variable_length_sequences,
)
from .transforms import (
    ComposeTransforms,
    time_warp_transform,
    jitter_transform,
    horizontal_flip_transform,
)
from .model import MultiHeadAttention, BasketballLSTM
from .training import train_model, evaluate_model
from .pruning import (
    prune_attention_heads,
    evaluate_pruned_model,
    analyze_and_visualize_pruning,
)
from .plots import plot_training_history, plot_confusion_matrix
from .io import save_model, load_model


#%% One-time setup

logger.info("Neural Network Processing started")

# Set random seed for reproducibility (same SEED as the original module top).
torch.manual_seed(config.SEED)
np.random.seed(config.SEED)


__all__ = [
    'logger',
    # dataset
    'BasketballPlayDataset', 'ScaleTransform', 'fit_scaler',
    'collate_variable_length_sequences',
    # transforms
    'ComposeTransforms', 'time_warp_transform', 'jitter_transform',
    'horizontal_flip_transform',
    # model
    'MultiHeadAttention', 'BasketballLSTM',
    # training
    'train_model', 'evaluate_model',
    # pruning
    'prune_attention_heads', 'evaluate_pruned_model',
    'analyze_and_visualize_pruning',
    # plots
    'plot_training_history', 'plot_confusion_matrix',
    # io
    'save_model', 'load_model',
]
