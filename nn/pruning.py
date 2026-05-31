'''
nn/pruning.py

Structured magnitude-based pruning of MultiHeadAttention heads, plus
post-pruning evaluation and inference-time comparison utilities.
'''


#%% Import libraries

import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from tqdm import tqdm

import config
from utils import export_dataframe_to_csv
from ._logger import logger
from .model import MultiHeadAttention
from .training import evaluate_model


#%% Prune attention heads

def prune_attention_heads(model, prune_amount=0.2):
    """
    Objective:
    Structurally prune the least-important attention heads and PHYSICALLY rebuild
    the Q/K/V/output projections at the reduced head count, so the parameter and
    FLOP reduction is real (materialized) rather than just a zero-mask.

    Importance per head combines the learned head_importance scores with the L2
    norm of that head's Q/K/V projection slices (magnitude-based). The lowest-
    scoring heads are dropped and the projection matrices are sliced down to the
    surviving heads. Any pre-existing prune mask is baked in first.

    Parameters:
    [nn.Module] model - The model containing a MultiHeadAttention layer to prune
    [float] prune_amount - Fraction of heads to prune (default: 0.2 = 20%)

    Returns:
    [nn.Module] model - The pruned model with materially fewer parameters
    """
    try:
        # Verify if the model has the MultiHeadAttention layer
        if not hasattr(model, 'attention') or not isinstance(model.attention, MultiHeadAttention):
            logger.warning("Model does not have a compatible MultiHeadAttention layer for pruning")
            return model

        attention = model.attention
        num_heads = attention.num_heads
        head_dim = attention.head_dim

        # Number of heads to keep (always keep at least one)
        n_prune = int(round(num_heads * prune_amount))
        keep_heads = max(1, num_heads - n_prune)
        if keep_heads >= num_heads:
            logger.info("Prune amount rounds to 0 heads; model left unchanged")
            return model

        # Bake any pre-existing head_importance mask into the parameter, then drop
        # the reparametrization so we operate on plain weights below.
        if prune.is_pruned(attention):
            prune.remove(attention, 'head_importance')

        # Per-head importance: learned score * (||Q_h|| + ||K_h|| + ||V_h||)
        with torch.no_grad():
            learned = attention.head_importance.detach().abs().view(-1)  # (num_heads,)
            q = attention.query.weight.detach().view(num_heads, head_dim, -1)
            k = attention.key.weight.detach().view(num_heads, head_dim, -1)
            v = attention.value.weight.detach().view(num_heads, head_dim, -1)
            weight_norm = q.norm(dim=(1, 2)) + k.norm(dim=(1, 2)) + v.norm(dim=(1, 2))
            score = learned * weight_norm

            # Indices of the heads to keep, in ascending order so the head layout
            # stays contiguous after slicing.
            keep_idx = torch.sort(torch.topk(score, keep_heads).indices).values

            # Expand kept-head indices into the row indices of the projection
            # weights (each head owns head_dim contiguous output rows).
            row_idx = torch.cat([
                torch.arange(h * head_dim, (h + 1) * head_dim) for h in keep_idx
            ])

            new_dim = keep_heads * head_dim
            in_dim = attention.query.in_features  # unchanged (== attention_dim)

            # Rebuild Q/K/V with fewer output rows (the dropped heads' rows removed)
            for name in ('query', 'key', 'value'):
                old = getattr(attention, name)
                new = nn.Linear(in_dim, new_dim)
                new.weight = nn.Parameter(old.weight.data[row_idx].clone())
                new.bias = nn.Parameter(old.bias.data[row_idx].clone())
                setattr(attention, name, new)

            # Rebuild output projection: input shrinks to new_dim, output stays at
            # attention_dim so the residual connection still lines up.
            old_out = attention.output_projection
            new_out = nn.Linear(new_dim, attention.hidden_dim)
            new_out.weight = nn.Parameter(old_out.weight.data[:, row_idx].clone())
            new_out.bias = nn.Parameter(old_out.bias.data.clone())
            attention.output_projection = new_out

            # Shrink the head-importance vector to the surviving heads
            attention.head_importance = nn.Parameter(
                attention.head_importance.data[keep_idx].clone()
            )

        # Update bookkeeping the forward pass relies on. head_dim (and thus scale)
        # is unchanged; attention.hidden_dim is the flattened multi-head width used
        # when reshaping weighted values, so it shrinks to new_dim.
        attention.num_heads = keep_heads
        attention.hidden_dim = new_dim

        logger.info(
            f"Pruned {num_heads - keep_heads}/{num_heads} attention heads "
            f"({(num_heads - keep_heads)/num_heads:.1%}); attention width "
            f"{num_heads * head_dim} -> {new_dim}"
        )

        return model

    except Exception as e:
        logger.error(f"Error in prune_attention_heads: {e}")
        logger.error("Pruning skipped, returning original model")
        return model


#%% Evaluate pruned model

def evaluate_pruned_model(model, test_loader, criterion, device='cpu'):
    """
    Objective:
    Evaluate a pruned model on the test set

    Parameters:
    [nn.Module] model - The pruned model to evaluate
    [DataLoader] test_loader - DataLoader for test data
    [nn.Module] criterion - Loss function
    [string] device - Device to evaluate on ('cpu', 'cuda', or 'mps')

    Returns:
    [dict] metrics - Evaluation metrics for the pruned model
    """
    try:
        # Log the evaluation start
        logger.info("Evaluating pruned model on test set...")

        # First, evaluate the model using the standard evaluation function
        metrics = evaluate_model(model, test_loader, criterion, device)

        # Calculate model size before and after pruning
        total_params = sum(p.numel() for p in model.parameters())
        nonzero_params = sum(p.nonzero().size(0) for p in model.parameters())
        sparsity = 1.0 - nonzero_params / total_params

        logger.info(f"Model parameters: {total_params}, Non-zero parameters: {nonzero_params}")
        logger.info(f"Model sparsity: {sparsity:.2%}")

        # Add pruning-specific metrics
        metrics['sparsity'] = sparsity
        metrics['total_params'] = total_params
        metrics['nonzero_params'] = nonzero_params

        return metrics

    except Exception as e:
        logger.error(f"Error in evaluate_pruned_model: {e}")
        raise


#%% Pruning comparison / visualization

def analyze_and_visualize_pruning(original_model, pruned_model, test_loader, device='cpu'):
    """
    Objective:
    Analyze and visualize the effect of pruning on model performance and size

    Parameters:
    [nn.Module] original_model - The original model before pruning
    [nn.Module] pruned_model - The pruned model
    [DataLoader] test_loader - DataLoader for test data
    [string] device - Device to use ('cpu', 'cuda', or 'mps')
    """
    try:
        # Set models to evaluation mode
        original_model.eval()
        pruned_model.eval()

        # Move models to device
        original_model = original_model.to(device)
        pruned_model = pruned_model.to(device)

        # Collect inference times
        original_times = []
        pruned_times = []

        # Measure inference time on test set
        with torch.no_grad():
            # Original model
            for features, _, _ in tqdm(test_loader, desc='Original Model Inference'):
                features = features.to(device)
                start_time = time.time()
                original_model(features)
                original_times.append(time.time() - start_time)

            # Pruned model
            for features, _, _ in tqdm(test_loader, desc='Pruned Model Inference'):
                features = features.to(device)
                start_time = time.time()
                pruned_model(features)
                pruned_times.append(time.time() - start_time)

        # Calculate average inference times
        avg_original_time = np.mean(original_times) * 1000  # Convert to ms
        avg_pruned_time = np.mean(pruned_times) * 1000  # Convert to ms
        speed_improvement = (avg_original_time - avg_pruned_time) / avg_original_time * 100

        # Calculate model sizes
        original_num_params = sum(p.numel() for p in original_model.parameters())
        nonzero_pruned_params = sum(p.nonzero().size(0) for p in pruned_model.parameters())
        size_reduction = (original_num_params - nonzero_pruned_params) / original_num_params * 100

        # Log the results
        logger.info(f"Original model parameters: {original_num_params}")
        logger.info(f"Pruned model non-zero parameters: {nonzero_pruned_params}")
        logger.info(f"Model size reduction: {size_reduction:.2f}%")
        logger.info(f"Original model inference time: {avg_original_time:.2f} ms/batch")
        logger.info(f"Pruned model inference time: {avg_pruned_time:.2f} ms/batch")
        logger.info(f"Speed improvement: {speed_improvement:.2f}%")

        # Visualization
        # Create a bar chart comparing original and pruned models
        plt.figure(figsize=(12, 6))

        # Plot inference time comparison
        plt.subplot(1, 2, 1)
        plt.bar(['Original', 'Pruned'], [avg_original_time, avg_pruned_time])
        plt.ylabel('Inference Time (ms/batch)')
        plt.title('Inference Time Comparison')

        # Plot model size comparison
        plt.subplot(1, 2, 2)
        plt.bar(['Original', 'Pruned'], [original_num_params, nonzero_pruned_params])
        plt.ylabel('Number of Parameters')
        plt.title('Model Size Comparison')

        plt.tight_layout()

        # Save the comparison plot
        comparison_path = os.path.join(config.MODEL_DIR, 'pruning_comparison.png')
        plt.savefig(comparison_path)
        logger.info(f"Pruning comparison plot saved to {comparison_path}")

        # Create a dictionary of pruning metrics
        pruning_metrics = {
            'original_params': original_num_params,
            'pruned_nonzero_params': nonzero_pruned_params,
            'size_reduction_percent': size_reduction,
            'original_inference_time_ms': avg_original_time,
            'pruned_inference_time_ms': avg_pruned_time,
            'speed_improvement_percent': speed_improvement
        }

        # Export pruning metrics to CSV
        pruning_metrics_df = pd.DataFrame([pruning_metrics])
        metrics_path = os.path.join(config.MODEL_DIR, 'pruning_metrics.csv')
        export_dataframe_to_csv(pruning_metrics_df, metrics_path, logger)

        return pruning_metrics

    except Exception as e:
        logger.error(f"Error in analyze_and_visualize_pruning: {e}")
        raise
