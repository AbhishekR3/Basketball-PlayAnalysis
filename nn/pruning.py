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
    Prune attention heads based on their importance scores using structured L1 norm

    Parameters:
    [nn.Module] model - The model containing MultiHeadAttention layers to prune
    [float] prune_amount - Amount of heads to prune (default: 0.2 = 20%)

    Returns:
    [nn.Module] model - The pruned model
    """
    try:
        # Verify if the model has the MultiHeadAttention layer
        if not hasattr(model, 'attention') or not isinstance(model.attention, MultiHeadAttention):
            logger.warning("Model does not have a compatible MultiHeadAttention layer for pruning")
            return model

        # Get the attention module
        attention_module = model.attention

        # Apply L1 structured pruning to head_importance
        prune.ln_structured(
            attention_module,
            name='head_importance',
            amount=prune_amount,
            n=1,  # L1 norm
            dim=0  # Prune along dimension 0 (head dimension)
        )

        # Log pruning information
        # Count non-zero rows in head_importance to determine remaining heads
        remaining_heads = torch.sum(torch.any(attention_module.head_importance != 0, dim=1)).item()
        total_heads = attention_module.num_heads
        pruned_heads = total_heads - remaining_heads

        logger.info(f"Pruned {pruned_heads}/{total_heads} attention heads ({pruned_heads/total_heads:.1%})")

        # Make pruning permanent (optional for inference performance)
        prune.remove(attention_module, 'head_importance')

        return model

    except Exception as e:
        logger.error(f"Error in prune_attention_heads: {e}")
        logger.error(f"Pruning skipped, returning original model")
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
