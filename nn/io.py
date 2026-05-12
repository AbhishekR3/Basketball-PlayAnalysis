'''
nn/io.py

Model checkpoint helpers.
'''


#%% Import libraries

import os

import torch

from ._logger import logger


#%% Save / Load

def save_model(model, path):
    """
    Objective:
    Save the model to disk

    Parameters:
    [nn.Module] model - The model to save
    [string] path - Path to save the model
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)

        torch.save(model.state_dict(), path)
        logger.info(f"Model saved to {path}")
    except Exception as e:
        logger.error(f"Error in save_model: {e}")
        raise


def load_model(model, path, device='cpu'):
    """
    Objective:
    Load the model from disk

    Parameters:
    [nn.Module] model - The model to load weights into
    [string] path - Path to load the model from
    [string] device - Device to load the model to ('cpu', 'cuda', or 'mps')

    Returns:
    [nn.Module] model - The loaded model
    """
    try:
        model.load_state_dict(torch.load(path, map_location=device))
        model = model.to(device)
        logger.info(f"Model loaded from {path}")
        return model
    except Exception as e:
        logger.error(f"Error in load_model: {e}")
        raise
