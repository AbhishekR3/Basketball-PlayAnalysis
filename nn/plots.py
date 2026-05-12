'''
nn/plots.py

Plotting helpers: training-history curves and the test-set confusion matrix.
'''


#%% Import libraries

import os

import matplotlib.pyplot as plt
import seaborn as sns

from ._logger import logger


#%% Training history

def plot_training_history(history, save_path=None):
    """
    Objective:
    Plot the training history

    Parameters:
    [dict] history - Training history with keys: train_loss, val_loss, train_acc, val_acc, etc.
    [string] save_path - Path to save the plot (default: None)
    """
    try:
        plt.figure(figsize=(15, 10))

        # Plot loss
        plt.subplot(2, 2, 1)
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()

        # Plot accuracy
        plt.subplot(2, 2, 2)
        plt.plot(history['train_acc'], label='Train Accuracy')
        plt.plot(history['val_acc'], label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('Training and Validation Accuracy')
        plt.legend()

        # Plot precision
        plt.subplot(2, 2, 3)
        plt.plot(history['train_precision'], label='Train Precision')
        plt.plot(history['val_precision'], label='Validation Precision')
        plt.xlabel('Epoch')
        plt.ylabel('Precision')
        plt.title('Training and Validation Precision')
        plt.legend()

        # Plot recall
        plt.subplot(2, 2, 4)
        plt.plot(history['train_recall'], label='Train Recall')
        plt.plot(history['val_recall'], label='Validation Recall')
        plt.xlabel('Epoch')
        plt.ylabel('Recall')
        plt.title('Training and Validation Recall')
        plt.legend()

        plt.tight_layout()

        # Save the plot if a path is provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            logger.info(f"Training history plot saved to {save_path}")

        plt.show()

    except Exception as e:
        logger.error(f"Error in plot_training_history: {e}")
        raise


#%% Confusion matrix

def plot_confusion_matrix(cm, save_path=None):
    """
    Objective:
    Plot a confusion matrix

    Parameters:
    [numpy.ndarray] cm - Confusion matrix
    [string] save_path - Path to save the plot (default: None)
    """
    try:
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['No Pass', 'Pass'],
                    yticklabels=['No Pass', 'Pass'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')

        # Save the plot if a path is provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            logger.info(f"Confusion matrix plot saved to {save_path}")

        plt.show()

    except Exception as e:
        logger.error(f"Error in plot_confusion_matrix: {e}")
        raise
