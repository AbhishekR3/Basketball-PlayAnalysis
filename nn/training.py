'''
nn/training.py

Training and evaluation loops for the BasketballLSTM classifier. Reports
loss, accuracy, precision, recall and F1; supports early stopping and writes
training history / test metrics to MODEL_DIR.
'''


#%% Import libraries

import os
import time
from collections import defaultdict

import pandas as pd
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm

import config
from utils import export_dataframe_to_csv
from ._logger import logger
from .plots import plot_confusion_matrix


#%% Training

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device='cpu', early_stopping_patience=3):
    """
    Objective:
    Train the LSTM model with validation and early stopping

    Parameters:
    [nn.Module] model - The LSTM model to train
    [DataLoader] train_loader - DataLoader for training data
    [DataLoader] val_loader - DataLoader for validation data
    [nn.Module] criterion - Loss function
    [optim.Optimizer] optimizer - Optimizer
    [int] num_epochs - Number of epochs to train
    [string] device - Device to train on ('cpu', 'cuda', or 'mps')
    [int] early_stopping_patience - Number of epochs to wait for improvement

    Returns:
    [dict] history - Training history
    """
    try:
        # Move model to device
        model = model.to(device)

        # Initialize history
        history = defaultdict(list)

        # Initialize early stopping variables
        best_val_loss = float('inf')
        patience_counter = 0

        # Start time
        start_time = time.time()

        # Loop over epochs
        for epoch in range(num_epochs):
            epoch_start_time = time.time()

            # Training
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            all_train_preds = []
            all_train_labels = []

            for _, (features, labels, _) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Training')):
                # Move data to device
                features, labels = features.to(device), labels.to(device)

                # Zero the gradients
                optimizer.zero_grad()

                # Forward pass
                outputs, _ = model(features)

                # Calculate loss
                loss = criterion(outputs, labels)

                # Backward pass and optimize
                loss.backward()
                optimizer.step()

                # Update statistics
                train_loss += loss.item()
                predicted = (outputs >= 0.5).float()
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()

                # Store predictions and labels for metrics calculation
                all_train_preds.extend(predicted.cpu().numpy())
                all_train_labels.extend(labels.cpu().numpy())

            # Calculate average training loss and accuracy
            train_loss /= len(train_loader)
            train_acc = 100 * train_correct / train_total if train_total > 0 else 0

            # Calculate training precision, recall, and F1
            train_precision = precision_score(all_train_labels, all_train_preds, zero_division=0)
            train_recall = recall_score(all_train_labels, all_train_preds, zero_division=0)
            train_f1 = f1_score(all_train_labels, all_train_preds, zero_division=0)

            # Validation
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            all_val_preds = []
            all_val_labels = []

            with torch.no_grad():
                for features, labels, lengths in tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Validation'):
                    # Move data to device
                    features, labels = features.to(device), labels.to(device)

                    # Forward pass
                    outputs, _ = model(features)

                    # Calculate loss
                    loss = criterion(outputs, labels)

                    # Update statistics
                    val_loss += loss.item()
                    predicted = (outputs >= 0.5).float()
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

                    # Store predictions and labels for metrics calculation
                    all_val_preds.extend(predicted.cpu().numpy())
                    all_val_labels.extend(labels.cpu().numpy())

            # Calculate average validation loss and accuracy
            val_loss /= len(val_loader)
            val_acc = 100 * val_correct / val_total if val_total > 0 else 0

            # Calculate validation precision, recall, and F1
            val_precision = precision_score(all_val_labels, all_val_preds, zero_division=0)
            val_recall = recall_score(all_val_labels, all_val_preds, zero_division=0)
            val_f1 = f1_score(all_val_labels, all_val_preds, zero_division=0)

            # Calculate epoch time
            epoch_time = time.time() - epoch_start_time

            # Update history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            history['train_precision'].append(train_precision)
            history['train_recall'].append(train_recall)
            history['train_f1'].append(train_f1)
            history['val_precision'].append(val_precision)
            history['val_recall'].append(val_recall)
            history['val_f1'].append(val_f1)
            history['epoch_time'].append(epoch_time)

            # Print statistics
            logger.info(f'Epoch {epoch+1}/{num_epochs} - '
                        f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                        f'Train Precision: {train_precision:.4f}, Train Recall: {train_recall:.4f}, Train F1: {train_f1:.4f}, '
                        f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, '
                        f'Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}, '
                        f'Time: {epoch_time:.2f}s')

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save the best model
                try:
                    best_model_path = os.path.join(config.MODEL_DIR, 'best_model.pt')
                except TypeError:
                    best_model_path = 'best_model.pt'
                torch.save(model.state_dict(), best_model_path)
                logger.info(f"New best model saved with val_loss: {val_loss:.4f}")
            else:
                patience_counter += 1
                logger.info(f"Early stopping patience: {patience_counter}/{early_stopping_patience}")
                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break

        # Calculate total training time
        total_time = time.time() - start_time
        history['total_time'] = total_time
        logger.info(f"Training completed in {total_time:.2f} seconds")

        # Export training history to CSV
        history_df = pd.DataFrame(history)
        history_path = os.path.join(config.MODEL_DIR, 'training_history.csv')
        export_dataframe_to_csv(history_df, history_path, logger)

        return history

    except Exception as e:
        logger.error(f"Error in train_model: {e}")
        raise


#%% Evaluation

def evaluate_model(model, test_loader, criterion, device='cpu'):
    """
    Objective:
    Evaluate the LSTM model on the test set

    Parameters:
    [nn.Module] model - The trained LSTM model
    [DataLoader] test_loader - DataLoader for test data
    [nn.Module] criterion - Loss function
    [string] device - Device to evaluate on ('cpu', 'cuda', or 'mps')

    Returns:
    [dict] metrics - Evaluation metrics including loss, accuracy, precision, recall, and F1 score
    """
    try:
        # Move model to device
        model = model.to(device)

        # Set model to evaluation mode
        model.eval()

        test_loss = 0.0
        test_correct = 0
        test_total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for features, labels, _ in tqdm(test_loader, desc='Testing'):
                # Move data to device
                features, labels = features.to(device), labels.to(device)

                # Forward pass
                outputs, _ = model(features)

                # Calculate loss
                loss = criterion(outputs, labels)

                # Update statistics
                test_loss += loss.item()
                predicted = (outputs >= 0.5).float()
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()

                # Store predictions and labels for metrics calculation
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calculate average test loss and accuracy
        test_loss /= len(test_loader)
        test_acc = 100 * test_correct / test_total if test_total > 0 else 0

        # Calculate precision, recall, and F1 score
        test_precision = precision_score(all_labels, all_preds, zero_division=0)
        test_recall = recall_score(all_labels, all_preds, zero_division=0)
        test_f1 = f1_score(all_labels, all_preds, zero_division=0)

        # Calculate confusion matrix
        conf_matrix = confusion_matrix(all_labels, all_preds)

        # Store metrics in a dictionary
        metrics = {
            'loss': test_loss,
            'accuracy': test_acc,
            'precision': test_precision,
            'recall': test_recall,
            'f1_score': test_f1,
            'confusion_matrix': conf_matrix
        }

        # Log metrics
        logger.info(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%, '
                    f'Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}, Test F1: {test_f1:.4f}')

        # Export metrics to CSV
        metrics_df = pd.DataFrame([metrics])
        metrics_path = os.path.join(config.MODEL_DIR, 'test_metrics.csv')
        export_dataframe_to_csv(metrics_df, metrics_path, logger)

        # Plot and save confusion matrix
        plot_confusion_matrix(conf_matrix, os.path.join(config.MODEL_DIR, 'confusion_matrix.png'))

        return metrics

    except Exception as e:
        logger.error(f"Error in evaluate_model: {e}")
        raise
