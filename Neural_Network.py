'''
Basketball Neural Network
This file is the entrypoint for the basketball play classifier. The actual
implementation lives in the `nn` package; this script wires configuration and
runs the training / pruning / evaluation pipeline.

Key Concepts Implemented:
- LSTM (Long Short-Term Memory) - Sequence modeling capturing temporal dependencies in player movements with recurrent dropout
- Self-Attention - Mechanism for focusing on relevant parts of the sequence regardless of position
- Data Augmentation - Techniques like time warping, jittering, and flipping to increase dataset diversity
- Bi-directional Processing - Forward and backward pass for comprehensive temporal context
- Downsampling - Reducing the number of frames to manage computational load
- Model Pruning - Structured magnitude-based pruning of attention heads to reduce model size
- Metrics Evaluation - Accuracy, Precision, Recall, F1 score, and confusion matrix for model performance assessment
'''


#%% Import libraries

import os

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import config
from nn import (
    BasketballLSTM,
    BasketballPlayDataset,
    ComposeTransforms,
    ScaleTransform,
    analyze_and_visualize_pruning,
    collate_variable_length_sequences,
    fit_scaler,
    horizontal_flip_transform,
    jitter_transform,
    load_model,
    logger,
    plot_training_history,
    prune_attention_heads,
    save_model,
    time_warp_transform,
    train_model,
)


#%% Configure Docker containerization

config.setup_pipeline_dirs(logger=logger)


#%% Main Function

def main():
    """
    Objective:
    Main function to run the basketball play classification with model pruning
    """
    try:
        # Set parameters
        data_path = config.DATA_PATH

        batch_size = config.BATCH_SIZE
        num_epochs = config.NUM_EPOCHS
        learning_rate = config.LEARNING_RATE
        early_stopping_patience = config.EARLY_STOPPING_PATIENCE
        prune_amount = config.PRUNE_AMOUNT

        # Data augmentation parameters
        use_augmentation = config.USE_AUGMENTATION
        time_warp_sigma = config.TIME_WARP_SIGMA
        time_warp_knots = config.TIME_WARP_KNOTS
        jitter_intensity = config.JITTER_INTENSITY
        flip_probability = config.FLIP_PROBABILITY

        # Multi-head attention parameters
        num_heads = config.NUM_HEADS

        # Resolve compute device (CUDA -> MPS -> CPU, or DEVICE override)
        device = config.get_device(logger)

        # Fit scaler on training data
        logger.info("Fitting scaler on training data...")
        scaler = fit_scaler(data_path, split='train')

        # Create different transforms for training and validation/test
        if use_augmentation:
            logger.info("Creating data augmentation transforms...")

            # Find column indices for features that need to be flipped
            try:
                # Load a sample file to get column structure
                sample_file = None
                for split_dir in ['train', 'validation', 'test']:
                    split_path = os.path.join(data_path, split_dir)
                    if os.path.exists(split_path):
                        for subdir in ['pass', 'not-pass']:
                            subdir_path = os.path.join(split_path, subdir)
                            if os.path.exists(subdir_path):
                                files = [f for f in os.listdir(subdir_path) if f.endswith('.csv')]
                                if files:
                                    sample_file = os.path.join(subdir_path, files[0])
                                    break
                        if sample_file:
                            break

                if sample_file:
                    sample_df = pd.read_csv(sample_file)

                    # Get the indices directly by position if they exist
                    pos_x_col = None
                    vel_x_col = None

                    for i, col in enumerate(sample_df.columns):
                        if 'pos_x' in col.lower():
                            pos_x_col = i
                            logger.info(f"Found pos_x column at index {i}: {col}")
                        elif 'vel_x' in col.lower():
                            vel_x_col = i
                            logger.info(f"Found vel_x column at index {i}: {col}")

                    # Fallback to hardcoded indices if not found
                    if pos_x_col is None:
                        pos_x_col = config.POS_X_COL_FALLBACK
                        logger.info(f"Using fallback pos_x_col index: {pos_x_col}")
                    if vel_x_col is None:
                        vel_x_col = config.VEL_X_COL_FALLBACK
                        logger.info(f"Using fallback vel_x_col index: {vel_x_col}")
                else:
                    # No sample file found
                    logger.warning("No sample CSV files found in the dataset directory")
                    pos_x_col = config.POS_X_COL_FALLBACK
                    vel_x_col = config.VEL_X_COL_FALLBACK
                    logger.info(f"Using fallback column indices: pos_x={pos_x_col}, vel_x={vel_x_col}")

            except Exception as e:
                logger.warning(f"Error detecting column indices: {e}")
                pos_x_col = config.POS_X_COL_FALLBACK
                vel_x_col = config.VEL_X_COL_FALLBACK
                logger.info(f"Using fallback column indices after error: pos_x={pos_x_col}, vel_x={vel_x_col}")

            # Create the transforms
            train_transform = ComposeTransforms([
                # Time Warping Transformation
                time_warp_transform(sigma=time_warp_sigma, num_knots=time_warp_knots),

                # Jitter transformation
                jitter_transform(intensity=jitter_intensity),

                # Horizontal Flip Transformation
                horizontal_flip_transform(
                    flip_probability=flip_probability,
                    x_position_col=pos_x_col,
                    x_velocity_col=vel_x_col
                ),

                # Standard scaling (always applied last)
                ScaleTransform(scaler)
            ])

            logger.info("Data transformations applied")

        else:
            train_transform = ScaleTransform(scaler)
            logger.info("Data transformations disabled, using only standard scaling")

        # Validation and test data should not be augmented
        val_test_transform = ScaleTransform(scaler)

        # Create datasets
        train_dataset = BasketballPlayDataset(data_path, split='train', transform=train_transform)
        val_dataset = BasketballPlayDataset(data_path, split='validation', transform=val_test_transform)
        test_dataset = BasketballPlayDataset(data_path, split='test', transform=val_test_transform)

        # Calculate input dimension from an example
        if len(train_dataset) > 0:
            example_features, _ = train_dataset[0]
            input_dim = example_features.shape[1]
        else:
            logger.error("Training dataset is empty. Cannot determine input dimension.")
            raise ValueError("Training dataset is empty")

        # Create data loaders with custom collate function
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_variable_length_sequences
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_variable_length_sequences
        )

        # Initialize model with multi-head attention
        model = BasketballLSTM(
            input_dim=input_dim,
            hidden_dim=config.HIDDEN_DIM,
            output_dim=config.OUTPUT_DIM,
            num_layers=config.NUM_LAYERS,
            dropout=config.DROPOUT,
            recurrent_dropout=config.RECURRENT_DROPOUT,
            bidirectional=config.BIDIRECTIONAL,
            num_heads=num_heads
        )

        # Initialize loss function and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=config.WEIGHT_DECAY)  # L2 regularization

        # Cosine-annealing LR schedule over the full run (replaces the previous
        # fixed LR for all epochs)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

        # Train the model
        logger.info("Starting model training...")
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            num_epochs=num_epochs,
            device=device,
            early_stopping_patience=early_stopping_patience,
            scheduler=scheduler
        )

        # Plot and save training history
        history_plot_path = os.path.join(config.MODEL_DIR, 'training_history.png')
        plot_training_history(history, save_path=history_plot_path)

        # Load the best model
        best_model_path = os.path.join(config.MODEL_DIR, 'best_model.pt')
        if os.path.exists(best_model_path):
            model = load_model(model, best_model_path, device)

        # Save a copy of the original (unpruned) model
        original_model_path = os.path.join(config.MODEL_DIR, 'original_model.pt')
        save_model(model, original_model_path)
        logger.info("Original model saved before pruning")

        # Create a copy of the original model for comparison
        original_model = BasketballLSTM(
            input_dim=input_dim,
            hidden_dim=config.HIDDEN_DIM,
            output_dim=config.OUTPUT_DIM,
            num_layers=config.NUM_LAYERS,
            dropout=config.DROPOUT,
            recurrent_dropout=config.RECURRENT_DROPOUT,
            bidirectional=config.BIDIRECTIONAL,
            num_heads=num_heads
        )

        original_model.load_state_dict(torch.load(original_model_path, map_location=device))
        original_model = original_model.to(device)

        # Apply pruning to the model
        logger.info(f"Applying structured magnitude-based pruning ({prune_amount:.1%} of attention heads)...")
        pruned_model = prune_attention_heads(model, prune_amount=prune_amount)

        # Save the pruned model
        pruned_model_path = os.path.join(config.MODEL_DIR, 'pruned_model.pt')
        save_model(pruned_model, pruned_model_path)
        logger.info("Pruned model saved")

        # Save the final model (in this case, the pruned model)
        final_model_path = os.path.join(config.MODEL_DIR, 'basketball_lstm_model.pt')
        save_model(pruned_model, final_model_path)

        logger.info("Training, pruning, and evaluation completed successfully!")

    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise


if __name__ == "__main__":
    main()
