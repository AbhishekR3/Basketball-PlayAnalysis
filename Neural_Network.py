'''
Basketball Neural Network
This file implements deep learning models for analyzing basketball play patterns.

Key Concepts Implemented:
- LSTM (Long Short-Term Memory) - Sequence modeling capturing temporal dependencies in player movements with recurrent dropout
- Self-Attention - Mechanism for focusing on relevant parts of the sequence regardless of position
- Data Augmentation - Techniques like time warping, jittering, and flipping to increase dataset diversity
- Bi-directional Processing - Forward and backward pass for comprehensive temporal context
- Downsampling - Reducing the number of frames to manage computational load
- Model Pruning - Structured magnitude-based pruning of attention heads to reduce model size
- Metrics Evaluation - Accuracy, Precision, Recall, F1 score, and confusion matrix for model performance assessment
'''

#%%

# Import libraries
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.prune as prune
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from scipy.interpolate import interp1d
import secrets
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
import time
import seaborn as sns
from utils import export_dataframe_to_csv, configure_logger

#%% Set up logging and random seeds

# Configure logging
logger = configure_logger('neural_network')
logger.info("Neural Network Processing started")

# Set random seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

#%% Configure Docker containerization
try:
    # Set environment variables from Docker if available
    log_dir = os.environ.get('LOG_DIR', '/app/logs')
    output_dir = os.environ.get('OUTPUT_DIR', '/app/output')
    model_dir = os.environ.get('MODEL_DIR', '/app/output/models')
    tracking_dir = os.environ.get('TRACKING_DIR', '/app/output/tracking_data')
    
    def ensure_dir(directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    # Make sure directories exist
    ensure_dir(log_dir)
    ensure_dir(output_dir)
    ensure_dir(model_dir)
    ensure_dir(tracking_dir)
    
    print('Log Directory:', log_dir)
    print('Output Directory:', output_dir)
    print('Model Directory:', model_dir)
    print('Tracking Data Directory:', tracking_dir)

except Exception as e:
    # Fallback to local paths using current directory
    current_dir = os.getcwd()
    log_dir = os.path.join(current_dir, 'logs')
    output_dir = os.path.join(current_dir, 'output')
    model_dir = os.path.join(current_dir, 'models')
    tracking_dir = os.path.join(current_dir, 'tracking_data')
    
    logger.error(f"Error in setting up Docker environment: {e}")
    print(f"Error in creating environment for containers: {e}")

#%% Dataset Class

class BasketballPlayDataset(Dataset):
    """Dataset class for basketball plays"""
    
    def __init__(self, root_dir, split='train', transform=None):
        """
        Objective:
        Initialize the basketball play dataset with the proper directory structure

        Parameters:
        [string] root_dir - Root directory of the dataset
        [string] split - Dataset split ('train', 'test', or 'validation')
        [callable] transform - Optional transform to be applied to the features
        """
        try:
            self.root_dir = root_dir
            self.split = split
            self.transform = transform
            
            # Get all CSV files in the split directory
            self.pass_dir = os.path.join(root_dir, split, 'pass')
            self.no_pass_dir = os.path.join(root_dir, split, 'not-pass')
            
            # Check if directories exist
            if not os.path.exists(self.pass_dir):
                logger.warning(f"Pass directory does not exist: {self.pass_dir}")
                self.pass_files = []
            else:
                self.pass_files = [os.path.join(self.pass_dir, f) for f in os.listdir(self.pass_dir) if f.endswith('.csv')]
            
            if not os.path.exists(self.no_pass_dir):
                logger.warning(f"No-pass directory does not exist: {self.no_pass_dir}")
                self.no_pass_files = []
            else:
                self.no_pass_files = [os.path.join(self.no_pass_dir, f) for f in os.listdir(self.no_pass_dir) if f.endswith('.csv')]
            
            self.all_files = self.pass_files + self.no_pass_files
            self.labels = [1] * len(self.pass_files) + [0] * len(self.no_pass_files)
            
            # Shuffle the files and labels together
            indices = np.arange(len(self.all_files))
            np.random.shuffle(indices)
            self.all_files = [self.all_files[i] for i in indices]
            self.labels = [self.labels[i] for i in indices]
            
            logger.info(f"Loaded {len(self.all_files)} files for {split} split")
            
        except Exception as e:
            logger.error(f"Error initializing BasketballPlayDataset: {e}")
            raise
    
    def __len__(self):
        """
        Objective:
        Return the number of samples in the dataset
        
        Returns:
        [int] length - Number of samples
        """
        try:
            return len(self.all_files)
        except Exception as e:
            logger.error(f"Error in __len__: {e}")
            raise
    
    def __getitem__(self, idx):
        """
        Objective:
        Get a sample from the dataset at the given index
        
        Parameters:
        [int] idx - Index of the sample to retrieve
        
        Returns:
        [tuple] (features, label) - Features and label of the sample
        """
        try:
            csv_path = self.all_files[idx]
            label = self.labels[idx]
            
            # Read the CSV file
            df = pd.read_csv(csv_path)
            
            # Sort by Frame to ensure temporal coherence
            df = df.sort_values(by='Frame')

            # Apply regular interval downsampling - select every nth frame
            frame_values = df['Frame'].unique()
            nth_frame_selected = 1 # Select every nth frame, change this to control the downsampling rate
            selected_frames = frame_values[::nth_frame_selected]
            
            # Filter dataframe to only include selected frames
            df = df[df['Frame'].isin(selected_frames)]
            
            # If no frames remain after downsampling, return error
            if len(df) == 0:
                logger.error(f"No frames remain after downsampling for file: {csv_path}")
                print(f"No frames remain after downsampling for file: {csv_path}")
                raise ValueError(f"No frames remain after downsampling for file: {csv_path}")
            
            # Make sure to include TrackID, Frame, and Rank as specified
            feature_cols = [col for col in df.columns if col not in ['Unnamed: 0']]
            features = df[feature_cols].values
            
            # Apply transformations if provided
            if self.transform:
                features = self.transform(features)
            
            # Convert to tensor
            features = torch.FloatTensor(features)
            label = torch.FloatTensor([label])
            
            return features, label
        
        except Exception as e:
            logger.error(f"Error in __getitem__ for index {idx}: {e}")
            raise

#%% Data Processing Functions

def fit_scaler(dataset_path, split='train'):
    """
    Objective:
    Fit a standard scaler on the training data to normalize features
    
    Parameters:
    [string] dataset_path - Path to the dataset
    [string] split - Dataset split to use for fitting (default is 'train')
    
    Returns:
    [StandardScaler] scaler - Fitted standard scaler
    """
    try:
        # Create a dataset instance
        dataset = BasketballPlayDataset(dataset_path, split=split)
        
        # Initialize the scaler
        scaler = StandardScaler()
        
        # Collect all features for fitting
        all_features = []
        
        # Use all files for fitting
        for file in tqdm(dataset.all_files, desc="Fitting scaler"):
            df = pd.read_csv(file)
            feature_cols = [col for col in df.columns if col not in ['Unnamed: 0']]
            features = df[feature_cols].values
            all_features.append(features)
        
        # Concatenate and fit
        all_features = np.vstack([feat for feat in all_features if feat.shape[0] > 0])
        scaler.fit(all_features)
        
        logger.info(f"Scaler fitted on {len(all_features)} samples")
        return scaler
    
    except Exception as e:
        logger.error(f"Error in fit_scaler: {e}")
        raise

class ScaleTransform:
    """Transform class for scaling features"""
    
    def __init__(self, scaler):
        """
        Objective:
        Initialize the transform with a fitted scaler
        
        Parameters:
        [StandardScaler] scaler - Fitted standard scaler
        """
        try:
            self.scaler = scaler
        except Exception as e:
            logger.error(f"Error initializing ScaleTransform: {e}")
            raise
    
    def __call__(self, features):
        """
        Objective:
        Apply the scaling transformation to the features
        
        Parameters:
        [numpy.ndarray] features - Features to scale
        
        Returns:
        [numpy.ndarray] scaled_features - Scaled features
        """
        try:
            return self.scaler.transform(features)
        except Exception as e:
            logger.error(f"Error in ScaleTransform.__call__: {e}")
            raise

def collate_variable_length_sequences(batch):
    """
    Objective:
    Custom collate function for handling variable length sequences in batches
    
    Parameters:
    [list] batch - Batch of (features, label) tuples
    
    Returns:
    [tuple] (padded_features, labels, lengths) - Padded features, labels, and sequence lengths
    """
    try:
        # Sort batch by sequence length (descending)
        batch.sort(key=lambda x: x[0].shape[0], reverse=True)
        
        # Get sequences and labels
        sequences, labels = zip(*batch)
        
        # Get sequence lengths
        lengths = [seq.shape[0] for seq in sequences]
        max_length = max(lengths)
        
        # Get feature dimension
        feature_dim = sequences[0].shape[1]
        
        # Pad sequences
        padded_sequences = torch.zeros(len(sequences), max_length, feature_dim)
        for i, seq in enumerate(sequences):
            padded_sequences[i, :seq.shape[0], :] = seq
        
        # Stack labels
        labels = torch.stack(labels)
        
        return padded_sequences, labels, lengths
    
    except Exception as e:
        logger.error(f"Error in collate_variable_length_sequences: {e}")
        raise

#%% Neural Network Components

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism for sequence modeling with prunable heads"""
    
    def __init__(self, hidden_dim, num_heads=8):
        """
        Objective:
        Initialize the multi-head self-attention module with prunable heads
        
        Parameters:
        [int] hidden_dim - Dimension of the hidden state
        [int] num_heads - Number of attention heads (default: 8)
        """
        try:
            super(MultiHeadAttention, self).__init__()
            self.hidden_dim = hidden_dim
            self.num_heads = num_heads
            self.head_dim = hidden_dim // num_heads
            
            # Ensure hidden_dim is divisible by num_heads
            assert hidden_dim % num_heads == 0, "Hidden dimension must be divisible by number of heads"
            
            # Define the attention components
            self.query = nn.Linear(hidden_dim, hidden_dim)
            self.key = nn.Linear(hidden_dim, hidden_dim)
            self.value = nn.Linear(hidden_dim, hidden_dim)
            
            # Output projection
            self.output_projection = nn.Linear(hidden_dim, hidden_dim)
            
            # Scale factor for attention scores
            self.scale = torch.sqrt(torch.FloatTensor([self.head_dim]))
            
            # Head importance scores (for pruning)
            self.head_importance = nn.Parameter(torch.ones(num_heads, 1))
            
        except Exception as e:
            logger.error(f"Error initializing MultiHeadAttention: {e}")
            raise
    
    def forward(self, hidden_state):
        """
        Objective:
        Forward pass of the multi-head self-attention module
        
        Parameters:
        [torch.Tensor] hidden_state - Input hidden state (batch_size, seq_len, hidden_dim)
        
        Returns:
        [torch.Tensor] attended - Attention-weighted output
        [torch.Tensor] attention_weights - Attention weights for all heads
        """
        try:
            batch_size = hidden_state.shape[0]
            seq_len = hidden_state.shape[1]
            
            # Move scale to the same device as hidden state
            self.scale = self.scale.to(hidden_state.device)
            
            # Linear projections
            Q = self.query(hidden_state)  # (batch_size, seq_len, hidden_dim)
            K = self.key(hidden_state)    # (batch_size, seq_len, hidden_dim)
            V = self.value(hidden_state)  # (batch_size, seq_len, hidden_dim)
            
            # Reshape for multi-head attention
            Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            
            # Calculate attention scores
            energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
            
            # Apply softmax to get attention weights
            attention = torch.softmax(energy, dim=-1)
            
            # Store attention weights for later use
            attention_weights = attention
            
            # Apply head importance (for pruning) - MODIFIED FOR 2D
            # Reshape head_importance for broadcasting [num_heads, 1] -> [1, num_heads, 1, 1]
            head_importance = self.head_importance.view(1, self.num_heads, 1, 1)
            attention = attention * head_importance
            
            # Apply attention weights to values
            weighted_V = torch.matmul(attention, V)
            
            # Reshape back to original dimensions
            weighted_V = weighted_V.permute(0, 2, 1, 3).contiguous()
            weighted_V = weighted_V.view(batch_size, seq_len, self.hidden_dim)
            
            # Apply output projection
            attended = self.output_projection(weighted_V)
            
            return attended, attention_weights
            
        except Exception as e:
            logger.error(f"Error in MultiHeadAttention.forward: {e}")
            raise

class BasketballLSTM(nn.Module):
    """LSTM model for basketball play classification with prunable attention"""
    
    def __init__(self, input_dim, hidden_dim=192, output_dim=1, num_layers=2, 
                dropout=0.3, recurrent_dropout=0.15, bidirectional=True, num_heads=8):
        """
        Objective:
        Initialize the LSTM model for basketball play classification with prunable attention
        
        Parameters:
        [int] input_dim - Dimension of input features
        [int] hidden_dim - Dimension of LSTM hidden state (default: 192)
        [int] output_dim - Dimension of output (1 for binary classification)
        [int] num_layers - Number of LSTM layers (default: 2)
        [float] dropout - Dropout probability (default: 0.3)
        [float] recurrent_dropout - Recurrent dropout probability (default: 0.15)
        [bool] bidirectional - Whether to use bidirectional LSTM (default: True)
        [int] num_heads - Number of attention heads (default: 8)
        """
        try:
            super(BasketballLSTM, self).__init__()
            
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers
            self.bidirectional = bidirectional
            self.num_directions = 2 if bidirectional else 1
            
            # LSTM layers
            self.lstm = nn.LSTM(
                input_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional
            )
            
            # Apply recurrent dropout
            self.recurrent_dropout = nn.Dropout(recurrent_dropout)
            
            # Multi-head self-attention layer (prunable)
            attention_dim = hidden_dim * self.num_directions
            self.attention = MultiHeadAttention(attention_dim, num_heads=num_heads)
            
            # Layer normalization
            self.layer_norm = nn.LayerNorm(attention_dim)
            
            # Output layer
            self.fc = nn.Linear(attention_dim, output_dim)
            
            # Sigmoid activation for binary classification
            self.sigmoid = nn.Sigmoid()
            
            logger.info(f"Initialized BasketballLSTM model with {input_dim} input features, "
                        f"{hidden_dim} hidden dim, {num_layers} layers, "
                        f"bidirectional={bidirectional}, attention_heads={num_heads}")
            
        except Exception as e:
            logger.error(f"Error initializing BasketballLSTM: {e}")
            raise
    
    def forward(self, x, hidden=None):
        """
        Objective:
        Forward pass of the LSTM model
        
        Parameters:
        [torch.Tensor] x - Input sequence of shape (batch_size, seq_len, input_dim)
        [tuple] hidden - Initial hidden state (optional)
        
        Returns:
        [torch.Tensor] out - Output prediction
        [tuple] hidden - Final hidden state
        """
        try:
            batch_size = x.size(0)
            
            # Initialize hidden state if not provided
            if hidden is None:
                h0 = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_dim).to(x.device)
                c0 = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_dim).to(x.device)
                hidden = (h0, c0)
            
            # LSTM forward
            lstm_out, hidden = self.lstm(x, hidden)  # lstm_out: (batch_size, seq_len, hidden_dim * num_directions)
            
            # Apply recurrent dropout to the output
            lstm_out = self.recurrent_dropout(lstm_out)
            
            # Apply multi-head self-attention
            attended, _ = self.attention(lstm_out)
            
            # Residual connection and layer normalization
            attended = self.layer_norm(lstm_out + attended)
            
            # Use the average of the attended output across the sequence
            out = attended.mean(dim=1)
            
            # Final prediction
            out = self.fc(out)
            out = self.sigmoid(out)
            
            return out, hidden
            
        except Exception as e:
            logger.error(f"Error in BasketballLSTM.forward: {e}")
            raise

#%% Model Pruning Functions

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

#%% Data Augmentation Functions

class ComposeTransforms:
    """Composes multiple transforms together."""
    
    def __init__(self, transforms):
        """
        Objective:
        Initialize a composition of transforms
        
        Parameters:
        [list] transforms - List of transform functions to apply
        """
        try:
            self.transforms = transforms
        except Exception as e:
            logger.error(f"Error initializing ComposeTransforms: {e}")
            raise
    
    def __call__(self, features):
        """
        Objective:
        Apply each transform in sequence
        
        Parameters:
        [numpy.ndarray] features - Features to transform
        
        Returns:
        [numpy.ndarray] transformed_features - Transformed features
        """
        try:
            transformed_features = features.copy()
            for transform in self.transforms:
                transformed_features = transform(transformed_features)
            return transformed_features
        except Exception as e:
            logger.error(f"Error in ComposeTransforms.__call__: {e}")
            raise

def time_warp_transform(sigma=0.2, num_knots=4):
    """
    Objective:
    Create a time warping transform with given parameters
    
    Parameters:
    [float] sigma - Standard deviation for the displacement of knots (0.2=low, 0.5=medium)
    [int] num_knots - Number of knots to use for warping (fewer knots = smoother warping)
    
    Returns:
    [callable] transform - A transform function that applies time warping
    """
    try:
        def transform(features):
            """
            Objective:
            Apply time warping to the input features
            
            Parameters:
            [numpy.ndarray] features - Features to transform of shape (time_steps, features)
            
            Returns:
            [numpy.ndarray] warped_features - Time warped features
            """
            try:
                # Skip warping if sequence is too short
                seq_len = features.shape[0]
                if seq_len <= num_knots + 2:
                    return features
                
                # Create knot positions for warping
                knot_positions = np.linspace(0, seq_len - 1, num_knots + 2).astype(int)
                source_knots = np.copy(knot_positions)
                
                # First and last knots should remain fixed to maintain sequence boundaries
                target_knots = np.copy(source_knots)
                
                # Add random displacement to interior knots
                displacement = np.random.normal(0, sigma * seq_len, num_knots)
                target_knots[1:-1] += displacement.astype(int)
                
                # Ensure target knots remain within bounds and are strictly increasing
                target_knots = np.clip(target_knots, 0, seq_len - 1)
                target_knots = np.sort(target_knots)
                
                # Create warping function using linear interpolation
                warping_func = interp1d(source_knots, target_knots, kind='linear', bounds_error=False, 
                                        fill_value=(target_knots[0], target_knots[-1]))
                
                # Apply warping to each time step
                warped_indices = np.clip(warping_func(np.arange(seq_len)), 0, seq_len - 1).astype(int)
                warped_features = features[warped_indices]
                
                return warped_features
                
            except Exception as e:
                logger.error(f"Error in time warping transform: {e}")
                return features  # Return original features on error
        
        return transform
    
    except Exception as e:
        logger.error(f"Error creating time warping transform: {e}")
        raise


def jitter_transform(intensity=0.05, exclude_columns=None):
    """
    Objective:
    Create a more robust jitter transform that selectively adds noise based on feature type and handles edge cases
    
    Parameters:
    [float] intensity - Intensity of jitter as a fraction of feature standard deviation (0.05=low)
    [list] exclude_columns - Optional list of column indices to exclude from jittering
    
    Returns:
    [callable] transform - A transform function that applies jittering
    """
    try:
        def transform(features):
            """
            Objective:
            Apply selective jittering to the input features with improved error handling
            
            Parameters:
            [numpy.ndarray] features - Features to transform
            
            Returns:
            [numpy.ndarray] jittered_features - Features with added noise
            """
            try:
                # Check if features is empty or has invalid shape
                if features.size == 0 or features.ndim != 2:
                    logger.warning("Skipping jitter: Invalid feature shape or empty array")
                    return features
                
                # Make a copy to avoid modifying the original
                jittered_features = features.copy()
                
                # Initialize the exclude_columns set if it's None
                exclude_cols = set() if exclude_columns is None else set(exclude_columns)
                
                # Automatic column exclusion (add categorical/boolean columns)
                for col_idx in range(features.shape[1]):
                    # Skip if explicitly excluded
                    if col_idx in exclude_cols:
                        continue
                        
                    # Get the column data
                    column = features[:, col_idx]
                    
                    # Skip if column is empty
                    if len(column) == 0:
                        continue
                    
                    # Skip if all values are the same (likely categorical/boolean)
                    if np.all(column == column[0]):
                        continue
                    
                    # Skip if column contains only a few unique values (likely categorical)
                    unique_values = np.unique(column)
                    if len(unique_values) < 5:
                        continue
                    
                    try:
                        # Compute standard deviation for this column
                        col_std = np.nanstd(column.astype(np.float64))
                        
                        # If std is valid and non-zero, add noise
                        if np.isfinite(col_std) and col_std > 1e-10:
                            # Generate noise for this column - use a smaller multiplier for very large stds
                            # to prevent extreme values
                            noise_scale = min(col_std, 1.0) * intensity
                            noise = np.random.normal(0, noise_scale, size=len(column))
                            
                            # Add noise to column
                            jittered_features[:, col_idx] = column + noise
                        else:
                            # For very small std, add minimal noise to avoid unchanged values
                            tiny_noise = np.random.normal(0, intensity * 0.001, size=len(column))
                            jittered_features[:, col_idx] = column + tiny_noise
                    except Exception as column_error:
                        logger.debug(f"Skipping jitter on column {col_idx}: {column_error}")
                        continue
                
                return jittered_features
            
            except Exception as e:
                logger.warning(f"Error in jitter transform: {e}")
                return features  # Return original features on error
        
        return transform
    
    except Exception as e:
        logger.error(f"Error creating jitter transform: {e}")
        raise

def horizontal_flip_transform(flip_probability=0.5, x_position_col=None, x_velocity_col=None):
    """
    Objective:
    Create a horizontal flip transform for spatial coordinates and velocities
    
    Parameters:
    [float] flip_probability - Probability of applying the flip
    [int] x_position_col - Column index for x position
    [int] x_velocity_col - Column index for x velocity
    
    Returns:
    [callable] transform - A transform function that applies horizontal flipping
    """
    try:
        def transform(features):
            """
            Objective:
            Apply horizontal flip to the input features
            
            Parameters:
            [numpy.ndarray] features - Features to transform
            
            Returns:
            [numpy.ndarray] flipped_features - Features with horizontally flipped coordinates
            """
            try:
                # Skip based on probability
                if secrets.SystemRandom().random() > flip_probability:
                    return features
                
                # Make a copy of features to avoid modifying the original
                flipped_features = features.copy()
                
                # Flip x positions (negate values)
                if x_position_col is not None:
                    flipped_features[:, x_position_col] = -flipped_features[:, x_position_col]
                
                # Flip x velocities (negate values)
                if x_velocity_col is not None:
                    flipped_features[:, x_velocity_col] = -flipped_features[:, x_velocity_col]
                
                return flipped_features
            
            except Exception as e:
                logger.error(f"Error in horizontal flip transform: {e}")
                return features  # Return original features on error
        
        return transform
    
    except Exception as e:
        logger.error(f"Error creating horizontal flip transform: {e}")
        raise

#%% Training and Evaluation Functions

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
                    best_model_path = os.path.join(model_dir, 'best_model.pt')
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
        history_path = os.path.join(model_dir, 'training_history.csv')
        export_dataframe_to_csv(history_df, history_path, logger)
        
        return history
            
    except Exception as e:
        logger.error(f"Error in train_model: {e}")
        raise

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
        metrics_path = os.path.join(model_dir, 'test_metrics.csv')
        export_dataframe_to_csv(metrics_df, metrics_path, logger)
        
        # Plot and save confusion matrix
        plot_confusion_matrix(conf_matrix, os.path.join(model_dir, 'confusion_matrix.png'))
        
        return metrics
            
    except Exception as e:
        logger.error(f"Error in evaluate_model: {e}")
        raise

#%% Utility Functions

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
        comparison_path = os.path.join(model_dir, 'pruning_comparison.png')
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
        metrics_path = os.path.join(model_dir, 'pruning_metrics.csv')
        export_dataframe_to_csv(pruning_metrics_df, metrics_path, logger)
        
        return pruning_metrics
    
    except Exception as e:
        logger.error(f"Error in analyze_and_visualize_pruning: {e}")
        raise

#%% Main Function

def main():
    """
    Objective:
    Main function to run the basketball play classification with model pruning
    """
    try:
        # Set parameters
        try:
            # Use environment variables if available
            data_path = os.environ.get('DATA_PATH', '/Users/abhishekramesh/Desktop/Passing')
        except TypeError:
            data_path = '/Users/abhishekramesh/Desktop/Passing'
            
        batch_size = 32
        num_epochs = 20
        learning_rate = 0.0005
        early_stopping_patience = 4
        prune_amount = 0.4
        
        # Data augmentation parameters
        use_augmentation = True
        time_warp_sigma = 0.2  # Low-medium sigma
        time_warp_knots = 4    # Low knot value
        jitter_intensity = 0.05  # Low jittering
        flip_probability = 0.5  # 50% chance of applying horizontal flip
        
        # Multi-head attention parameters
        num_heads = 8  # Number of attention heads
        
        # Check device type
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_built() and torch.backends.mps.is_available():
            device = torch.device('mps')  # Apple Silicon GPU
        else:
            device = torch.device('cpu')
            
        logger.info(f"Using device: {device}")
        
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
                        pos_x_col = 16  # Hardcoded fallback
                        logger.info(f"Using fallback pos_x_col index: {pos_x_col}")
                    if vel_x_col is None:
                        vel_x_col = 20  # Hardcoded fallback
                        logger.info(f"Using fallback vel_x_col index: {vel_x_col}")
                else:
                    # No sample file found
                    logger.warning("No sample CSV files found in the dataset directory")
                    pos_x_col = 16  # Hardcoded fallback
                    vel_x_col = 20  # Hardcoded fallback
                    logger.info(f"Using fallback column indices: pos_x={pos_x_col}, vel_x={vel_x_col}")
            
            except Exception as e:
                logger.warning(f"Error detecting column indices: {e}")
                pos_x_col = 16  # Hardcoded fallback
                vel_x_col = 20  # Hardcoded fallback
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
            hidden_dim=128,
            output_dim=1,
            num_layers=2,
            dropout=0.6,
            recurrent_dropout=0.3,
            bidirectional=True,
            num_heads=num_heads
        )
        
        # Initialize loss function and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)  # L2 regularization
        
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
            early_stopping_patience=early_stopping_patience
        )
        
        # Plot and save training history
        history_plot_path = os.path.join(model_dir, 'training_history.png')
        plot_training_history(history, save_path=history_plot_path)
        
        # Load the best model
        best_model_path = os.path.join(model_dir, 'best_model.pt')
        if os.path.exists(best_model_path):
            model = load_model(model, best_model_path, device)
        
        # Save a copy of the original (unpruned) model
        original_model_path = os.path.join(model_dir, 'original_model.pt')
        save_model(model, original_model_path)
        logger.info("Original model saved before pruning")
        
        # Create a copy of the original model for comparison
        original_model = BasketballLSTM(
            input_dim=input_dim,
            hidden_dim=128,
            output_dim=1,
            num_layers=2,
            dropout=0.6,
            recurrent_dropout=0.3,
            bidirectional=True,
            num_heads=num_heads
        )

        original_model.load_state_dict(torch.load(original_model_path, map_location=device))
        original_model = original_model.to(device)
        
        # Apply pruning to the model
        logger.info(f"Applying structured magnitude-based pruning ({prune_amount:.1%} of attention heads)...")
        pruned_model = prune_attention_heads(model, prune_amount=prune_amount)
        
        # Save the pruned model
        pruned_model_path = os.path.join(model_dir, 'pruned_model.pt')
        save_model(pruned_model, pruned_model_path)
        logger.info("Pruned model saved")
        
        # Save the final model (in this case, the pruned model)
        final_model_path = os.path.join(model_dir, 'basketball_lstm_model.pt')
        save_model(pruned_model, final_model_path)
        
        logger.info("Training, pruning, and evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise

if __name__ == "__main__":
    main()