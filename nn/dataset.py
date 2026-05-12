'''
nn/dataset.py

Dataset, scaler and collate utilities for the basketball play classifier.
'''


#%% Import libraries

import os

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from tqdm import tqdm

import config
from ._logger import logger


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
            nth_frame_selected = config.NTH_FRAME_SELECTED
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
