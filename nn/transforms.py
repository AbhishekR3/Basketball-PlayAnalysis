'''
nn/transforms.py

Data augmentation transforms used by the basketball play classifier:
time warping, jittering, and horizontal flipping. ComposeTransforms chains
them together.
'''


#%% Import libraries

import secrets

import numpy as np
from scipy.interpolate import interp1d

from ._logger import logger


#%% Compose

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


#%% Time warping

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


#%% Jitter

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


#%% Horizontal flip

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
