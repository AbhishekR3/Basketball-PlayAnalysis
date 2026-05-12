'''
config.py

Centralized configuration for the Basketball Play Analysis pipeline.

All env-var-driven paths, hyperparameters, and tunables that were previously
duplicated across stage scripts live here. Stage scripts import from this
module rather than redeclaring their own constants. Environment variables
still override defaults; the /app/... defaults are Docker-oriented.
'''


#%% Import libraries

import os


#%% Reproducibility

# Random seed used by the Neural Network stage (torch + numpy).
SEED = 42


#%% Pipeline directories

# Read once at import. Mutated in-place by setup_pipeline_dirs() on fallback so
# that submodules accessing values via config.<NAME> see the updated path.
LOG_DIR = os.environ.get('LOG_DIR', '/app/logs')
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', '/app/output')
MODEL_DIR = os.environ.get('MODEL_DIR', '/app/output/models')
TRACKING_DIR = os.environ.get('TRACKING_DIR', '/app/output/tracking_data')
VIDEO_DIR = os.environ.get('VIDEO_DIR', '/app/simulations')
ASSETS_DIR = os.environ.get('ASSETS_DIR', '/app/assets')
DEEPSORT_DIR = os.environ.get('DeepSORT_DIR', '/app/deep_sort')
DATA_PATH = os.environ.get('DATA_PATH', '/Users/abhishekramesh/Desktop/Passing')


def ensure_dir(directory):
    """
    Objective:
    Create the given directory if it does not already exist

    Parameters:
    [string] directory - Directory path to create
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def setup_pipeline_dirs(logger=None):
    """
    Objective:
    Ensure the Neural Network pipeline directories exist; fall back to
    current-working-directory paths if creation fails (mirrors the
    original Neural_Network.py top-level try/except setup).

    Parameters:
    [logging.Logger] logger - Optional logger for error reporting
    """
    global LOG_DIR, OUTPUT_DIR, MODEL_DIR, TRACKING_DIR

    try:
        ensure_dir(LOG_DIR)
        ensure_dir(OUTPUT_DIR)
        ensure_dir(MODEL_DIR)
        ensure_dir(TRACKING_DIR)

        print('Log Directory:', LOG_DIR)
        print('Output Directory:', OUTPUT_DIR)
        print('Model Directory:', MODEL_DIR)
        print('Tracking Data Directory:', TRACKING_DIR)

    except Exception as e:
        current_dir = os.getcwd()
        LOG_DIR = os.path.join(current_dir, 'logs')
        OUTPUT_DIR = os.path.join(current_dir, 'output')
        MODEL_DIR = os.path.join(current_dir, 'models')
        TRACKING_DIR = os.path.join(current_dir, 'tracking_data')

        if logger is not None:
            logger.error(f"Error in setting up Docker environment: {e}")
        print(f"Error in creating environment for containers: {e}")


#%% Object_Tracking - YOLO confidence thresholds

# filter_lowconfidence defaults from Object_Tracking.py
YOLO_BASKETBALL_SCORE = 0.5
YOLO_PLAYER_SCORE = 0.8


#%% Feature_Engineering

# Default video frame rate used by temporal-feature processing
FPS = 30


#%% Data_Loading

# PostgreSQL connection string. Default mirrors the value previously hardcoded
# in Data_Loading.py; replace via the DB_URL env var before running against a
# real database.
DB_URL = os.environ.get(
    'DB_URL',
    'postgresql://postgres:password@localhost:5432/postgres',
)


#%% Neural_Network hyperparameters

# Training loop
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 0.0005
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 4
PRUNE_AMOUNT = 0.4

# Model architecture
HIDDEN_DIM = 128
OUTPUT_DIM = 1
NUM_LAYERS = 2
DROPOUT = 0.6
RECURRENT_DROPOUT = 0.3
BIDIRECTIONAL = True
NUM_HEADS = 8

# Data augmentation
USE_AUGMENTATION = True
TIME_WARP_SIGMA = 0.2
TIME_WARP_KNOTS = 4
JITTER_INTENSITY = 0.05
FLIP_PROBABILITY = 0.5

# Dataset downsampling (BasketballPlayDataset.__getitem__)
NTH_FRAME_SELECTED = 1

# Hardcoded fallback column indices used when pos_x / vel_x cannot be located
# from the CSV header. Preserved from the original Neural_Network.main().
POS_X_COL_FALLBACK = 16
VEL_X_COL_FALLBACK = 20
