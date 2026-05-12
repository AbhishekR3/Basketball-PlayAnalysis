'''
nn/_logger.py

Single logger instance shared across every nn.* submodule. Created once at
package import time so the Neural Network stage produces one log stream
regardless of how many submodules import it.
'''


#%% Import libraries

from utils import configure_logger


#%% Shared logger

logger = configure_logger('neural_network')
