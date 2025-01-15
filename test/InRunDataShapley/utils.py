import os
import pickle
import tempfile
import shutil
import tiktoken
import torch
import random
import numpy as np


def set_seed(seed):
    # Set seed for Python's built-in random module
    random.seed(seed)
    
    # Set seed for NumPy's random number generator
    np.random.seed(seed)
    
    # Set seed for PyTorch's random number generator
    torch.manual_seed(seed)
    
    # If you are using GPUs
    if torch.cuda.is_available():
        # Set seed for all CUDA devices
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # Ensure that CUDA operations are deterministic
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False