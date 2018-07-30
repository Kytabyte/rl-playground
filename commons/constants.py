"""
  constants
"""
import numpy as np
import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MACHINE_EPS = np.finfo(np.float32).eps.item()
