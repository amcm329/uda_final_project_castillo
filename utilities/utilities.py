import os
import csv
import time
import math 

import rasterio
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ============================================================
# We define utilities (timing)
# ============================================================

def format_seconds(seconds):
    """
    Formats elapsed seconds into a human-readable string.

    Args:
        seconds (float): Elapsed time in seconds.

    Returns:
        str: Formatted time string.
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes = int(seconds // 60)
    rem = seconds - 60 * minutes
    return f"{minutes}m {rem:.2f}s"

