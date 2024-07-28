"""
matsjfunke
"""

import numpy as np

def relu(feature_map):
    return np.maximum(0, feature_map)
