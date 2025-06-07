import numpy as np
from scipy.ndimage import distance_transform_edt

def compute_radar_confidence(radar_mask, kernel_size=15):
    distance_map = distance_transform_edt(~radar_mask)
    confidence_map = np.exp(-distance_map / kernel_size)
    return confidence_map