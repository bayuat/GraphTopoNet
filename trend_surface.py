import numpy as np
from sklearn.preprocessing import PolynomialFeatures

def trend_surface(X, degree=2):
    height, width = X.shape
    x_coords, y_coords = np.meshgrid(range(width), range(height))
    coords = np.stack([x_coords.flatten(), y_coords.flatten()], axis=1)
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    trend = poly.fit_transform(coords)
    return trend.reshape(height, width, -1).transpose(2, 0, 1)