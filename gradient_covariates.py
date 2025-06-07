import numpy as np

def gradient_covariates(X):
    gradients = [np.gradient(X[i]) for i in range(X.shape[0])]
    flattened_gradients = [grad for gradient in gradients for grad in gradient]
    return np.array(flattened_gradients)