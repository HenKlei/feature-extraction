import numpy as np


def characteristic(k):
    def window(x):
        return (np.linalg.norm(x, np.inf) <= k) * 1.
    return window


def gaussian(sigma):
    def window(x):
        return np.exp(-np.linalg.norm(x)**2 / sigma**2) / (np.sqrt(2.*np.pi) * sigma)
    return window
