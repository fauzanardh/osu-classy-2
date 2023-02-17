from typing import Union, Tuple

import numpy as np


HIT_SD = 5


def smooth_hit(
    x: np.ndarray, mu: Union[float, Tuple[float, float]], sigma: float = HIT_SD
):
    """Smooths a hit with a normal distribution"""
    if isinstance(mu, float):
        z = (x - mu) / sigma
    elif isinstance(mu, tuple):
        a, b = mu
        z = np.where(x < a, x - a, np.where(x < b, 0, x - b)) / sigma
    else:
        raise TypeError("mu must be a float or a tuple of two floats")

    return np.exp(-0.5 * z**2)
