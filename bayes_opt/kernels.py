"""
Created on 19/05/2022
@author jdh
"""

import jax
import jax.numpy as jnp
from scipy.spatial.distance import sqeuclidean, cdist

def squared_exponential(xa, xb, sigma=1, l=1):

    # sq_norm = -0.5 * cdist(xa, xb, 'sqeuclidean')
    xa = xa[..., jnp.newaxis]
    xb = xb[..., jnp.newaxis]
    sq_norm = -0.5 * _sqeuclidean(xa, xb)
    return jnp.exp(sq_norm)


import numpy as np
import scipy
# Define the exponentiated quadratic
# def squared_exponential(xa, xb):
#     # L2 distance (Squared Euclidian)
#     sq_norm = -0.5 * scipy.spatial.distance.cdist(xa, xb, 'euclidean')
#
#     return np.exp(sq_norm)


def _sqeuclidean(xa, xb):

    return np.sqrt(((xa - xb.T) ** 2).sum(axis=1))

