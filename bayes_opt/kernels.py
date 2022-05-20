"""
Created on 19/05/2022
@author jdh
"""

import jax
import jax.numpy as jnp
from scipy.spatial.distance import sqeuclidean, cdist

def squared_exponential(xa, xb, sigma=1, l=1):

    # sq_norm = -0.5 * cdist(xa, xb, 'sqeuclidean')
    xa = xa.flatten()[:, jnp.newaxis]
    xb = xb.flatten()[:, jnp.newaxis]
    sq_norm = -0.5 * _sqeuclidean(xa, xb)
    return jnp.exp(sq_norm)


def _sqeuclidean(xa, xb):

    return ((xa - xb.T) ** 2)


