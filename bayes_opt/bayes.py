"""
Created on 20/05/2022
@author jdh
"""

import jax.numpy as jnp
from jax.scipy.stats import norm

# acquisition function
# for bayesian optimisation - we are trying to optimise the value

def expected_improvement(model, x_in, tradeoff_param=0.1):

    mean, cov = model.predict(x_in)
    sigma = jnp.sqrt(jnp.diag(cov))

    best_x_so_far = model.Y[jnp.argmax(model.X)]


    imp = mean - best_x_so_far - tradeoff_param
    Z = imp / sigma
    ei = imp * norm.cdf(Z) + (sigma * norm.pdf(Z))

    # ei if we know the value is zero
    ei = ei.at[sigma==0].set(0)

    return ei



    # wants to return a point to sample

