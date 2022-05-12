"""
Created on 12/05/2022
@author jdh
"""

import jax
import jax.numpy as jnp

def euler_step(f, X, dt):

    return X + (dt * f(X))


def f(X, alpha=0.66, beta=1.33, gamma=1, delta=1):

    x, y = X

    x_prime = (alpha * x) - (beta * x * y)
    y_prime = (delta * x * y) - (gamma * y)

    return jnp.array([x_prime, y_prime])

def step(carry, t):

    new_carry = euler_step(f, carry, dt)
    return (new_carry, new_carry)

initial_values = jnp.array([0.9, 0.9])
xs = jnp.linspace(0, 10, 1000)

dt = xs[1] - xs[0]

final, coords = jax.lax.scan(step, initial_values, xs)

x, y = coords.T

import matplotlib.pyplot as plt

plt.figure()
plt.plot(x)
plt.plot(y)
plt.show()