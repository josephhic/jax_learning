"""
Created on 12/05/2022
@author jdh
"""

# -*- coding: utf-8 -*-
"""
Demonstrates very basic use of ImageItem to display image data inside a ViewBox.
"""

from pyqtgraph.Qt import QtCore, QtGui
import numpy as np
import pyqtgraph as pg
import pyqtgraph.ptime as ptime

import jax
import jax.numpy as jnp

from scan_plot import scan_plot


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


frames = 1000
initial_values = jnp.array([0.9, 0.9])
xs = jnp.linspace(0, 10, frames)

dt = xs[1] - xs[0]

# final, coords = jax.lax.scan(step, initial_values, xs)

scan_plot(step, initial_values, xs, tracer_len=1000)
#
# x, y = coords.T
#
# shape = 600
# data = np.zeros(shape=(frames, shape, shape))
#
# pixel_coords = (((coords * shape)/(np.max(coords)))).astype(int)
#
# for i, (coord) in enumerate(pixel_coords):
#     x, y = coord
#     frame = data[i]
#     frame[x:x+10, y:y+10] += 1
#
#     if i > 11:
#         data[i:i+10:, x:x+10, y:y+10] += 0.2
#
#
#
# from plot import Plot as jaxplot
#
# live = jaxplot(data)
#
