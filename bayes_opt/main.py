"""
Created on 19/05/2022
@author jdh
"""

import jax
import jax.numpy as jnp
import jax.random as jr

import matplotlib.pyplot as plt

from kernels import *
from gp import *
import numpy as np
from bayes import *

import imageio

def func(x):
    return jnp.sin(x) + (jnp.power(x, 2) / 10)


n1 = 4
n2 = 100

measurement_noise = 0.2

domain = (-8, 8)

X1 = np.random.uniform(*domain, n1)
_x = np.linspace(*domain, 100)
y1 = func(X1)
y1 += (measurement_noise**2 * np.random.rand(y1.size))
X2 = np.linspace(*domain, n2)





gp = GP(squared_exponential)
gp.data(X1, y1)
m, c = gp.predict(X2)
s = jnp.sqrt(jnp.diag(c))

ei = expected_improvement(gp, X2)




plt.plot(X2, m)
plt.scatter(X1, y1)
plt.fill_between(X2.flat, m - 2*s, m + 2*s, alpha=0.1)
plt.vlines(X2[jnp.argmax(ei)], 0, m[jnp.argmax(ei)], linestyles='dashed')
plt.show()


plt.figure()
plt.plot(X2, ei)
plt.vlines(X2[jnp.argmax(ei)], 0, 1, linestyles='dashed')
plt.show()

