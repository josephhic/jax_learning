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

import imageio

#
# xlim = (-3, 3)
# X = jnp.expand_dims(jnp.linspace(*xlim, 25), 1)
# sigma = squared_exponential(X, X)
#
# plt.imshow(sigma)
# plt.show()

samples = 41
functions = 5
X = jnp.linspace(-4, 4, samples)[:, jnp.newaxis]

sigma = squared_exponential(X, X)
mean = jnp.zeros(samples)

# svd method works, cholesky (default) does not work
ys = jr.multivariate_normal(
    key=jax.random.PRNGKey(199),
    mean=mean,
    cov=sigma,
    method='svd'
)





def func(x):
    return jnp.sin(x) + (jnp.power(x, 2) / 10)


n1 = 10
n2 = 100

measurement_noise = 0.2

domain = (-8, 8)

X1 = np.random.uniform(*domain, n1)
_x = np.linspace(*domain, 100)
y1 = func(X1)
# y1 += (measurement_noise**2 * np.random.rand(y1.size))
X2 = np.linspace(*domain, n2)

# gp = GP(squared_exponential)
#
# gp.data(X1, y1)
#
# mean, cov = gp.predict(X2)

# plt.figure()
# plt.plot(X2, mean)
# plt.plot(_x, func(_x), alpha=0.3)
# plt.show()



plt.figure()
plt.ylim(jnp.min(y1) - 1, jnp.max(y1) + 1)


plt.scatter(X1, y1)
plt.plot(_x, func(_x))


mean1, cov1 = GP_noisy(X1, y1, X2, squared_exponential, measurement_noise_scale=measurement_noise)

gp = GP(squared_exponential)
gp.data(X1, y1)
m, c = gp.predict(X2)
s= jnp.sqrt(jnp.diag(c))

plt.plot(X2, m)
plt.fill_between(X2.flat, m - 2*s, m + 2*s, alpha=0.1)
plt.show()


this_y = jr.multivariate_normal(
    key=jax.random.PRNGKey(199),
    mean=mean1,
    cov=cov1,
    method='svd'
)

std = jnp.sqrt(jnp.diag(cov1))

plt.figure()
plt.plot(X2, mean1)
plt.fill_between(X2.flat, (mean1 - (2 * std)), (mean1 + (2 * std)), color='blue', alpha=0.1)
# plt.savefig('./noisy_gif/{}.png'.format(i))
plt.show()


# with imageio.get_writer('./noisy_gif/gif.gif', mode='I') as writer:
#     for filename in ['./noisy_gif/{}.png'.format(i) for i in range(X1.__len__())]:
#         image = imageio.imread(filename)
#         writer.append_data(image)


# atmospheric model

# x = np.linspace(0, 100, 1000)
# y = func(x)
# plt.figure()
# plt.plot(x, y)
# # plt.show()
#
# prior_mean = jnp.mean(y)
# prior_mean_function = lambda _: prior_mean