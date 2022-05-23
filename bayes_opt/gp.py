"""
Created on 19/05/2022
@author jdh
"""

import jax
import jax.numpy as jnp
import tensorflow_probability as tfp


class GP:

    def __init__(self, kernel_function, noise_scale=0.5):

        self.kernel_function = kernel_function
        # self.mean_function = mean_function
        self.noise_scale = noise_scale


    def data(self, X, Y):
        self.X = X
        self.Y = Y

    def add_data(self, x, y):

        self.X = jnp.append(self.X, x)
        self.Y = jnp.append(self.Y, y)


    def predict(self, x):

        sigma11 = self.kernel_function(self.X, self.X) + (self.noise_scale**2 * jnp.eye(self.X.size))
        sigma22 = self.kernel_function(x, x)
        sigma12 = self.kernel_function(self.X, x)

        mean_solver = jax.scipy.linalg.solve(sigma11, sigma12).T
        posterior_mean = mean_solver @ self.Y

        posterior_cov = sigma22 - (mean_solver @ sigma12)

        return posterior_mean, posterior_cov





def GP_noiseless(X1, y1, X2, kernel_function):
    '''
    :param X1: observation inputs
    :param y1: observated outputs
    :param X2: inputs for prediction
    :param kernel_function: kernel function
    :return:
    '''

    sigma11 = kernel_function(X1, X1)
    sigma22 = kernel_function(X2, X2)
    sigma12 = kernel_function(X1, X2)

    mean_solver = jax.scipy.linalg.solve(sigma11, sigma12).T
    posterior_mean = mean_solver @ y1

    posterior_cov = sigma22 - (mean_solver @ sigma12)

    return posterior_mean, posterior_cov


# noisy just adds some noise to the diagonal elements of the kernel
# diagonal elements only as gaussian noise is independently distributed
def GP_noisy(X1, y1, X2, kernel_function, measurement_noise_scale):
    '''
    :param X1: observation inputs
    :param y1: observated outputs
    :param X2: inputs for prediction
    :param kernel_function: kernel function
    :return:
    '''

    sigma11 = kernel_function(X1, X1) #+ (measurement_noise_scale**2 * jnp.eye(X1.size))
    sigma22 = kernel_function(X2, X2)
    sigma12 = kernel_function(X1, X2)

    mean_solver = jax.scipy.linalg.solve(sigma11, sigma12).T
    posterior_mean = mean_solver @ y1

    posterior_cov = sigma22 - (mean_solver @ sigma12)

    return posterior_mean, posterior_cov



