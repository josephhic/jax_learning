"""
Created on 20/05/2022
@author jdh
"""

import jax.numpy as jnp
from jax.scipy.stats import norm

import matplotlib.pyplot as plt
import imageio

from gp import GP

# acquisition function
# for bayesian optimisation - we are trying to optimise the value

class BayesOpt:

    def __init__(self, opt_function, kernel_function, initial_x, initial_y, domain, tradeoff_param=0.1):


        self.opt_function = opt_function
        self.GP = GP(kernel_function)
        self.domain = domain # linspace
        self.tradeoff_param = tradeoff_param

        self.GP.data(initial_x, initial_y)
        self.pngs = []
        self.file = './giffer/'

    def optimise(self, max_iter=20):

        for i, _ in enumerate(range(max_iter)):

            next_sample = self.next_sample()
            y_new = self.sample_function_at(next_sample)
            self._add_data(next_sample, y_new)
            # self.plot(i)

        # self.gif()




    def next_sample(self):

        ei = self._expected_improvement()

        sample_at = self.domain[jnp.argmax(ei)]
        return sample_at

    def sample_function_at(self, x):

        return self.opt_function(x)

    def _add_data(self, x, y):

        self.GP.add_data(x, y)

    def plot(self, i):
        plt.figure()
        plt.plot(self.domain, self.opt_function(self.domain))
        plt.scatter(self.GP.X, self.GP.Y)

        mean, covariance = self.GP.predict(self.domain)
        std = jnp.sqrt(jnp.diag(covariance))

        plt.fill_between(self.domain, mean - 2*std, mean + 2*std, alpha=0.2)

        filename = "{}{}.png".format(self.file, i)
        plt.savefig(filename)
        self.pngs.append(filename)
        plt.show()

    def gif(self):
        images = []
        for png in self.pngs:
            img = imageio.imread(png)
            images.append(img)

        imageio.mimsave('./gif.gif', images)



    def _expected_improvement(self):

        mean, cov = self.GP.predict(self.domain)
        sigma = jnp.sqrt(jnp.diag(cov))

        best_x_so_far = self.GP.Y[jnp.argmax(self.GP.X)]

        imp = mean - best_x_so_far - self.tradeoff_param
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + (sigma * norm.pdf(Z))

        # ei if we know the value is zero
        ei = ei.at[sigma == 0].set(0)

        return ei




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

