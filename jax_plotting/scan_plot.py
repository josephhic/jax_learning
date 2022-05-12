"""
Created on 12/05/2022
@author jdh
"""

from plot import Plot

import jax
import numpy as np


def scan_plot(step, initial_values, xs, tracer=True, tracer_len=10, size=600, pixel_size=5):

    final, coords = jax.lax.scan(step, initial_values, xs)

    rescaled_coords = (coords/np.max(coords))
    number_of_frames = xs.__len__()

    frames = np.zeros(shape=(number_of_frames, size, size))

    # -1 is hash factor for now otherwise we have index error
    pixel_coords = (size * rescaled_coords).astype(int) - 1


    for i, coord in enumerate(pixel_coords):

        x, y = coord
        frame = frames[i]

        frame[x:x+pixel_size, y:y+pixel_size] += 1

        if tracer:
            trace = np.min(np.array([tracer_len, i]))

            frames[i:i+trace, x:x+pixel_size, y:y+pixel_size] += 0.2


    Plot(frames)

