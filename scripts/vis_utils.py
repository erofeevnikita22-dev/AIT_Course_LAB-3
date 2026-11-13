from math import ceil
from math import sqrt

import numpy as np


def visualize_grid(xs, ubound=255.0, padding=1):
    n, h, w, c = xs.shape

    grid_size = int(ceil(sqrt(n)))

    grid_h = h * grid_size + padding * (grid_size - 1)
    grid_w = w * grid_size + padding * (grid_size - 1)

    grid = np.zeros(
        (grid_h, grid_w, c)
    )

    index = 0

    for row in range(grid_size):
        for col in range(grid_size):
            if index >= n:
                continue

            image = xs[index]
            low = np.min(image)
            high = np.max(image)

            if high > low:
                image = ubound * (
                    image - low
                ) / (high - low)

            y0 = row * (h + padding)
            x0 = col * (w + padding)

            grid[
                y0:y0 + h,
                x0:x0 + w,
            ] = image

            index += 1

    return grid
