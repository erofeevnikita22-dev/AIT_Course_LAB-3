import numpy as np


def eval_numerical_gradient(f, x, verbose=True, h=1e-5):
    grad = np.zeros_like(x)

    it = np.nditer(
        x,
        flags=["multi_index"],
        op_flags=["readwrite"],
    )

    while not it.finished:
        index = it.multi_index
        old_value = x[index]

        x[index] = old_value + h
        pos = f(x)

        x[index] = old_value - h
        neg = f(x)

        x[index] = old_value

        grad[index] = (pos - neg) / (2 * h)

        if verbose:
            print(index, grad[index])

        it.iternext()

    return grad


def eval_numerical_gradient_array(f, x, df, h=1e-5):
    grad = np.zeros_like(x)

    it = np.nditer(
        x,
        flags=["multi_index"],
        op_flags=["readwrite"],
    )

    while not it.finished:
        index = it.multi_index
        old_value = x[index]

        x[index] = old_value + h
        pos = f(x).copy()

        x[index] = old_value - h
        neg = f(x).copy()

        x[index] = old_value

        grad[index] = np.sum((pos - neg) * df) / (2 * h)

        it.iternext()

    return grad
