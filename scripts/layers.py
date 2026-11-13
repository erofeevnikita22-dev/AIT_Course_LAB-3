import numpy as np


def affine_forward(x, w, b):
    n = x.shape[0]
    x_row = x.reshape(n, -1)
    out = x_row.dot(w) + b
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    x, w, b = cache
    n = x.shape[0]
    x_row = x.reshape(n, -1)

    dx = dout.dot(w.T).reshape(x.shape)
    dw = x_row.T.dot(dout)
    db = np.sum(dout, axis=0)

    return dx, dw, db


def relu_forward(x):
    out = np.maximum(0, x)
    cache = x
    return out, cache


def relu_backward(dout, cache):
    x = cache
    dx = dout.copy()
    dx[x <= 0] = 0
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)

    n, d = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(d, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(d, dtype=x.dtype))

    if mode == "train":
        mean = np.mean(x, axis=0)
        var = np.var(x, axis=0)

        x_centered = x - mean
        inv_std = 1.0 / np.sqrt(var + eps)
        x_norm = x_centered * inv_std
        out = gamma * x_norm + beta

        running_mean = momentum * running_mean + (1 - momentum) * mean
        running_var = momentum * running_var + (1 - momentum) * var

        cache = (x_norm, gamma, inv_std)

    elif mode == "test":
        x_norm = (x - running_mean) / np.sqrt(running_var + eps)
        out = gamma * x_norm + beta
        cache = None

    else:
        raise ValueError("неверный режим batchnorm")

    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    x_norm, gamma, inv_std = cache
    n = dout.shape[0]

    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(dout * x_norm, axis=0)

    dx_norm = dout * gamma

    dx = (
        inv_std
        / n
        * (
            n * dx_norm
            - np.sum(dx_norm, axis=0)
            - x_norm * np.sum(dx_norm * x_norm, axis=0)
        )
    )

    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    p = dropout_param["p"]
    mode = dropout_param["mode"]

    if "seed" in dropout_param:
        np.random.seed(dropout_param["seed"])

    mask = None

    if mode == "train":
        mask = (np.random.rand(*x.shape) < p) / p
        out = x * mask
    elif mode == "test":
        out = x
    else:
        raise ValueError("неверный режим dropout")

    out = out.astype(x.dtype, copy=False)
    cache = (dropout_param, mask)
    return out, cache


def dropout_backward(dout, cache):
    dropout_param, mask = cache
    mode = dropout_param["mode"]

    if mode == "train":
        dx = dout * mask
    else:
        dx = dout

    return dx


def conv_forward_naive(x, w, b, conv_param):
    stride = conv_param["stride"]
    pad = conv_param["pad"]

    n, c, h, width = x.shape
    f, _, hh, ww = w.shape

    out_h = 1 + (h + 2 * pad - hh) // stride
    out_w = 1 + (width + 2 * pad - ww) // stride

    x_pad = np.pad(
        x,
        ((0, 0), (0, 0), (pad, pad), (pad, pad)),
        mode="constant",
    )

    out = np.zeros((n, f, out_h, out_w), dtype=x.dtype)

    for i in range(n):
        for j in range(f):
            for y in range(out_h):
                y0 = y * stride
                y1 = y0 + hh

                for x_pos in range(out_w):
                    x0 = x_pos * stride
                    x1 = x0 + ww

                    region = x_pad[i, :, y0:y1, x0:x1]
                    out[i, j, y, x_pos] = np.sum(region * w[j]) + b[j]

    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    x, w, b, conv_param = cache

    stride = conv_param["stride"]
    pad = conv_param["pad"]

    n, c, h, width = x.shape
    f, _, hh, ww = w.shape
    _, _, out_h, out_w = dout.shape

    x_pad = np.pad(
        x,
        ((0, 0), (0, 0), (pad, pad), (pad, pad)),
        mode="constant",
    )

    dx_pad = np.zeros_like(x_pad)
    dw = np.zeros_like(w)
    db = np.sum(dout, axis=(0, 2, 3))

    for i in range(n):
        for j in range(f):
            for y in range(out_h):
                y0 = y * stride
                y1 = y0 + hh

                for x_pos in range(out_w):
                    x0 = x_pos * stride
                    x1 = x0 + ww

                    region = x_pad[i, :, y0:y1, x0:x1]
                    value = dout[i, j, y, x_pos]

                    dw[j] += region * value
                    dx_pad[i, :, y0:y1, x0:x1] += w[j] * value

    if pad == 0:
        dx = dx_pad
    else:
        dx = dx_pad[:, :, pad:-pad, pad:-pad]

    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    pool_h = pool_param["pool_height"]
    pool_w = pool_param["pool_width"]
    stride = pool_param["stride"]

    n, c, h, width = x.shape

    out_h = 1 + (h - pool_h) // stride
    out_w = 1 + (width - pool_w) // stride

    out = np.zeros((n, c, out_h, out_w), dtype=x.dtype)

    for i in range(n):
        for j in range(c):
            for y in range(out_h):
                y0 = y * stride
                y1 = y0 + pool_h

                for x_pos in range(out_w):
                    x0 = x_pos * stride
                    x1 = x0 + pool_w

                    region = x[i, j, y0:y1, x0:x1]
                    out[i, j, y, x_pos] = np.max(region)

    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    x, pool_param = cache

    pool_h = pool_param["pool_height"]
    pool_w = pool_param["pool_width"]
    stride = pool_param["stride"]

    n, c, h, width = x.shape
    _, _, out_h, out_w = dout.shape

    dx = np.zeros_like(x)

    for i in range(n):
        for j in range(c):
            for y in range(out_h):
                y0 = y * stride
                y1 = y0 + pool_h

                for x_pos in range(out_w):
                    x0 = x_pos * stride
                    x1 = x0 + pool_w

                    region = x[i, j, y0:y1, x0:x1]
                    max_value = np.max(region)
                    mask = region == max_value

                    dx[i, j, y0:y1, x0:x1] += mask * dout[i, j, y, x_pos]

    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    n, c, h, w = x.shape
    x_new = x.transpose(0, 2, 3, 1).reshape(-1, c)

    out, cache = batchnorm_forward(x_new, gamma, beta, bn_param)
    out = out.reshape(n, h, w, c).transpose(0, 3, 1, 2)

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    n, c, h, w = dout.shape
    dout_new = dout.transpose(0, 2, 3, 1).reshape(-1, c)

    dx, dgamma, dbeta = batchnorm_backward(dout_new, cache)
    dx = dx.reshape(n, h, w, c).transpose(0, 3, 1, 2)

    return dx, dgamma, dbeta


def svm_loss(x, y):
    n = x.shape[0]
    correct = x[np.arange(n), y]
    margins = np.maximum(0, x - correct[:, None] + 1)
    margins[np.arange(n), y] = 0

    loss = np.sum(margins) / n

    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    count = np.sum(margins > 0, axis=1)
    dx[np.arange(n), y] -= count
    dx /= n

    return loss, dx


def softmax_loss(x, y):
    shifted = x - np.max(x, axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    n = x.shape[0]
    loss = -np.sum(np.log(probs[np.arange(n), y] + 1e-12)) / n

    dx = probs.copy()
    dx[np.arange(n), y] -= 1
    dx /= n

    return loss, dx
