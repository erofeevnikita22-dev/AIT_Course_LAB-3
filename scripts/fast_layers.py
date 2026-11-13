import numpy as np


def _get_indices(x_shape, field_h, field_w, pad, stride):
    n, c, h, w = x_shape

    out_h = (h + 2 * pad - field_h) // stride + 1
    out_w = (w + 2 * pad - field_w) // stride + 1

    i0 = np.repeat(np.arange(field_h), field_w)
    i0 = np.tile(i0, c)

    i1 = stride * np.repeat(np.arange(out_h), out_w)

    j0 = np.tile(np.arange(field_w), field_h * c)
    j1 = stride * np.tile(np.arange(out_w), out_h)

    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(c), field_h * field_w).reshape(-1, 1)

    return k, i, j


def _im2col(x, field_h, field_w, pad, stride):
    x_pad = np.pad(
        x,
        ((0, 0), (0, 0), (pad, pad), (pad, pad)),
        mode="constant",
    )

    k, i, j = _get_indices(x.shape, field_h, field_w, pad, stride)
    cols = x_pad[:, k, i, j]
    cols = cols.transpose(1, 2, 0).reshape(field_h * field_w * x.shape[1], -1)

    return cols


def _col2im(cols, x_shape, field_h, field_w, pad, stride):
    n, c, h, w = x_shape

    x_pad = np.zeros(
        (n, c, h + 2 * pad, w + 2 * pad),
        dtype=cols.dtype,
    )

    k, i, j = _get_indices(x_shape, field_h, field_w, pad, stride)
    cols_new = cols.reshape(c * field_h * field_w, -1, n)
    cols_new = cols_new.transpose(2, 0, 1)

    np.add.at(x_pad, (slice(None), k, i, j), cols_new)

    if pad == 0:
        return x_pad

    return x_pad[:, :, pad:-pad, pad:-pad]


def conv_forward_fast(x, w, b, conv_param):
    stride = conv_param["stride"]
    pad = conv_param["pad"]

    n, c, h, width = x.shape
    f, _, hh, ww = w.shape

    out_h = 1 + (h + 2 * pad - hh) // stride
    out_w = 1 + (width + 2 * pad - ww) // stride

    x_cols = _im2col(x, hh, ww, pad, stride)
    result = w.reshape(f, -1).dot(x_cols) + b.reshape(-1, 1)

    out = result.reshape(f, out_h, out_w, n)
    out = out.transpose(3, 0, 1, 2)

    cache = (x, w, b, conv_param, x_cols)
    return out, cache


def conv_backward_fast(dout, cache):
    x, w, b, conv_param, x_cols = cache

    stride = conv_param["stride"]
    pad = conv_param["pad"]

    f, c, hh, ww = w.shape

    dout_new = dout.transpose(1, 2, 3, 0).reshape(f, -1)

    db = np.sum(dout, axis=(0, 2, 3))
    dw = dout_new.dot(x_cols.T).reshape(w.shape)

    dx_cols = w.reshape(f, -1).T.dot(dout_new)
    dx = _col2im(dx_cols, x.shape, hh, ww, pad, stride)

    return dx, dw, db


def max_pool_forward_fast(x, pool_param):
    pool_h = pool_param["pool_height"]
    pool_w = pool_param["pool_width"]
    stride = pool_param["stride"]

    n, c, h, w = x.shape

    if pool_h == pool_w == stride and h % pool_h == 0 and w % pool_w == 0:
        x_new = x.reshape(
            n,
            c,
            h // pool_h,
            pool_h,
            w // pool_w,
            pool_w,
        )

        out = x_new.max(axis=3).max(axis=4)
        cache = ("reshape", x, x_new, out, pool_param)
        return out, cache

    x_new = x.reshape(n * c, 1, h, w)
    cols = _im2col(x_new, pool_h, pool_w, 0, stride)
    max_idx = np.argmax(cols, axis=0)
    out = cols[max_idx, np.arange(max_idx.size)]

    out_h = 1 + (h - pool_h) // stride
    out_w = 1 + (w - pool_w) // stride

    out = out.reshape(out_h, out_w, n, c).transpose(2, 3, 0, 1)
    cache = ("im2col", x, cols, max_idx, pool_param)

    return out, cache


def max_pool_backward_fast(dout, cache):
    method = cache[0]

    if method == "reshape":
        _, x, x_new, out, pool_param = cache

        mask = x_new == out[:, :, :, None, :, None]
        dout_new = dout[:, :, :, None, :, None]

        dx_new = mask * dout_new

        count = np.sum(mask, axis=(3, 5), keepdims=True)
        dx_new = dx_new / count

        return dx_new.reshape(x.shape)

    _, x, cols, max_idx, pool_param = cache

    pool_h = pool_param["pool_height"]
    pool_w = pool_param["pool_width"]
    stride = pool_param["stride"]

    n, c, h, w = x.shape

    dx_cols = np.zeros_like(cols)
    dout_new = dout.transpose(2, 3, 0, 1).ravel()
    dx_cols[max_idx, np.arange(max_idx.size)] = dout_new

    dx = _col2im(
        dx_cols,
        (n * c, 1, h, w),
        pool_h,
        pool_w,
        0,
        stride,
    )

    return dx.reshape(x.shape)
