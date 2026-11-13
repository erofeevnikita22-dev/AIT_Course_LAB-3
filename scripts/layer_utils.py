from .layers import affine_forward
from .layers import affine_backward
from .layers import relu_forward
from .layers import relu_backward
from .layers import spatial_batchnorm_forward
from .layers import spatial_batchnorm_backward
from .fast_layers import conv_forward_fast
from .fast_layers import conv_backward_fast
from .fast_layers import max_pool_forward_fast
from .fast_layers import max_pool_backward_fast


def affine_relu_forward(x, w, b):
    a, affine_cache = affine_forward(x, w, b)
    out, relu_cache = relu_forward(a)
    cache = (affine_cache, relu_cache)
    return out, cache


def affine_relu_backward(dout, cache):
    affine_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = affine_backward(da, affine_cache)
    return dx, dw, db


def conv_relu_forward(x, w, b, conv_param):
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    out, relu_cache = relu_forward(a)
    cache = (conv_cache, relu_cache)
    return out, cache


def conv_relu_backward(dout, cache):
    conv_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db


def conv_relu_pool_forward(x, w, b, conv_param, pool_param):
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    s, relu_cache = relu_forward(a)
    out, pool_cache = max_pool_forward_fast(s, pool_param)
    cache = (conv_cache, relu_cache, pool_cache)
    return out, cache


def conv_relu_pool_backward(dout, cache):
    conv_cache, relu_cache, pool_cache = cache

    ds = max_pool_backward_fast(dout, pool_cache)
    da = relu_backward(ds, relu_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)

    return dx, dw, db


def conv_bn_relu_forward(x, w, b, gamma, beta, conv_param, bn_param):
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    a_norm, bn_cache = spatial_batchnorm_forward(a, gamma, beta, bn_param)
    out, relu_cache = relu_forward(a_norm)

    cache = (conv_cache, bn_cache, relu_cache)
    return out, cache


def conv_bn_relu_backward(dout, cache):
    conv_cache, bn_cache, relu_cache = cache

    da_norm = relu_backward(dout, relu_cache)
    da, dgamma, dbeta = spatial_batchnorm_backward(da_norm, bn_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)

    return dx, dw, db, dgamma, dbeta
