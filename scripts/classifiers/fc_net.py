import numpy as np

from ..layers import affine_forward
from ..layers import affine_backward
from ..layers import relu_forward
from ..layers import relu_backward
from ..layers import batchnorm_forward
from ..layers import batchnorm_backward
from ..layers import dropout_forward
from ..layers import dropout_backward
from ..layers import softmax_loss
from ..layer_utils import affine_relu_forward
from ..layer_utils import affine_relu_backward


class TwoLayerNet:
    def __init__(
        self,
        input_dim=3 * 32 * 32,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
    ):
        self.params = {}
        self.reg = reg

        self.params["W1"] = weight_scale * np.random.randn(
            input_dim,
            hidden_dim,
        )
        self.params["b1"] = np.zeros(hidden_dim)

        self.params["W2"] = weight_scale * np.random.randn(
            hidden_dim,
            num_classes,
        )
        self.params["b2"] = np.zeros(num_classes)

    def loss(self, x, y=None):
        w1 = self.params["W1"]
        b1 = self.params["b1"]
        w2 = self.params["W2"]
        b2 = self.params["b2"]

        hidden, cache1 = affine_relu_forward(x, w1, b1)
        scores, cache2 = affine_forward(hidden, w2, b2)

        if y is None:
            return scores

        loss, dscores = softmax_loss(scores, y)
        loss += 0.5 * self.reg * (
            np.sum(w1 * w1)
            + np.sum(w2 * w2)
        )

        dhidden, dw2, db2 = affine_backward(dscores, cache2)
        dx, dw1, db1 = affine_relu_backward(dhidden, cache1)

        grads = {}
        grads["W1"] = dw1 + self.reg * w1
        grads["b1"] = db1
        grads["W2"] = dw2 + self.reg * w2
        grads["b2"] = db2

        return loss, grads


class FullyConnectedNet:
    def __init__(
        self,
        hidden_dims,
        input_dim=3 * 32 * 32,
        num_classes=10,
        dropout=1,
        normalization=None,
        reg=0.0,
        weight_scale=1e-2,
        dtype=np.float32,
        seed=None,
    ):
        self.normalization = normalization
        self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        dims = [input_dim] + list(hidden_dims) + [num_classes]

        for i in range(1, self.num_layers + 1):
            self.params["W" + str(i)] = weight_scale * np.random.randn(
                dims[i - 1],
                dims[i],
            )
            self.params["b" + str(i)] = np.zeros(dims[i])

            if normalization == "batchnorm" and i < self.num_layers:
                self.params["gamma" + str(i)] = np.ones(dims[i])
                self.params["beta" + str(i)] = np.zeros(dims[i])

        self.dropout_param = {}

        if self.use_dropout:
            self.dropout_param = {
                "mode": "train",
                "p": dropout,
            }

            if seed is not None:
                self.dropout_param["seed"] = seed

        self.bn_params = []

        if self.normalization == "batchnorm":
            self.bn_params = [
                {"mode": "train"}
                for _ in range(self.num_layers - 1)
            ]

        for key in self.params:
            self.params[key] = self.params[key].astype(dtype)

    def loss(self, x, y=None):
        x = x.astype(self.dtype)
        mode = "test" if y is None else "train"

        if self.use_dropout:
            self.dropout_param["mode"] = mode

        if self.normalization == "batchnorm":
            for bn_param in self.bn_params:
                bn_param["mode"] = mode

        out = x
        caches = []

        for i in range(1, self.num_layers):
            w = self.params["W" + str(i)]
            b = self.params["b" + str(i)]

            out, affine_cache = affine_forward(out, w, b)

            bn_cache = None
            if self.normalization == "batchnorm":
                gamma = self.params["gamma" + str(i)]
                beta = self.params["beta" + str(i)]

                out, bn_cache = batchnorm_forward(
                    out,
                    gamma,
                    beta,
                    self.bn_params[i - 1],
                )

            out, relu_cache = relu_forward(out)

            dropout_cache = None
            if self.use_dropout:
                out, dropout_cache = dropout_forward(
                    out,
                    self.dropout_param,
                )

            caches.append(
                (
                    affine_cache,
                    bn_cache,
                    relu_cache,
                    dropout_cache,
                )
            )

        w_last = self.params["W" + str(self.num_layers)]
        b_last = self.params["b" + str(self.num_layers)]

        scores, final_cache = affine_forward(
            out,
            w_last,
            b_last,
        )

        if mode == "test":
            return scores

        loss, dout = softmax_loss(scores, y)

        for i in range(1, self.num_layers + 1):
            w = self.params["W" + str(i)]
            loss += 0.5 * self.reg * np.sum(w * w)

        grads = {}

        dout, dw, db = affine_backward(dout, final_cache)

        grads["W" + str(self.num_layers)] = (
            dw + self.reg * w_last
        )
        grads["b" + str(self.num_layers)] = db

        for i in range(self.num_layers - 1, 0, -1):
            (
                affine_cache,
                bn_cache,
                relu_cache,
                dropout_cache,
            ) = caches[i - 1]

            if self.use_dropout:
                dout = dropout_backward(
                    dout,
                    dropout_cache,
                )

            dout = relu_backward(
                dout,
                relu_cache,
            )

            if self.normalization == "batchnorm":
                dout, dgamma, dbeta = batchnorm_backward(
                    dout,
                    bn_cache,
                )

                grads["gamma" + str(i)] = dgamma
                grads["beta" + str(i)] = dbeta

            dout, dw, db = affine_backward(
                dout,
                affine_cache,
            )

            w = self.params["W" + str(i)]

            grads["W" + str(i)] = dw + self.reg * w
            grads["b" + str(i)] = db

        return loss, grads
