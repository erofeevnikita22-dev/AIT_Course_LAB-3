import numpy as np

from ..layers import affine_forward
from ..layers import affine_backward
from ..layers import softmax_loss
from ..layer_utils import affine_relu_forward
from ..layer_utils import affine_relu_backward
from ..layer_utils import conv_relu_pool_forward
from ..layer_utils import conv_relu_pool_backward


class ThreeLayerConvNet:
    def __init__(
        self,
        input_dim=(3, 32, 32),
        num_filters=32,
        filter_size=7,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
        dtype=np.float32,
    ):
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        c, h, w = input_dim

        self.params["W1"] = weight_scale * np.random.randn(
            num_filters,
            c,
            filter_size,
            filter_size,
        )
        self.params["b1"] = np.zeros(num_filters)

        pool_h = h // 2
        pool_w = w // 2

        self.params["W2"] = weight_scale * np.random.randn(
            num_filters * pool_h * pool_w,
            hidden_dim,
        )
        self.params["b2"] = np.zeros(hidden_dim)

        self.params["W3"] = weight_scale * np.random.randn(
            hidden_dim,
            num_classes,
        )
        self.params["b3"] = np.zeros(num_classes)

        for key in self.params:
            self.params[key] = self.params[key].astype(dtype)

    def loss(self, x, y=None):
        w1 = self.params["W1"]
        b1 = self.params["b1"]
        w2 = self.params["W2"]
        b2 = self.params["b2"]
        w3 = self.params["W3"]
        b3 = self.params["b3"]

        filter_size = w1.shape[2]

        conv_param = {
            "stride": 1,
            "pad": (filter_size - 1) // 2,
        }

        pool_param = {
            "pool_height": 2,
            "pool_width": 2,
            "stride": 2,
        }

        out1, cache1 = conv_relu_pool_forward(
            x,
            w1,
            b1,
            conv_param,
            pool_param,
        )

        out2, cache2 = affine_relu_forward(
            out1,
            w2,
            b2,
        )

        scores, cache3 = affine_forward(
            out2,
            w3,
            b3,
        )

        if y is None:
            return scores

        loss, dscores = softmax_loss(scores, y)

        loss += 0.5 * self.reg * (
            np.sum(w1 * w1)
            + np.sum(w2 * w2)
            + np.sum(w3 * w3)
        )

        dout2, dw3, db3 = affine_backward(
            dscores,
            cache3,
        )

        dout1, dw2, db2 = affine_relu_backward(
            dout2,
            cache2,
        )

        dx, dw1, db1 = conv_relu_pool_backward(
            dout1,
            cache1,
        )

        grads = {}
        grads["W1"] = dw1 + self.reg * w1
        grads["b1"] = db1
        grads["W2"] = dw2 + self.reg * w2
        grads["b2"] = db2
        grads["W3"] = dw3 + self.reg * w3
        grads["b3"] = db3

        return loss, grads
