import numpy as np


def sgd(w, dw, config=None):
    if config is None:
        config = {}

    config.setdefault("learning_rate", 1e-2)

    next_w = w - config["learning_rate"] * dw
    return next_w, config


def sgd_momentum(w, dw, config=None):
    if config is None:
        config = {}

    config.setdefault("learning_rate", 1e-2)
    config.setdefault("momentum", 0.9)

    v = config.get("velocity", np.zeros_like(w))

    v = config["momentum"] * v - config["learning_rate"] * dw
    next_w = w + v

    config["velocity"] = v

    return next_w, config


def rmsprop(w, dw, config=None):
    if config is None:
        config = {}

    config.setdefault("learning_rate", 1e-2)
    config.setdefault("decay_rate", 0.99)
    config.setdefault("epsilon", 1e-8)
    config.setdefault("cache", np.zeros_like(w))

    cache = config["cache"]
    decay = config["decay_rate"]

    cache = decay * cache + (1 - decay) * (dw ** 2)

    next_w = w - config["learning_rate"] * dw / (
        np.sqrt(cache) + config["epsilon"]
    )

    config["cache"] = cache

    return next_w, config


def adam(w, dw, config=None):
    if config is None:
        config = {}

    config.setdefault("learning_rate", 1e-3)
    config.setdefault("beta1", 0.9)
    config.setdefault("beta2", 0.999)
    config.setdefault("epsilon", 1e-8)
    config.setdefault("m", np.zeros_like(w))
    config.setdefault("v", np.zeros_like(w))
    config.setdefault("t", 0)

    beta1 = config["beta1"]
    beta2 = config["beta2"]

    config["t"] += 1

    config["m"] = beta1 * config["m"] + (1 - beta1) * dw
    config["v"] = beta2 * config["v"] + (1 - beta2) * (dw ** 2)

    m_corr = config["m"] / (1 - beta1 ** config["t"])
    v_corr = config["v"] / (1 - beta2 ** config["t"])

    next_w = w - config["learning_rate"] * m_corr / (
        np.sqrt(v_corr) + config["epsilon"]
    )

    return next_w, config
