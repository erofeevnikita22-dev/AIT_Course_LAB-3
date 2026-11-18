import os
import pickle
import numpy as np


def load_pickle(file_obj):
    return pickle.load(file_obj, encoding="latin1")


def load_CIFAR_batch(filename):
    with open(filename, "rb") as file_obj:
        datadict = load_pickle(file_obj)

    x = datadict["data"]
    y = datadict["labels"]

    x = x.reshape(
        10000,
        3,
        32,
        32
    ).transpose(0, 2, 3, 1).astype(float)

    y = np.array(y)

    return x, y


def load_CIFAR10(root):
    xs = []
    ys = []

    for batch in range(1, 6):
        filename = os.path.join(
            root,
            "data_batch_%d" % batch
        )

        x, y = load_CIFAR_batch(filename)

        xs.append(x)
        ys.append(y)

    x_train = np.concatenate(xs)
    y_train = np.concatenate(ys)

    x_test, y_test = load_CIFAR_batch(
        os.path.join(root, "test_batch")
    )

    return x_train, y_train, x_test, y_test


def find_cifar10_dir():
    base = os.path.join(
        os.path.dirname(__file__),
        "datasets"
    )

    candidates = [
        os.path.join(base, "cifar-10-batches-py"),
        base
    ]

    for path in candidates:
        if os.path.isfile(
            os.path.join(path, "data_batch_1")
        ):
            return path

    raise FileNotFoundError(
        "папка cifar 10 не найдена"
    )


def get_CIFAR10_data(
    num_training=49000,
    num_validation=1000,
    num_test=1000,
    subtract_mean=True
):
    cifar10_dir = find_cifar10_dir()

    x_train, y_train, x_test, y_test = load_CIFAR10(
        cifar10_dir
    )

    x_val = x_train[
        num_training:num_training + num_validation
    ]
    y_val = y_train[
        num_training:num_training + num_validation
    ]

    x_train = x_train[:num_training]
    y_train = y_train[:num_training]

    x_test = x_test[:num_test]
    y_test = y_test[:num_test]

    if subtract_mean:
        mean_image = np.mean(x_train, axis=0)

        x_train -= mean_image
        x_val -= mean_image
        x_test -= mean_image

    x_train = x_train.transpose(0, 3, 1, 2).copy()
    x_val = x_val.transpose(0, 3, 1, 2).copy()
    x_test = x_test.transpose(0, 3, 1, 2).copy()

    return {
        "X_train": x_train,
        "y_train": y_train,
        "X_val": x_val,
        "y_val": y_val,
        "X_test": x_test,
        "y_test": y_test
    }
