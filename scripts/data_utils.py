from pathlib import Path
import pickle

import numpy as np


def load_pickle(file_obj):
    return pickle.load(file_obj, encoding="latin1")


def load_cifar_batch(filename):
    with open(filename, "rb") as file_obj:
        data = load_pickle(file_obj)

    x = data["data"]
    y = data["labels"]

    x = x.reshape(10000, 3, 32, 32)
    x = x.transpose(0, 2, 3, 1).astype(np.float32)

    y = np.array(y)

    return x, y


def load_cifar10(root):
    root = Path(root)

    xs = []
    ys = []

    for batch in range(1, 6):
        filename = root / ("data_batch_%d" % batch)
        x, y = load_cifar_batch(filename)

        xs.append(x)
        ys.append(y)

    x_train = np.concatenate(xs)
    y_train = np.concatenate(ys)

    x_test, y_test = load_cifar_batch(root / "test_batch")

    return x_train, y_train, x_test, y_test


def find_cifar10_dir():
    candidates = [
        Path("scripts/datasets/cifar-10-batches-py"),
        Path("scripts/datasets"),
        Path("../lab_1-2/scripts/datasets/cifar-10-batches-py"),
        Path("lab_1-2/scripts/datasets/cifar-10-batches-py"),
        Path("../scripts/datasets/cifar-10-batches-py"),
    ]

    for path in candidates:
        if (path / "data_batch_1").exists():
            return path

    raise FileNotFoundError(
        "папка cifar 10 не найдена"
    )


def get_cifar10_data(
    cifar10_dir=None,
    num_training=49000,
    num_validation=1000,
    num_test=1000,
    subtract_mean=True,
):
    if cifar10_dir is None:
        cifar10_dir = find_cifar10_dir()

    x_train, y_train, x_test, y_test = load_cifar10(
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

        x_train = x_train - mean_image
        x_val = x_val - mean_image
        x_test = x_test - mean_image

    x_train = x_train.transpose(0, 3, 1, 2).copy()
    x_val = x_val.transpose(0, 3, 1, 2).copy()
    x_test = x_test.transpose(0, 3, 1, 2).copy()

    return {
        "x_train": x_train,
        "y_train": y_train,
        "x_val": x_val,
        "y_val": y_val,
        "x_test": x_test,
        "y_test": y_test,
    }
