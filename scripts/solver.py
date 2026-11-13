import numpy as np

from scripts import optim


class Solver:
    def __init__(self, model, data, **kwargs):
        self.model = model

        self.x_train = data["x_train"]
        self.y_train = data["y_train"]
        self.x_val = data["x_val"]
        self.y_val = data["y_val"]

        self.update_rule_name = kwargs.pop(
            "update_rule",
            "sgd",
        )
        self.optim_config = kwargs.pop(
            "optim_config",
            {},
        )
        self.lr_decay = kwargs.pop(
            "lr_decay",
            1.0,
        )
        self.batch_size = kwargs.pop(
            "batch_size",
            100,
        )
        self.num_epochs = kwargs.pop(
            "num_epochs",
            10,
        )
        self.print_every = kwargs.pop(
            "print_every",
            100,
        )
        self.verbose = kwargs.pop(
            "verbose",
            True,
        )
        self.num_train_samples = kwargs.pop(
            "num_train_samples",
            1000,
        )
        self.num_val_samples = kwargs.pop(
            "num_val_samples",
            None,
        )

        if kwargs:
            raise ValueError("есть неизвестные параметры solver")

        self.update_rule = getattr(
            optim,
            self.update_rule_name,
        )

        self._reset()

    def _reset(self):
        self.epoch = 0
        self.best_val_acc = 0
        self.best_params = {}

        self.loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []

        self.optim_configs = {}

        for name in self.model.params:
            self.optim_configs[name] = dict(
                self.optim_config
            )

    def _step(self):
        num_train = self.x_train.shape[0]

        indexes = np.random.choice(
            num_train,
            self.batch_size,
            replace=True,
        )

        x_batch = self.x_train[indexes]
        y_batch = self.y_train[indexes]

        loss, grads = self.model.loss(
            x_batch,
            y_batch,
        )

        self.loss_history.append(loss)

        for name in self.model.params:
            w = self.model.params[name]
            dw = grads[name]
            config = self.optim_configs[name]

            next_w, next_config = self.update_rule(
                w,
                dw,
                config,
            )

            self.model.params[name] = next_w
            self.optim_configs[name] = next_config

    def check_accuracy(
        self,
        x,
        y,
        num_samples=None,
        batch_size=200,
    ):
        n = x.shape[0]

        if num_samples is not None and n > num_samples:
            indexes = np.random.choice(
                n,
                num_samples,
                replace=False,
            )

            x = x[indexes]
            y = y[indexes]
            n = num_samples

        predictions = []

        for start in range(0, n, batch_size):
            end = start + batch_size

            scores = self.model.loss(
                x[start:end]
            )

            predictions.append(
                np.argmax(scores, axis=1)
            )

        predictions = np.hstack(predictions)

        return np.mean(predictions == y)

    def train(self):
        num_train = self.x_train.shape[0]

        iterations_per_epoch = max(
            num_train // self.batch_size,
            1,
        )

        num_iterations = (
            self.num_epochs
            * iterations_per_epoch
        )

        for iteration in range(num_iterations):
            self._step()

            if (
                self.verbose
                and iteration % self.print_every == 0
            ):
                print(
                    "итерация",
                    iteration + 1,
                    "из",
                    num_iterations,
                    "loss",
                    round(self.loss_history[-1], 6),
                )

            epoch_end = (
                (iteration + 1)
                % iterations_per_epoch
                == 0
            )

            if epoch_end:
                self.epoch += 1

                for name in self.optim_configs:
                    self.optim_configs[name]["learning_rate"] *= (
                        self.lr_decay
                    )

            first_iteration = iteration == 0
            last_iteration = iteration == num_iterations - 1

            if first_iteration or last_iteration or epoch_end:
                train_acc = self.check_accuracy(
                    self.x_train,
                    self.y_train,
                    num_samples=self.num_train_samples,
                )

                val_acc = self.check_accuracy(
                    self.x_val,
                    self.y_val,
                    num_samples=self.num_val_samples,
                )

                self.train_acc_history.append(
                    train_acc
                )
                self.val_acc_history.append(
                    val_acc
                )

                if self.verbose:
                    print(
                        "эпоха",
                        self.epoch,
                        "из",
                        self.num_epochs,
                        "точность train",
                        round(train_acc, 4),
                        "точность val",
                        round(val_acc, 4),
                    )

                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.best_params = {
                        name: value.copy()
                        for name, value in self.model.params.items()
                    }

        if self.best_params:
            self.model.params = self.best_params
