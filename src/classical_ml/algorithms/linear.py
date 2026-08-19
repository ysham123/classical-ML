"""Linear models written from scratch with NumPy.

Four of these are the same algorithm with one thing changed each time, which is the
clearest way to see what each piece of a linear model actually does. The perceptron
updates on a thresholded prediction; Adaline swaps that threshold for the identity
activation, which makes the loss differentiable; logistic regression swaps the identity
for a sigmoid and squared error for the log likelihood; linear regression drops the
threshold from the prediction entirely.

Every one of them is checked against its scikit-learn counterpart in ``tests/``.
"""

from __future__ import annotations

import numpy as np


class Perceptron:
    """Rosenblatt's perceptron, updated once per training example.

    Args:
        eta: Learning rate in (0.0, 1.0].
        n_iter: Passes over the training set.
        random_state: Seed for the small random weight initialisation.

    Attributes:
        w_: Weights after fitting.
        b_: Bias unit after fitting.
        errors_: Number of misclassifications in each epoch.
    """

    def __init__(self, eta: float = 0.01, n_iter: int = 50, random_state: int = 1) -> None:
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> Perceptron:
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.float64(0.0)
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_ += update * xi
                self.b_ += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X: np.ndarray) -> np.ndarray:
        return np.dot(X, self.w_) + self.b_

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.where(self.net_input(X) >= 0.0, 1, 0)


class AdalineGD:
    """Adaptive linear neuron trained by full-batch gradient descent.

    The weight update uses the identity activation rather than the step function, so
    the mean squared error loss is differentiable and every epoch uses the whole
    training set at once.

    Attributes:
        losses_: Mean squared error after each epoch.
    """

    def __init__(self, eta: float = 0.01, n_iter: int = 50, random_state: int = 1) -> None:
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> AdalineGD:
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.float64(0.0)
        self.losses_ = []

        for _ in range(self.n_iter):
            output = self.activation(self.net_input(X))
            errors = y - output
            self.w_ += self.eta * 2.0 * X.T.dot(errors) / X.shape[0]
            self.b_ += self.eta * 2.0 * errors.mean()
            self.losses_.append((errors**2).mean())
        return self

    def net_input(self, X: np.ndarray) -> np.ndarray:
        return np.dot(X, self.w_) + self.b_

    def activation(self, X: np.ndarray) -> np.ndarray:
        """Identity activation, kept explicit to mirror the network diagram."""
        return X

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)


class AdalineSGD:
    """Adaline trained by stochastic gradient descent, one example at a time.

    Reaches a good solution in far fewer passes than full-batch descent and supports
    ``partial_fit`` for streaming data, at the cost of a noisier loss curve.

    Args:
        shuffle: Reshuffle the training set every epoch to avoid cycles.
    """

    def __init__(
        self,
        eta: float = 0.01,
        n_iter: int = 10,
        shuffle: bool = True,
        random_state: int | None = None,
    ) -> None:
        self.eta = eta
        self.n_iter = n_iter
        self.shuffle = shuffle
        self.random_state = random_state
        self.w_initialized = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> AdalineSGD:
        self._initialize_weights(X.shape[1])
        self.losses_ = []
        for _ in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            losses = [self._update_weights(xi, target) for xi, target in zip(X, y)]
            self.losses_.append(np.mean(losses))
        return self

    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> AdalineSGD:
        """Update the model on new data without reinitialising the weights."""
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)
        return self

    def _shuffle(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        r = self.rgen.permutation(len(y))
        return X[r], y[r]

    def _initialize_weights(self, m: int) -> None:
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size=m)
        self.b_ = np.float64(0.0)
        self.w_initialized = True

    def _update_weights(self, xi: np.ndarray, target: float) -> float:
        output = self.activation(self.net_input(xi))
        error = target - output
        self.w_ += self.eta * 2.0 * xi * error
        self.b_ += self.eta * 2.0 * error
        return error**2

    def net_input(self, X: np.ndarray) -> np.ndarray:
        return np.dot(X, self.w_) + self.b_

    def activation(self, X: np.ndarray) -> np.ndarray:
        return X

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)


class LogisticRegressionGD:
    """Logistic regression trained by full-batch gradient descent.

    Structurally identical to ``AdalineGD`` from chapter 2: the only changes are the
    sigmoid activation and the log-likelihood loss in place of the identity activation
    and mean squared error.
    """

    def __init__(self, eta: float = 0.01, n_iter: int = 50, random_state: int = 1) -> None:
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> LogisticRegressionGD:
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.float64(0.0)
        self.losses_ = []

        for _ in range(self.n_iter):
            output = self.activation(self.net_input(X))
            errors = y - output
            self.w_ += self.eta * X.T.dot(errors) / X.shape[0]
            self.b_ += self.eta * errors.mean()
            loss = (-y.dot(np.log(output)) - (1 - y).dot(np.log(1 - output))) / X.shape[0]
            self.losses_.append(loss)
        return self

    def net_input(self, X: np.ndarray) -> np.ndarray:
        return np.dot(X, self.w_) + self.b_

    def activation(self, z: np.ndarray) -> np.ndarray:
        """Logistic sigmoid, clipped to keep the log loss finite."""
        return 1.0 / (1.0 + np.exp(-np.clip(z, -250, 250)))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)


class LinearRegressionGD:
    """Ordinary least squares fitted by full-batch gradient descent.

    The same update as ``AdalineGD`` in chapter 2 without the threshold, which is the
    point the chapter makes: Adaline was already a linear regression with a classifier
    bolted on the end.
    """

    def __init__(self, eta: float = 0.01, n_iter: int = 50, random_state: int = 1) -> None:
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> LinearRegressionGD:
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.array([0.0])
        self.losses_ = []

        for _ in range(self.n_iter):
            output = self.net_input(X)
            errors = y - output
            self.w_ += self.eta * 2.0 * X.T.dot(errors) / X.shape[0]
            self.b_ += self.eta * 2.0 * errors.mean()
            self.losses_.append((errors**2).mean())
        return self

    def net_input(self, X: np.ndarray) -> np.ndarray:
        return np.dot(X, self.w_) + self.b_

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.net_input(X)
