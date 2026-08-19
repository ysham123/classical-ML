"""A multilayer perceptron in NumPy, with backpropagation derived by hand.

One hidden layer, sigmoid activations, mean squared error, minibatch gradient descent.
No autograd is involved: ``NeuralNetMLP.backward`` applies the chain rule explicitly,
and ``tests/test_algorithms_neural_net.py`` checks every one of its gradients against
central finite differences.
"""

from __future__ import annotations

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -250, 250)))


def int_to_onehot(y: np.ndarray, num_labels: int) -> np.ndarray:
    """One-hot encode integer labels into a (n_examples, num_labels) matrix."""
    ary = np.zeros((y.shape[0], num_labels))
    for index, value in enumerate(y):
        ary[index, value] = 1
    return ary


class NeuralNetMLP:
    """A one-hidden-layer perceptron with sigmoid activations and an MSE loss.

    Args:
        num_features: Input dimensionality (784 for MNIST).
        num_hidden: Units in the hidden layer.
        num_classes: Output units, one per class.
        random_seed: Seed for the weight initialisation.
    """

    def __init__(self, num_features: int, num_hidden: int, num_classes: int, random_seed: int = 123) -> None:
        self.num_classes = num_classes
        rng = np.random.RandomState(random_seed)

        self.weight_h = rng.normal(loc=0.0, scale=0.1, size=(num_hidden, num_features))
        self.bias_h = np.zeros(num_hidden)
        self.weight_out = rng.normal(loc=0.0, scale=0.1, size=(num_classes, num_hidden))
        self.bias_out = np.zeros(num_classes)

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Propagate a minibatch through both layers."""
        z_h = np.dot(x, self.weight_h.T) + self.bias_h
        a_h = sigmoid(z_h)
        z_out = np.dot(a_h, self.weight_out.T) + self.bias_out
        a_out = sigmoid(z_out)
        return a_h, a_out

    def backward(self, x, a_h, a_out, y):
        """Backpropagate the squared error, layer by layer.

        The chain rule is applied twice: once from the loss to the output weights, and
        once more through the hidden activation to the input weights. Every term below
        is written separately so it lines up with the derivation in the chapter.
        """
        y_onehot = int_to_onehot(y, self.num_classes)

        # Output layer: dLoss/dOutWeight = dLoss/dOutAct * dOutAct/dOutNet * dOutNet/dOutWeight
        d_loss__d_a_out = 2.0 * (a_out - y_onehot) / y.shape[0]
        d_a_out__d_z_out = a_out * (1.0 - a_out)  # derivative of the sigmoid
        delta_out = d_loss__d_a_out * d_a_out__d_z_out

        d_loss__dw_out = np.dot(delta_out.T, a_h)
        d_loss__db_out = np.sum(delta_out, axis=0)

        # Hidden layer: the same product, continued back through the output weights.
        d_z_out__a_h = self.weight_out
        d_loss__a_h = np.dot(delta_out, d_z_out__a_h)
        d_a_h__d_z_h = a_h * (1.0 - a_h)
        d_z_h__d_w_h = x

        d_loss__d_w_h = np.dot((d_loss__a_h * d_a_h__d_z_h).T, d_z_h__d_w_h)
        d_loss__d_b_h = np.sum(d_loss__a_h * d_a_h__d_z_h, axis=0)

        return d_loss__dw_out, d_loss__db_out, d_loss__d_w_h, d_loss__d_b_h


def minibatch_generator(X: np.ndarray, y: np.ndarray, minibatch_size: int, seed: int = 123):
    """Yield shuffled minibatches, reshuffling the indices on every call."""
    indices = np.arange(X.shape[0])
    np.random.RandomState(seed).shuffle(indices)
    for start in range(0, indices.shape[0] - minibatch_size + 1, minibatch_size):
        batch_idx = indices[start : start + minibatch_size]
        yield X[batch_idx], y[batch_idx]


def mse_loss(targets: np.ndarray, probas: np.ndarray, num_labels: int = 10) -> float:
    onehot_targets = int_to_onehot(targets, num_labels)
    return float(np.mean((onehot_targets - probas) ** 2))


def accuracy(targets: np.ndarray, predicted_labels: np.ndarray) -> float:
    return float(np.mean(predicted_labels == targets))


def compute_mse_and_acc(nnet: NeuralNetMLP, X, y, num_labels: int = 10, minibatch_size: int = 100):
    """Evaluate loss and accuracy in minibatches so memory stays flat."""
    mse, correct_pred, num_examples, num_batches = 0.0, 0, 0, 0
    for features, targets in minibatch_generator(X, y, minibatch_size):
        _, probas = nnet.forward(features)
        predicted_labels = np.argmax(probas, axis=1)
        mse += mse_loss(targets, probas, num_labels)
        correct_pred += (predicted_labels == targets).sum()
        num_examples += targets.shape[0]
        num_batches += 1
    return mse / num_batches, correct_pred / num_examples


def train(model: NeuralNetMLP, X_train, y_train, X_valid, y_valid, num_epochs: int,
          learning_rate: float = 0.1, minibatch_size: int = 100, verbose: bool = True):
    """Minibatch gradient descent over ``num_epochs`` passes."""
    epoch_loss, epoch_train_acc, epoch_valid_acc = [], [], []

    for epoch in range(num_epochs):
        for X_batch, y_batch in minibatch_generator(X_train, y_train, minibatch_size, seed=123 + epoch):
            a_h, a_out = model.forward(X_batch)
            d_w_out, d_b_out, d_w_h, d_b_h = model.backward(X_batch, a_h, a_out, y_batch)

            model.weight_h -= learning_rate * d_w_h
            model.bias_h -= learning_rate * d_b_h
            model.weight_out -= learning_rate * d_w_out
            model.bias_out -= learning_rate * d_b_out

        train_mse, train_acc = compute_mse_and_acc(model, X_train, y_train,
                                                   num_labels=model.num_classes,
                                                   minibatch_size=minibatch_size)
        _, valid_acc = compute_mse_and_acc(model, X_valid, y_valid,
                                           num_labels=model.num_classes,
                                           minibatch_size=minibatch_size)
        epoch_loss.append(train_mse)
        epoch_train_acc.append(train_acc)
        epoch_valid_acc.append(valid_acc)

        if verbose and ((epoch + 1) % 10 == 0 or epoch == 0):
            print(f"    epoch {epoch + 1:>3}/{num_epochs} | train MSE {train_mse:.4f} | "
                  f"train acc {train_acc * 100:.2f}% | valid acc {valid_acc * 100:.2f}%")

    return epoch_loss, epoch_train_acc, epoch_valid_acc
