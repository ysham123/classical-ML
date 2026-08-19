"""The hand-derived backprop gradients are checked against finite differences."""

from __future__ import annotations

import numpy as np
import pytest

from classical_ml.algorithms import (
    NeuralNetMLP,
    compute_mse_and_acc,
    int_to_onehot,
    minibatch_generator,
    mse_loss,
    train,
)
from classical_ml.algorithms.neural_net import sigmoid


def test_int_to_onehot():
    encoded = int_to_onehot(np.array([0, 2, 1]), num_labels=3)
    assert encoded.tolist() == [[1, 0, 0], [0, 0, 1], [0, 1, 0]]


def test_sigmoid_does_not_overflow():
    values = sigmoid(np.array([-1e6, 0.0, 1e6]))
    assert np.all(np.isfinite(values))
    assert values[1] == pytest.approx(0.5)


def test_forward_shapes():
    model = NeuralNetMLP(num_features=6, num_hidden=4, num_classes=3)
    a_h, a_out = model.forward(np.random.RandomState(0).normal(size=(5, 6)))
    assert a_h.shape == (5, 4)
    assert a_out.shape == (5, 3)


def test_backward_matches_numerical_gradients():
    """A central-difference check on every parameter of a small network.

    This is the test that actually validates the chain rule as written out in
    ``NeuralNetMLP.backward``. An accuracy threshold would not catch a subtly wrong
    derivative, because gradient descent tends to make progress anyway; a gradient
    check does.
    """
    rng = np.random.RandomState(1)
    X = rng.normal(size=(4, 5))
    y = np.array([0, 1, 2, 1])
    model = NeuralNetMLP(num_features=5, num_hidden=3, num_classes=3, random_seed=1)

    a_h, a_out = model.forward(X)
    d_w_out, d_b_out, d_w_h, d_b_h = model.backward(X, a_h, a_out, y)

    def loss() -> float:
        return mse_loss(y, model.forward(X)[1], num_labels=3)

    epsilon = 1e-5
    for parameter, analytic in (
        (model.weight_out, d_w_out),
        (model.bias_out, d_b_out),
        (model.weight_h, d_w_h),
        (model.bias_h, d_b_h),
    ):
        flat = parameter.reshape(-1)
        numeric = np.zeros_like(flat)
        for index in range(flat.size):
            original = flat[index]
            flat[index] = original + epsilon
            plus = loss()
            flat[index] = original - epsilon
            minus = loss()
            flat[index] = original
            numeric[index] = (plus - minus) / (2 * epsilon)
        # mse_loss averages over classes as well as examples; backward does not.
        assert np.allclose(numeric * 3, analytic.reshape(-1), atol=1e-6)


def test_minibatch_generator_covers_the_data_without_repeats():
    X = np.arange(40).reshape(20, 2).astype(float)
    y = np.arange(20)
    batches = list(minibatch_generator(X, y, minibatch_size=5))
    assert len(batches) == 4
    seen = np.concatenate([batch_y for _, batch_y in batches])
    assert sorted(seen.tolist()) == list(range(20))


def test_training_improves_accuracy():
    rng = np.random.RandomState(0)
    X = rng.normal(size=(400, 20))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    model = NeuralNetMLP(num_features=20, num_hidden=10, num_classes=2, random_seed=1)

    _, before = compute_mse_and_acc(model, X, y, num_labels=2, minibatch_size=50)
    train(model, X[:300], y[:300], X[300:], y[300:], num_epochs=30, learning_rate=0.5,
          minibatch_size=50, verbose=False)
    _, after = compute_mse_and_acc(model, X, y, num_labels=2, minibatch_size=50)

    assert after > before
    assert after > 0.85
