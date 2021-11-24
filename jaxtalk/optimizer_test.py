from copy import deepcopy

import jax.numpy as jnp
import numpy as np
import pytest

from jaxtalk.models import Linear
from jaxtalk.optimizer import Optimizer


@pytest.fixture
def model(key):
    return Linear(input_size=8, output_size=8, key=key)


@pytest.fixture
def optimizer():
    return Optimizer(lr=1e-2, grad_clip=1)


def test_optimizer__step(key, model, optimizer):
    parameters = model.get_parameters()
    grads = Linear(input_size=8, output_size=8, key=key)
    grads.set_parameters(
        {
            "weight": 2 * jnp.ones_like(parameters["weight"]),
            "bias": 2 * jnp.ones_like(parameters["bias"]),
        }
    )
    optimizer.step(model, grads)
    new_parameters = model.get_parameters()
    np.testing.assert_raises(
        AssertionError,
        np.testing.assert_array_equal,
        new_parameters["weight"],
        parameters["weight"],
    )
