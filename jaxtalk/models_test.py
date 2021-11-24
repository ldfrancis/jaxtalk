import jax
import jax.numpy as jnp
import numpy as np

from jaxtalk.models import Linear


def test_linear__get_parameters(linear: Linear):
    parameters = linear.get_parameters()
    assert parameters["weight"].shape == (16, 16)
    assert parameters["bias"].shape == (16,)


def test_linear__set_parameters(linear):
    parameters = {
        "weight": jnp.ones((16, 16)),
        "bias": jnp.ones((16,)),
    }
    linear.set_parameters(parameters)
    new_parameters = linear.get_parameters()
    np.testing.assert_equal(
        np.array(parameters["weight"]),
        np.array(new_parameters["weight"]),
    )
    np.testing.assert_equal(
        np.array(parameters["bias"]),
        np.array(new_parameters["bias"]),
    )


def test_linear__update_parameters(linear):
    parameters = {"weight": jnp.ones((16, 16))}
    linear.update_parameters(parameters)
    new_parameters = linear.get_parameters()
    np.testing.assert_equal(
        np.array(parameters["weight"]),
        np.array(new_parameters["weight"]),
    )


def test_linear__get_weight(linear):
    weight1 = linear.get_parameters()["weight"]
    weight2 = linear.get_weight()["weight"]
    np.testing.assert_equal(np.array(weight1), np.array(weight2))


def test_linear__check_usage(linear, image_features):
    image_encoding = linear(image_features)
    assert image_encoding.shape == (4, 16)


def test_linear__check_grad(linear, image_features):
    @jax.jit
    @jax.grad
    def loss_fn(linear, img_feat):
        enc = linear(img_feat)
        loss = jnp.sum(enc)
        return loss

    grads = loss_fn(linear, image_features)
    weight = grads.get_parameters()["weight"]
    bias = grads.get_parameters()["bias"]
    assert weight.shape == (16, 16)
    assert bias.shape == (16,)
    np.testing.assert_raises(
        AssertionError,
        np.testing.assert_array_equal,
        weight,
        linear.get_parameters()["weight"],
    )
    np.testing.assert_raises(
        AssertionError,
        np.testing.assert_array_equal,
        weight,
        jnp.zeros_like(weight),
    )
    np.testing.assert_raises(
        AssertionError,
        np.testing.assert_array_equal,
        bias,
        jnp.zeros_like(bias),
    )


def test_embedding__check_usage(embedding, word_tokens):
    word_encoding = embedding(word_tokens)
    assert word_encoding.shape == (4, 10, 16)


def test_embedding__check_grad(embedding, word_tokens):
    @jax.jit
    @jax.grad
    def loss_fn(embedding, word_tokens):
        enc = embedding(word_tokens)
        loss = jnp.sum(enc)
        return loss

    grads = loss_fn(embedding, word_tokens)
    weight = grads.get_parameters()["weight"]
    bias = grads.get_parameters()["bias"]
    assert weight.shape == (10, 16)
    assert bias.shape == (16,)
    np.testing.assert_raises(
        AssertionError,
        np.testing.assert_array_equal,
        weight,
        embedding.get_parameters()["weight"],
    )
    np.testing.assert_raises(
        AssertionError,
        np.testing.assert_array_equal,
        weight,
        jnp.zeros_like(weight),
    )
    np.testing.assert_raises(
        AssertionError,
        np.testing.assert_array_equal,
        bias,
        jnp.zeros_like(bias),
    )


def test_lstm__get_parameters(lstm):
    parameters = lstm.get_parameters()
    assert parameters["weight"].shape == (
        16 + 32,
        32 * 4,
    )
    assert parameters["bias"].shape == (32 * 4,)


def test_lstm__set_parameters(lstm):
    parameters = {
        "weight": jnp.ones((16 + 32, 32 * 4)),
        "bias": jnp.ones((32 * 4,)),
    }
    lstm.set_parameters(parameters)
    new_parameters = lstm.get_parameters()
    np.testing.assert_equal(
        np.array(parameters["weight"]),
        np.array(new_parameters["weight"]),
    )
    np.testing.assert_equal(
        np.array(parameters["bias"]),
        np.array(new_parameters["bias"]),
    )


def test_lstm__update_parameters(lstm):
    parameters = {"weight": jnp.ones((16 + 32, 32 * 4))}
    lstm.update_parameters(parameters)
    new_parameters = lstm.get_parameters()
    np.testing.assert_equal(
        np.array(parameters["weight"]),
        np.array(new_parameters["weight"]),
    )


def test_lstm__get_weight(lstm):
    weight1 = lstm.get_parameters()["weight"]
    weight2 = lstm.get_weight()["weight"]
    np.testing.assert_equal(np.array(weight1), np.array(weight2))


def test_lstm__usage(lstm, embedding, word_tokens):
    word_encoding = embedding(word_tokens)
    outputs = lstm(word_encoding)
    assert word_encoding.shape == (4, 10, 16)
    assert outputs.shape == (4, 10, 32)


def test_lstm__check_grad(lstm, embedding, word_tokens):
    @jax.jit
    @jax.grad
    def loss_fn(lstm, embedding, word_tokens):
        enc = embedding(word_tokens)
        outputs = lstm(enc)
        loss = jnp.sum(outputs)
        return loss

    grads = loss_fn(lstm, embedding, word_tokens)
    weight = grads.get_parameters()["weight"]
    bias = grads.get_parameters()["bias"]
    assert weight.shape == (16 + 32, 32 * 4)
    assert bias.shape == (32 * 4,)
    np.testing.assert_raises(
        AssertionError,
        np.testing.assert_array_equal,
        weight,
        embedding.get_parameters()["weight"],
    )
    np.testing.assert_raises(
        AssertionError,
        np.testing.assert_array_equal,
        weight,
        jnp.zeros_like(weight),
    )
    np.testing.assert_raises(
        AssertionError,
        np.testing.assert_array_equal,
        bias,
        jnp.zeros_like(bias),
    )


def test_captionmodel__get_parameters(
    captionmodel,
):
    parameters = captionmodel.get_parameters()
    assert parameters["lstm"]["weight"].shape == (
        16 + 32,
        32 * 4,
    )
    assert parameters["lstm"]["bias"].shape == (32 * 4,)
    assert parameters["word_encoder"]["weight"].shape == (
        10,
        16,
    )
    assert parameters["word_encoder"]["bias"].shape == (16,)
    assert parameters["image_encoder"]["weight"].shape == (
        16,
        16,
    )
    assert parameters["image_encoder"]["bias"].shape == (16,)
    assert parameters["word_decoder"]["weight"].shape == (
        32,
        10 - 1,
    )
    assert parameters["word_decoder"]["bias"].shape == (10 - 1,)


def test_captionmodel__set_parameters(
    captionmodel,
):
    parameters = {
        "lstm": {
            "weight": jnp.ones((16 + 32, 32 * 4)),
            "bias": jnp.ones((32 * 4,)),
        },
        "word_encoder": {
            "weight": jnp.ones((10, 16)),
            "bias": jnp.ones((16,)),
        },
        "image_encoder": {
            "weight": jnp.ones((16, 16)),
            "bias": jnp.ones((16,)),
        },
        "word_decoder": {
            "weight": jnp.ones((32, 10 - 1)),
            "bias": jnp.ones((10 - 1,)),
        },
    }
    captionmodel.set_parameters(parameters)
    new_parameters = captionmodel.get_parameters()
    np.testing.assert_equal(
        np.array(parameters["lstm"]["weight"]),
        np.array(new_parameters["lstm"]["weight"]),
    )
    np.testing.assert_equal(
        np.array(parameters["word_encoder"]["bias"]),
        np.array(new_parameters["word_encoder"]["bias"]),
    )


def test_captionmodel__update_parameters(
    captionmodel,
):
    parameters = {"lstm": {"weight": jnp.ones((16 + 32, 32 * 4))}}
    captionmodel.update_parameters(parameters)
    new_parameters = captionmodel.get_parameters()
    np.testing.assert_equal(
        np.array(parameters["lstm"]["weight"]),
        np.array(new_parameters["lstm"]["weight"]),
    )


def test_captionmodel__usage(captionmodel, image_features, word_tokens):
    output_logits = captionmodel.forward(image_features, word_tokens)
    assert output_logits.shape == (4, 10 + 1, 9)


def test_captionmodel__check_grad(captionmodel, image_features, word_tokens):
    @jax.jit
    @jax.grad
    def loss_fn(captionmodel, image_features, word_tokens):
        output_logits = captionmodel.forward(image_features, word_tokens)
        loss = 100 * jnp.sum(output_logits)
        return loss

    grads = loss_fn(captionmodel, image_features, word_tokens)
    grad_params = grads.get_parameters()
    params = captionmodel.get_parameters()
    assert grad_params["lstm"]["weight"].shape == (16 + 32, 32 * 4)
    assert grad_params["lstm"]["bias"].shape == (32 * 4,)

    for key in params.keys():
        for key2 in ["weight", "bias"]:
            np.testing.assert_raises(
                AssertionError,
                np.testing.assert_array_equal,
                grad_params[key][key2],
                params[key][key2],
            )
            np.testing.assert_raises(
                AssertionError,
                np.testing.assert_array_equal,
                grad_params[key][key2],
                jnp.zeros_like(grad_params[key][key2]),
            )
