import jax
import jax.numpy as jnp
import pytest

from jaxtalk.losses import CaptionLoss


@pytest.fixture
def output_logits(key):
    _, subkey = jax.random.split(key)
    return jax.random.uniform(subkey, (4, 11, 9), minval=0, maxval=20)


@pytest.fixture
def gt_tokens(key):
    _, subkey = jax.random.split(key)
    return jax.random.randint(subkey, (4, 10), minval=0, maxval=9)


@pytest.fixture
def captionloss():
    return CaptionLoss(num_tokens=9, pad_token=10, start_token=0)


def test_captionloss__usage(captionloss, output_logits, gt_tokens):
    output_probs = jax.nn.softmax(output_logits, axis=-1)
    loss, log2_perplexity = captionloss(output_probs, gt_tokens)
    assert loss < log2_perplexity
    assert loss >= 0
    assert log2_perplexity >= 0
