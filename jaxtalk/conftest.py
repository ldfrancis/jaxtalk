from typing import Dict

import jax
import pytest
from jax._src.random import KeyArray

from jaxtalk.models import LSTM, CaptionModel, Embedding, Linear
from jaxtalk.types import DictNest, JaxArray


@pytest.fixture
def key() -> KeyArray:
    return jax.random.PRNGKey(0)


@pytest.fixture
def linear(key: KeyArray) -> Linear:
    key, subkey = jax.random.split(key)
    return Linear(input_size=16, output_size=16, key=subkey)


@pytest.fixture
def word_decoder(key: KeyArray) -> Linear:
    key, subkey = jax.random.split(key)
    return Linear(input_size=32, output_size=10, key=subkey)


@pytest.fixture
def ixtoword() -> Dict[int, str]:
    return {
        0: ".",
        1: "a",
        2: "e",
        3: "i",
        4: "o",
        5: "u",
        6: "l",
        7: "m",
        8: "n",
        9: "#PAD#",
    }


@pytest.fixture
def wordtoix() -> Dict[str, int]:
    return {
        "#start#": 0,
        "a": 1,
        "e": 2,
        "i": 3,
        "o": 4,
        "u": 5,
        "l": 6,
        "m": 7,
        "n": 8,
        "#PAD#": 9,
    }


@pytest.fixture
def captionmodel(
    key: KeyArray, wordtoix: Dict[str, int], ixtoword: Dict[int, str]
) -> CaptionModel:
    key, subkey = jax.random.split(key)
    return CaptionModel(
        embedding_size=16,
        hidden_size=32,
        image_feature_size=16,
        wordtoix=wordtoix,
        ixtoword=ixtoword,
        max_len=18,
        key=subkey,
    )


@pytest.fixture
def linear2(key: KeyArray) -> Linear:
    key, subkey = jax.random.split(key)
    return Linear(input_size=16, output_size=16, key=subkey)


@pytest.fixture
def lstm(key: KeyArray) -> LSTM:
    key, subkey = jax.random.split(key)
    return LSTM(
        embedding_size=16,
        hidden_size=32,
        key=subkey,
    )


@pytest.fixture
def embedding(key: KeyArray) -> Embedding:
    key, subkey = jax.random.split(key)
    return Embedding(input_size=10, output_size=16, key=subkey)


@pytest.fixture
def image_features(key: KeyArray) -> JaxArray:
    key, subkey = jax.random.split(key)
    return jax.random.normal(subkey, (4, 16))


@pytest.fixture
def word_tokens(key: KeyArray) -> JaxArray:
    key, subkey = jax.random.split(key)
    return jax.random.randint(subkey, (4, 10), 0, 10)


@pytest.fixture
def nest_batch(image_features: JaxArray, word_tokens: JaxArray) -> DictNest:
    return {
        "feats": image_features,
        "tokens": word_tokens,
    }
