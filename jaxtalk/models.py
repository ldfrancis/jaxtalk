from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from jaxtalk.utils import init_array, init_bias


class LSTM:
    _parameters: Dict[str, jnp.ndarray] = {}

    def __init__(
        self,
        embedding_size: int = 1,
        hidden_size: int = 1,
        dropout=0.0,
        key=None,
        seed=0,
    ) -> None:
        self._embedding_size = embedding_size
        self._hidden_size = hidden_size
        self._dropout = dropout
        self._seed = seed
        self._key = key if key is not None else jax.random.PRNGKey(seed)
        _, subkey = jax.random.split(self._key)
        self._keys = {"weight": subkey}
        self._init()

    def _init(self):
        self._parameters = {
            "weight": init_array(
                self._keys["weight"],
                self._embedding_size + self._hidden_size,
                4 * self._hidden_size,
            ),
            "bias": init_bias(4 * self._hidden_size),
        }

    def args(self):
        return (
            self._embedding_size,
            self._hidden_size,
            self._dropout,
            self._key,
            self._seed,
        )

    def get_parameters(self):
        return self._parameters

    def set_parameters(self, parameters):
        self._parameters = parameters

    def update_parameters(self, parameters) -> None:
        self._parameters.update(parameters)

    def get_weight(self):
        return {"weight": self._parameters["weight"]}

    def _forward(
        self,
        inp: jnp.ndarray,
        hidden: jnp.ndarray,
        cell: jnp.ndarray,
        weight: jnp.ndarray,
        bias: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        gates = (
            jnp.dot(
                jnp.concatenate([hidden, inp], axis=1),
                weight,
            )
            + bias
        )
        input_gate = jax.nn.sigmoid(gates[:, : self._hidden_size])
        forget_gate = jax.nn.sigmoid(gates[:, self._hidden_size : 2 * self._hidden_size])
        output_gate = jax.nn.sigmoid(
            gates[:, 2 * self._hidden_size : 3 * self._hidden_size]
        )
        cell_info = jax.nn.tanh(gates[:, 3 * self._hidden_size :])
        cell_state = forget_gate * cell + input_gate * cell_info
        hidden = output_gate * jax.nn.tanh(cell_state)
        return hidden, cell_state

    def step(self, inp, hidden=None, cell=None):
        hidden = (
            hidden
            if hidden is not None
            else jnp.zeros(
                (
                    inp.shape[0],
                    self._hidden_size,
                )
            )
        )
        cell = (
            cell
            if cell is not None
            else jnp.zeros(
                (
                    inp.shape[0],
                    self._hidden_size,
                )
            )
        )
        weight = self._parameters["weight"]
        bias = self._parameters["bias"]
        hidden, cell = jax.jit(self._forward)(inp, hidden, cell, weight, bias)
        return hidden, cell

    def __call__(self, inp, hidden=None, cell=None):
        hidden = (
            hidden
            if hidden is not None
            else jnp.zeros(
                (
                    inp.shape[0],
                    self._hidden_size,
                )
            )
        )
        cell = (
            cell
            if cell is not None
            else jnp.zeros(
                (
                    inp.shape[0],
                    self._hidden_size,
                )
            )
        )
        hiddens = []
        for i in range(inp.shape[1]):
            hidden, cell = self.step(
                inp[:, i, :],
                hidden,
                cell,
            )
            hiddens += [hidden]
        return jnp.stack(hiddens, axis=1)


# Register LSTM Model
def _lstm_flatten_(model: Any) -> Tuple[List[jnp.ndarray], List[str]]:
    """Flatten the model as a container
    This helps when registering as a pytree node in jax.
    """
    parameters = model.get_parameters()
    params = [parameters["weight"], parameters["bias"]]
    args = model.args()
    return params, args


def _lstm_unflatten_(args, params) -> LSTM:
    """Unflatten the model from its parameters"""
    parameters = {"weight": params[0], "bias": params[1]}
    model = LSTM(*args)
    model.set_parameters(parameters)
    return model


jax.tree_util.register_pytree_node(LSTM, _lstm_flatten_, _lstm_unflatten_)


class Linear:
    """Applies a Linear transformation on an input"""

    _parameters: Dict[str, jnp.ndarray] = {}

    def __init__(
        self,
        input_size,
        output_size,
        key: Optional[np.array] = None,
        seed=0,
    ) -> None:
        self._input_size = input_size
        self._output_size = output_size
        self._key = key if key is not None else jax.random.PRNGKey(seed)
        _, subkey = jax.random.split(self._key)
        self._keys = {"weight": subkey}
        self._init()

    def _init(self):
        """Initialize model parameters, weight and bias"""
        self._parameters: Dict[str, jnp.ndarray] = {
            "weight": init_array(
                self._keys["weight"], self._input_size, self._output_size
            ),
            "bias": init_bias(self._output_size),
        }

    def args(self):
        return self._input_size, self._output_size, self._key

    def get_parameters(self) -> Dict[str, jnp.ndarray]:
        """Returns model parameters"""
        return self._parameters

    def set_parameters(self, parameters: Dict[str, jnp.ndarray]) -> None:
        """Sets the model parameters"""
        self._parameters = parameters

    def update_parameters(self, parameters) -> None:
        self._parameters.update(parameters)

    def get_weight(self):
        weight_keys = ["weight"]
        return {key: self._parameters[key] for key in weight_keys}

    def _forward(
        self, inp: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray
    ) -> jnp.ndarray:
        """Forward pass through model"""
        return jnp.dot(inp, weight) + bias

    def __call__(self, inp):
        weight = self._parameters["weight"]
        bias = self._parameters["bias"]
        return jax.jit(self._forward)(inp, weight, bias)


# Register Linear Model
def _linear_flatten_(model: Any) -> Tuple[List[jnp.ndarray], List[Any]]:
    """Flatten the model as a container
    This helps when registering as a pytree node in jax.
    """
    parameters = model.get_parameters()
    param_list = [parameters["weight"], parameters["bias"]]
    args = model.args()
    return param_list, args


def _linear_unflatten_(args, params) -> Linear:
    """Unflatten the model from its parameters"""
    parameters = {"weight": params[0], "bias": params[1]}
    model = Linear(*args)
    model.set_parameters(parameters)
    return model


jax.tree_util.register_pytree_node(Linear, _linear_flatten_, _linear_unflatten_)


class Embedding(Linear):
    def __call__(self, inp: int):
        inp = jax.nn.one_hot(inp, self._input_size)
        return super().__call__(inp)


# Register Embedding Model
def _embedding_flatten_(model: Any) -> Tuple[List[jnp.ndarray], List[Any]]:
    """Flatten the model as a container
    This helps when registering as a pytree node in jax.
    """
    parameters = model.get_parameters()
    param_list = [parameters["weight"], parameters["bias"]]
    args = model.args()
    return param_list, args


def _embedding_unflatten_(args, params) -> Embedding:
    """Unflatten the model from its parameters"""
    parameters = {"weight": params[0], "bias": params[1]}
    model = Embedding(*args)
    model.set_parameters(parameters)
    return model


jax.tree_util.register_pytree_node(Embedding, _embedding_flatten_, _embedding_unflatten_)


class CaptionModel:
    def __init__(
        self,
        embedding_size,
        hidden_size,
        image_feature_size,
        wordtoix,
        ixtoword,
        max_len,
        key=None,
        seed=0,
    ) -> None:
        self._embedding_size = embedding_size
        self._hidden_size = hidden_size
        self._image_feature_size = image_feature_size
        self._wordtoix = wordtoix
        self._ixtoword = ixtoword
        self._max_len = max_len + 1  # plus #START#/#PAD# tokens
        self._key = key if key is not None else jax.random.PRNGKey(seed)
        self._seed = seed
        self._init()

    def _init(self):
        self.image_encoder = Linear(
            self._image_feature_size, self._embedding_size, key=self._key
        )
        self.word_encoder = Embedding(len(self._wordtoix), self._embedding_size)
        self.lstm = LSTM(self._embedding_size, self._hidden_size, key=self._key)
        self.word_decoder = Linear(
            self._hidden_size, len(self._ixtoword) - 1
        )  # minus #PAD#
        self._parameters = {
            "image_encoder": self.image_encoder.get_parameters(),
            "word_encoder": self.word_encoder.get_parameters(),
            "lstm": self.lstm.get_parameters(),
            "word_decoder": self.word_decoder.get_parameters(),
        }

    def args(self):
        return (
            self._embedding_size,
            self._hidden_size,
            self._image_feature_size,
            self._wordtoix,
            self._ixtoword,
            self._max_len - 1,
            self._key,
            self._seed,
        )

    def get_parameters(self):
        return {
            "image_encoder": self.image_encoder.get_parameters(),
            "word_encoder": self.word_encoder.get_parameters(),
            "lstm": self.lstm.get_parameters(),
            "word_decoder": self.word_decoder.get_parameters(),
        }

    def set_parameters(self, parameters):
        for key, value in parameters.items():
            getattr(self, key).set_parameters(value)

    def update_parameters(self, parameters):
        for key, value in parameters.items():
            getattr(self, key).update_parameters(value)

    def forward(self, img_feat: jnp.ndarray, sent_tokens: jnp.ndarray):
        img_encoding = self.image_encoder(img_feat)
        word_encoding = self.word_encoder(sent_tokens)
        input_encoding = jnp.concatenate(
            [jnp.expand_dims(img_encoding, 1), word_encoding], axis=1
        )
        output_hiddens = self.lstm(input_encoding)
        output_logits = self.word_decoder(output_hiddens)
        return output_logits

    def predict(self, img_feat, max_len=None):
        max_len = max_len or self._max_len
        img_encoding = self.image_encoder(img_feat)
        start_token = jnp.zeros((img_feat.shape[0], 1))
        inp_embedding = jnp.squeeze(self.word_encoder(start_token), axis=1)
        hidden, cell = self.lstm.step(img_encoding)
        probs = []
        tokens = []
        # import pdb

        # pdb.set_trace()
        words = ["" for _ in range(img_feat.shape[0])]
        for _ in range(max_len):
            hidden, cell = self.lstm.step(inp_embedding, hidden, cell)
            logits = self.word_decoder(hidden)
            prob = jax.nn.softmax(logits, axis=-1)
            token = jnp.argmax(prob, axis=-1)
            inp_embedding = self.word_encoder(token)
            probs += [prob]
            tokens += [token]
            for i, t in enumerate(token):
                words[i] += f"{self._ixtoword[t]} "
        for i, w in enumerate(words):
            words[i] = w.split(".")[0]
        return words


# Register Caption Model
def _caption_model_flatten_(
    model: Any,
):
    """Flatten the model as a container
    This helps when registering as a pytree node in jax.
    """
    parameters = model.get_parameters()
    params = [
        parameters["image_encoder"]["weight"],
        parameters["image_encoder"]["bias"],
        parameters["word_encoder"]["weight"],
        parameters["word_encoder"]["bias"],
        parameters["lstm"]["weight"],
        parameters["lstm"]["bias"],
        parameters["word_decoder"]["weight"],
        parameters["word_decoder"]["bias"],
    ]
    args = model.args()
    return params, args


def _caption_model_unflatten_(args, params) -> CaptionModel:
    """Unflatten the model from its parameters"""
    parameters = {
        "image_encoder": {"weight": params[0], "bias": params[1]},
        "word_encoder": {"weight": params[2], "bias": params[3]},
        "lstm": {"weight": params[4], "bias": params[5]},
        "word_decoder": {"weight": params[6], "bias": params[7]},
    }
    model = CaptionModel(*args)
    model.set_parameters(parameters)
    return model


jax.tree_util.register_pytree_node(
    CaptionModel, _caption_model_flatten_, _caption_model_unflatten_
)
