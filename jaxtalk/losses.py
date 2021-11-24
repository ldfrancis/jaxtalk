from typing import Tuple

import jax
import jax.numpy as jnp


class CaptionLoss:
    def __init__(
        self, num_tokens: int, pad_token: int = 1, start_token: int = 0
    ) -> None:
        self._pad_token: int = pad_token
        self._start_token: int = start_token
        self._num_tokens: int = num_tokens

    def set_pad_token(self, token: int) -> None:
        self._pad_token = token

    def __call__(
        self, predicted_probs: jnp.ndarray, gt_tokens: jnp.ndarray
    ) -> Tuple[jnp.float32, jnp.float32]:
        gt_tokens = jnp.concatenate(
            [gt_tokens, jnp.ones((gt_tokens.shape[0], 1)) * self._pad_token], axis=-1
        )
        gt_one_hot = jax.nn.one_hot(
            gt_tokens,
            self._num_tokens,
        )
        loss = (
            -1
            * jnp.sum(jnp.log(predicted_probs) * gt_one_hot)
            * (1.0 / (gt_tokens.shape[0]))
        )
        log2_perplexity = (
            -1
            * jnp.sum(jnp.log2(predicted_probs) * gt_one_hot)
            * (1.0 / (jnp.sum(gt_one_hot)))
        )
        return loss, 2 ** log2_perplexity
