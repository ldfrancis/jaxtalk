"""Based on neuraltalk
"""
import jax
import jax.numpy as jnp

from jaxtalk.models import CaptionModel
from jaxtalk.types import Nest


class Optimizer:
    def __init__(self, lr: float, grad_clip: int) -> None:
        self._grad_clip = grad_clip
        self._lr = lr

    def step(self, model: CaptionModel, grads: Nest) -> None:
        grads = jax.tree_map(
            lambda x: jnp.clip(x, -1 * self._grad_clip, self._grad_clip), grads
        )
        model.set_parameters(
            jax.tree_multimap(
                lambda m, g: m - self._lr * g, model, grads
            ).get_parameters()
        )

    def set_lr(self, lr: float) -> None:
        self._lr = lr
