import json
import logging
import pickle
import random
import time
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from jaxtalk.data import BasicDataProvider
from jaxtalk.losses import CaptionLoss
from jaxtalk.models import CaptionModel
from jaxtalk.optimizer import Optimizer
from jaxtalk.utils import Logger, Mean


class Trainer:
    def __init__(
        self,
        ixtoword: Dict[int, str],
        grad_clip=1,
        regc=1e-3,
        evaluate_every=1,
        name: str = "",
    ) -> None:
        self._name = name or "Trainer"
        self._num_tokens: int = len(ixtoword) - 1  # minus pad
        self._pad_token: int = len(ixtoword)
        self._evaluate_every = evaluate_every
        self._optimizer = Optimizer(1e-3, grad_clip)
        self._loss_fn: CaptionLoss = CaptionLoss(self._num_tokens, self._pad_token)
        self._weight_decay: float = regc
        self._logger = Logger(self._name, logging.INFO)
        self._global_step = 0
        self._epoch = 0
        self.history: List[Dict[str, Any]] = []
        self._state: Dict[str, Any] = {}
        self._time = time.time()
        self._best_perplexity = np.inf
        self._checkpoint_path = Path(f"logs/checkpoint/{self._name}")
        self._checkpoint_path.mkdir(parents=True, exist_ok=True)
        history_path = Path(f"logs/histories/{self._name}")
        history_path.mkdir(parents=True, exist_ok=True)
        self._histroy_file = history_path / "history.json"

    def _step(
        self,
        model: CaptionModel,
        nest_batch: Dict[str, jnp.ndarray],
    ) -> jnp.ndarray:
        img_feats = nest_batch["feats"]
        tokens = jnp.concatenate(
            [jnp.zeros((img_feats.shape[0], 1)), nest_batch["tokens"]], axis=1
        )
        output_logits = model.forward(img_feats, tokens)
        output_probs = jax.nn.softmax(output_logits, axis=2)[:, 1:, :]
        return output_probs

    def _grad(
        self,
        model: CaptionModel,
        nest_batch: Dict[str, jnp.ndarray],
    ) -> jnp.float32:
        output_probs = self._step(model, nest_batch)
        tokens = nest_batch["tokens"]
        loss, _ = self._loss_fn(output_probs, tokens)
        regloss = 0
        for value in model.get_parameters().values():
            regloss = regloss + jnp.sum(value["weight"] * value["weight"])
        loss = loss + self._weight_decay * (regloss * (1 / tokens.shape[0]))
        return loss

    def _train_step(
        self,
        model: CaptionModel,
        nest_batch: Dict[str, jnp.ndarray],
    ) -> Tuple[jnp.float32, jnp.float32, jnp.ndarray]:
        output_probs = jax.jit(self._step)(model, nest_batch)
        loss, log2_perplexity = jax.jit(self._loss_fn)(
            output_probs, nest_batch["tokens"]
        )
        grads = jax.jit(jax.grad(self._grad))(model, nest_batch)
        # import pdb

        # pdb.set_trace()
        self._global_step += 1
        return loss, log2_perplexity, output_probs, grads

    def _eval_step(
        self,
        model: CaptionModel,
        nest_batch: Dict[str, jnp.ndarray],
    ) -> Tuple[jnp.float32, jnp.float32, jnp.ndarray]:
        output_probs = self._step(model, nest_batch)
        loss, log2_perplexity = self._loss_fn(output_probs, nest_batch["tokens"])
        return loss, log2_perplexity, output_probs

    def _evaluate(self, model: CaptionModel, data_gen: Generator) -> None:
        loss = Mean()
        perplexity = Mean()
        batch = None
        nest_batch = None
        for batch, nest_batch in data_gen:
            (_loss, _log2_perplexity, _,) = jax.jit(
                self._eval_step
            )(model, nest_batch)
            loss.update(_loss)
            perplexity.update(_log2_perplexity)
        if batch is None or nest_batch is None:
            raise ValueError(
                "batch or nest_batch cannot be None. Please supply a valid data"
                " generator"
            )
        self._logger.info(
            (f"Evaluation:\nLoss: {loss} | Perplexity: {perplexity.state():.2f}")
        )
        words, gt_words = self.predict(model, batch, nest_batch)
        self._set_state(
            "eval",
            float(loss.state()),
            float(perplexity.state()),
            float(loss.state()),
            float(perplexity.state()),
            words,
            gt_words,
        )
        if perplexity.state() < self._best_perplexity:
            checkpoint_file = self._checkpoint_path / (
                f"checkpoint_{self._epoch}_"
                f"{self._global_step}_"
                f"{perplexity.state():.2f}.jxc"
            )
            pickle.dump(
                model.get_parameters(),
                open(
                    str(checkpoint_file.absolute()),
                    "wb",
                ),
            )
            self._best_perplexity = perplexity.state()

    def predict(
        self,
        model: CaptionModel,
        batch: List[Dict[str, Any]],
        nest_batch: Dict[str, jnp.ndarray],
    ) -> Tuple[List[str], List[str]]:
        idx = random.randint(1, len(batch) - 1)
        words = model.predict(
            nest_batch["feats"][idx][None, :],
            max_len=(nest_batch["tokens"].shape[1] + 5),
        )
        self._logger.info(
            (
                f"\nSample Predictions:\nPrediction: {words}\nGround truth: "
                f"{[batch[idx]['sentence']['raw']]}"
            )
        )
        return words, [batch[idx]["sentence"]["raw"]]

    def _set_state(
        self,
        split: str,
        loss: float,
        perplexity: float,
        mean_loss: float,
        mean_perplexity: float,
        predicted_words: List[str] = [],
        gt_words: List[str] = [],
    ) -> None:
        self._state = {
            "split": split,
            "loss": loss,
            "perplexity": perplexity,
            "mean_loss": mean_loss,
            "mean_perplexity": mean_perplexity,
            "global_step": self._global_step,
            "predicted_words": predicted_words,
            "gt_words": gt_words,
        }
        self.history += [self._state]

    def _update_state(self, **kwargs) -> None:
        self._state.update(kwargs)
        self.history[-1].update(self._state)

    def _get_last_state(self, split) -> Optional[Dict[str, Any]]:
        for i in range(len(self.history) - 1, -1, -1):
            state = self.history[i]
            if state["split"] == split:
                return state
        return None

    def _dump_history(self) -> None:
        json.dump(
            self.history,
            open(
                str(self._histroy_file.absolute()),
                "w",
            ),
        )

    def fit(
        self,
        model: CaptionModel,
        data_provider: BasicDataProvider,
        epochs: int,
        lr: float,
    ) -> None:
        self._optimizer.set_lr(lr)
        self._model = model
        mean_loss = Mean()
        mean_perplexity = Mean()
        for epoch in range(1, epochs + 1):
            batch = None
            nest_batch = None
            self._epoch = epoch
            for i, (batch, nest_batch) in enumerate(
                data_provider.iter_image_sentence_pair_batch("train")
            ):

                self._time = time.time()
                loss, log2_perplexity, _, grads = jax.jit(self._train_step)(
                    model, nest_batch
                )
                self._optimizer.step(model, grads)
                tm = time.time() - self._time
                # import pdb

                # pdb.set_trace()
                # for key in prev_params.keys():
                #     for key2 in ["weight", "bias"]:
                #         np.testing.assert_raises(
                #             AssertionError,
                #             np.testing.assert_array_equal,
                #             prev_params[key][key2],
                #             new_params[key][key2],
                #         )

                mean_loss.update(loss)
                mean_perplexity.update(log2_perplexity)
                self._logger.info(
                    f"Epoch: {epoch}/{epochs} | Batch: {i:3d} | Time: {tm:1.2f} | Loss: "
                    f"{loss:3.2f} | Perplexity: {log2_perplexity:3.2f} | "
                    f"Mean Loss: {mean_loss.state():3.2f} | Mean Perplexity: "
                    f"{mean_perplexity.state():3.2f}"
                )
                self._set_state(
                    "train",
                    float(loss),
                    float(log2_perplexity),
                    float(mean_loss.state()),
                    float(mean_perplexity.state()),
                )
            words, gt_words = self.predict(model, batch, nest_batch)
            self._update_state(
                predicted_words=words,
                gt_words=gt_words,
            )
            if epoch % self._evaluate_every == 0:
                self._evaluate(
                    model,
                    data_provider.iter_image_sentence_pair_batch("val"),
                )
            self._dump_history()
