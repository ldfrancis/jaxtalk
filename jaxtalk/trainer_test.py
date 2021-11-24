import json
import os
import pickle
from typing import Dict

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxtalk.models import CaptionModel
from jaxtalk.trainer import Trainer
from jaxtalk.types import DictNest


@pytest.fixture
def trainer(ixtoword: Dict[int, str]):
    return Trainer(ixtoword=ixtoword, name="test_trainer")


def test_trainer__step(
    trainer: Trainer, captionmodel: CaptionModel, nest_batch: DictNest
):
    output_probs = trainer._step(captionmodel, nest_batch)
    assert output_probs.shape == (
        4,
        10 + 1,
        9,
    )  # plus start token
    prob_sum = np.sum(output_probs, axis=-1)
    np.testing.assert_almost_equal(prob_sum, np.ones_like(prob_sum))


def test_trainer__grad(
    trainer: Trainer, captionmodel: CaptionModel, nest_batch: DictNest
):
    grads = (jax.grad(trainer._grad))(captionmodel, nest_batch)
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


def test_trainer__train_step(
    trainer: Trainer, captionmodel: CaptionModel, nest_batch: DictNest
):
    (
        loss,
        log2_perplexity,
        output_probs,
    ) = trainer._train_step(captionmodel, nest_batch)
    assert loss >= 0
    assert log2_perplexity >= 0
    assert loss < log2_perplexity
    assert output_probs.shape == (4, 11, 9)


def test_trainer__eval_step(
    trainer: Trainer, captionmodel: CaptionModel, nest_batch: DictNest
):
    (
        loss,
        log2_perplexity,
        output_probs,
    ) = trainer._eval_step(captionmodel, nest_batch)
    assert loss >= 0
    assert log2_perplexity >= 0
    assert loss < log2_perplexity
    assert output_probs.shape == (4, 11, 9)


def test_trainer__evaluate(
    trainer: Trainer, captionmodel: CaptionModel, nest_batch: DictNest
):
    def data_gen():
        for _ in range(10):
            batch = []
            for _ in range(nest_batch["feats"].shape[0]):
                batch.append(
                    {
                        "feat": [],
                        "sentence": {"raw_tokens": ""},
                    }
                )
            yield batch, nest_batch

    trainer._evaluate(captionmodel, data_gen())
    assert os.path.exists("logs/checkpoint/test_trainer/checkpoint_0_" "0_28.53.jxc")
    assert os.path.exists("logs/logger/test_trainer")

    checkpoint = pickle.load(
        open("logs/checkpoint/test_trainer/checkpoint_0_0_28.53.jxc", "rb")
    )
    for lyr in checkpoint.keys():
        for k in ["weight", "bias"]:
            np.testing.assert_equal(
                checkpoint[lyr][k], captionmodel.get_parameters()[lyr][k]
            )


def test_trainer__predict(
    trainer: Trainer, captionmodel: CaptionModel, nest_batch: DictNest
):
    batch = []
    for _ in range(nest_batch["feats"].shape[0]):
        batch.append(
            {
                "feat": [],
                "sentence": {"raw_tokens": ""},
            }
        )
    words, _ = trainer.predict(captionmodel, batch, nest_batch)
    assert all([isinstance(w, str) for w in words])


def test_trainer__set_state(trainer: Trainer):
    state = dict(
        split="train",
        loss=0.0,
        perplexity=1.9,
        mean_loss=0.0,
        mean_perplexity=2.1,
        predicted_words=["aminit", "jamisi"],
        gt_words=["ogbeni", "funmi"],
    )
    trainer._set_state(**state)

    assert len(trainer.history) == 1
    assert all([state[k] == trainer._state[k] for k in state.keys()])
    assert trainer.history[-1] == trainer._state


def test_trainer__update_state(trainer: Trainer):
    state = dict(
        split="train",
        loss=0.0,
        perplexity=1.9,
        mean_loss=0.0,
        mean_perplexity=2.1,
        predicted_words=["aminit", "jamisi"],
        gt_words=["ogbeni", "funmi"],
    )
    trainer._set_state(**state)
    trainer._update_state(mean_loss=765)
    assert len(trainer.history) == 1
    assert trainer.history[-1]["mean_loss"] == 765
    assert trainer._state["mean_loss"] == 765


def test_trainer__get_last_state(trainer: Trainer):
    state = dict(
        split="train",
        loss=0.0,
        perplexity=1.9,
        mean_loss=0.0,
        mean_perplexity=2.1,
        predicted_words=["aminit", "jamisi"],
        gt_words=["ogbeni", "funmi"],
    )
    trainer._set_state(**state)
    state["loss"] = 678
    trainer._set_state(**state)
    state["mean_loss"] = 879
    trainer._set_state(**state)
    last_state = trainer._get_last_state("train")
    assert len(trainer.history) == 3
    assert trainer.history[-1]["mean_loss"] == 879
    assert last_state["mean_loss"] == 879


def test_trainer__dump_history(trainer: Trainer):
    state = dict(
        split="train",
        loss=0.0,
        perplexity=1.9,
        mean_loss=0.0,
        mean_perplexity=2.1,
        predicted_words=["aminit", "jamisi"],
        gt_words=["ogbeni", "funmi"],
    )
    for _ in range(10):
        trainer._set_state(**state)
    trainer._dump_history()
    assert os.path.exists("logs/histories/test_trainer/history.json")
    loaded_history = json.load(open("logs/histories/test_trainer/history.json", "r"))
    assert len(loaded_history) == 10


def test_trainer__fit(
    trainer: Trainer, captionmodel: CaptionModel, nest_batch: DictNest
):
    class DP:
        def iter_image_sentence_pair_batch(self, split):
            for _ in range(10):
                batch = []
                for _ in range(nest_batch["feats"].shape[0]):
                    batch.append(
                        {
                            "feat": [],
                            "sentence": {"raw": ""},
                        }
                    )
                yield batch, nest_batch

    trainer.fit(model=captionmodel, data_provider=DP(), epochs=1, lr=1e-1)
