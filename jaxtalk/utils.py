import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Generator, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np


class Logger(logging.Logger):
    def __init__(self, name, level=logging.INFO):
        super().__init__(name, level)
        self._name = f"logs/logger/{name}"
        self.setup()

    def setup(self):
        # formatter
        formatter = logging.Formatter(
            "%(module)s.py:%(lineno)d|%(levelname)s" "|%(name)s: %(message)s"
        )

        # file handler
        logfile = Path(f"{self._name}/log_{get_datetime()}.txt")
        logfile.parent.mkdir(parents=True, exist_ok=True)
        filehandler = logging.FileHandler(logfile)
        filehandler.setFormatter(formatter)

        # stream handler
        streamhandlerStdOut = logging.StreamHandler(sys.stdout)
        streamhandlerStdErr = logging.StreamHandler(sys.stderr)
        streamhandlerStdOut.setFormatter(formatter)
        streamhandlerStdErr.setFormatter(formatter)

        # add handlers
        self.addHandler(filehandler)
        self.addHandler(streamhandlerStdOut)
        self.setLevel("DEBUG")


def get_datetime():
    format = "%Y_%m_%d_%H_%M_%S"
    now = datetime.now()
    dtime = now.strftime(format)
    return dtime


def create_vocabulary(
    sentence_gen: Generator, word_count_threshold: int
) -> Tuple[Dict[str, int], Dict[int, str], Dict[str, int], int, List[int]]:
    """Create vocabulary based on the implementation in Andrej Karpathy's neuraltalk
    https://github.com/karpathy/neuraltalk
    preProBuildWordVocab(sentence_iterator, word_count_threshold)
    """
    logger = Logger("create_vocabulary")
    logger.info(
        "preprocessing word counts and creating vocab based on word count"
        f" threshold {word_count_threshold}"
    )
    start_time: float = time.time()
    word_counts: Dict[str, int] = {}
    num_sents: int = 0
    sentence_lengths = []
    for sent in sentence_gen:
        num_sents += 1
        sentence_lengths += [len(sent["tokens"])]
        for word in sent["tokens"]:
            word_counts[word] = word_counts.get(word, 0) + 1
    vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
    logger.info(
        f"filtered words from {len(word_counts)} to {len(vocab)} in "
        f"{time.time()-start_time:.2f}"
    )

    # with K distinct words:
    # - there are K+1 possible inputs (START token and all the words)
    # - there are K+1 possible outputs (END token and all the words)
    # we use ixtoword to take predicted indeces and map them to words for output
    # visualization
    # we use wordtoix to take raw words and get their index in word vector matrix
    ixtoword = {}
    ixtoword[
        0
    ] = "."  # period at the end of the sentence. make first dimension be end token
    wordtoix = {}
    wordtoix["#START#"] = 0  # make first vector be the start token
    idx = 1
    for w in vocab:
        wordtoix[w] = idx
        ixtoword[idx] = w
        idx += 1
    wordtoix["#PAD#"] = len(wordtoix)
    ixtoword[len(ixtoword)] = "#PAD#"

    return wordtoix, ixtoword, word_counts, num_sents, sentence_lengths


def compute_bias_init_vector(
    ixtoword: Dict[int, str],
    word_counts: Dict[str, int],
    num_sents: int,
) -> np.ndarray:
    """Based on the implementation in neuraltalk by Andrej Karpathy
    https://github.com/karpathy/neuraltalk
    preProBuildWordVocab(sentence_iterator, word_count_threshold)
    """
    # compute bias vector, which is related to the log probability of the distribution
    # of the labels (words) and how often they occur. We will use this vector to
    # initialize the decoder weights, so that the loss function doesnt show a huge
    # increase in performance very quickly (which is just the network learning this
    # anyway, for the most part). This makes he visualizations of the cost function
    # nicer because it doesn't look like a hockey stick. for example on Flickr8K, doing
    # this brings down initial perplexity from ~2500 to ~170.
    word_counts["."] = num_sents
    bias_init_vector = np.array(
        [1.0 * word_counts[ixtoword[i]] for i in ixtoword if ixtoword[i] in word_counts]
    )
    bias_init_vector /= np.sum(bias_init_vector)  # normalize to frequencies
    bias_init_vector = np.log(bias_init_vector)
    bias_init_vector -= np.max(bias_init_vector)  # shift to nice numeric range

    return bias_init_vector


def init_array(key, *args):
    """Initialize an array with shape provided by *args
    https://github.com/karpathy/neuraltalk
    initw(n,d)
    """
    return jax.random.uniform(key, args, minval=-1, maxval=1) * 0.05


def init_bias(*args):
    """Initialize an array with shape provided by *args
    https://github.com/karpathy/neuraltalk
    initw(n,d)
    """
    return jnp.zeros(args)


class Mean:
    def __init__(self) -> None:
        self._val = 0
        self._counter = 0
        self.reset()

    def update(self, val):
        self._counter += 1

        self._val = self._val + (1.0 / self._counter) * (val - self._val)

        return self._val

    def reset(self):
        self._counter = 0
        self._val = 0

    def state(self):
        return self._val

    def __repr__(self):
        return f"{self._val}"
