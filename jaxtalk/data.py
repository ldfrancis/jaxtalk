"""Andrej Karpathy's data provider implementation.
As implemented in neuraltalk
https://github.com/karpathy/neuraltalk
"""
import json
import os
import random
from typing import Any, DefaultDict, Dict, Generator, List

import jax.numpy as jnp
import numpy as np
import scipy.io

from jaxtalk.utils import Logger, create_vocabulary


class BasicDataProvider:
    """A data provider for that provides in form of image captions pairs"""

    def __init__(
        self, dataset_name: str, word_count_threshold: int, batch_size, max_sentence_len
    ) -> None:
        self._logger = Logger(f"BasicDataProvider_{dataset_name}")
        self._logger.info("Initializing data provider for dataset {dataset_name}...")

        self._batch_size = batch_size
        self._max_sentence_len = max_sentence_len

        # !assumptions on folder structure
        self.dataset_root = os.path.join("data", dataset_name)
        self.image_root = os.path.join("data", dataset_name, "imgs")

        # load the dataset into memory
        dataset_path = os.path.join(self.dataset_root, "dataset.json")
        self._logger.info(f"reading {dataset_path}")
        self.dataset = json.load(open(dataset_path, "r"))

        # load the image features into memory
        features_path = os.path.join(self.dataset_root, "vgg_feats.mat")
        self._logger.info(f"reading {features_path}")
        features_struct = scipy.io.loadmat(features_path)
        self.features = features_struct["feats"]

        # group images by their train/val/test split into a dictionary -> list structure
        self.split = DefaultDict(list)
        self.img_sent_pairs: Dict[str, List[Dict[str, Any]]] = DefaultDict(list)
        for img in self.dataset["images"]:
            self.split[img["split"]].append(img)
            img = self._get_image(img)
            self.img_sent_pairs[img["split"]].extend(
                [
                    {
                        "image": {
                            "imgid": img["imgid"],
                            "filename": img["filename"],
                            "feat": img["feat"],
                            "local_file_path": img["local_file_path"],
                        },
                        "sentence": self._get_sentence(sent),
                    }
                    for sent in img["sentences"]
                ]
            )

        (
            self.wordtoix,
            self.ixtoword,
            self.word_counts,
            self.num_sents,
            self.sentence_lengths,
        ) = create_vocabulary(self.iter_sentences("train"), word_count_threshold)

    def _get_image(self, img: Dict[str, Any]) -> Dict[str, Any]:
        """create an image structure for the driver"""
        # lazily fill in some attributes
        if "local_file_path" not in img:
            img["local_file_path"] = os.path.join(self.image_root, img["filename"])
        if "feat" not in img:  # also fill in the features
            feature_index = img[
                "imgid"
            ]  # NOTE: imgid is an integer, and it indexes into features
            img["feat"] = self.features[:, feature_index]
        return img

    def _get_sentence(self, sent: str) -> str:
        """create a sentence structure for the driver"""
        # NOOP for now
        return sent

    def get_split_size(self, split: str, ofwhat: str = "sentences") -> int:
        """return size of a split, either number of sentences or number of images"""
        if ofwhat == "sentences":
            return sum(len(img["sentences"]) for img in self.split[split])
        else:  # assume images
            return len(self.split[split])

    def sample_image_sentence_pair(self, split: str = "train") -> Dict[str, Any]:
        """sample image sentence pair from a split"""
        images = self.split[split]

        img = random.choice(images)
        sent = random.choice(img["sentences"])

        out: Dict[str, Any] = {}
        out["image"] = self._get_image(img)
        out["sentence"] = self._get_sentence(sent)
        return out

    def iter_image_sentence_pair(self, split: str = "train") -> Generator:
        """Obtain image sentence pair dictionary"""
        random.shuffle(self.img_sent_pairs[split])
        for data in enumerate(self.img_sent_pairs[split]):
            yield data

    def iter_image_sentence_pair_batch(
        self,
        split: str = "train",
    ) -> Generator:
        """Obtain list of image sentence pair dictionary"""
        batch = []
        feats = []
        tokens = []
        random.shuffle(self.img_sent_pairs[split])
        for data in self.img_sent_pairs[split]:

            feats += [data["image"]["feat"]]
            _tokens = (
                [
                    self.wordtoix[word]
                    for word in data["sentence"]["tokens"]
                    if word in self.wordtoix
                ]
                + [0]
            )[: self._max_sentence_len]
            num_tokens = len(_tokens)
            num_pad_tokens = self._max_sentence_len - num_tokens
            _tokens += [self.wordtoix["#PAD#"]] * num_pad_tokens
            tokens += [np.array(_tokens)]
            batch.append(data)
            if len(batch) >= self._batch_size:
                feats = jnp.stack(feats, axis=0)
                tokens = jnp.stack(tokens, axis=0)
                array_batch = {"feats": feats, "tokens": tokens}
                yield batch, array_batch
                batch = []
                feats = []
                tokens = []
        if batch:
            feats = jnp.stack(feats, axis=0)
            tokens = jnp.stack(tokens, axis=0)
            array_batch = {"feats": feats, "tokens": tokens}
            yield batch, array_batch

    def iter_sentences(self, split: str = "train") -> Generator:
        """Obtain sentences"""
        for img in self.split[split]:
            for sent in img["sentences"]:
                yield self._get_sentence(sent)

    def iter_images(
        self, split: str = "train", shuffle: bool = False, max_images=-1
    ) -> Generator:
        """Obtain images"""
        imglist = self.split[split]
        indices: List = list(range(len(imglist)))
        if shuffle:
            random.shuffle(indices)
        if max_images > 0:
            indices = indices[: min(len(indices), max_images)]  # crop the list
        for i in indices:
            yield self._get_image(imglist[i])
