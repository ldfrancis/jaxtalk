"""Main application module """

import argparse
import logging
import pickle

from jaxtalk.data import BasicDataProvider
from jaxtalk.models import CaptionModel
from jaxtalk.trainer import Trainer
from jaxtalk.utils import Logger, compute_bias_init_vector, create_vocabulary


def parse_args() -> argparse.Namespace:
    """Parse command line arguments
    taken from neuraltalk
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset",
        dest="dataset",
        default="flickr8k",
        help="dataset: flickr8k/flickr30k",
    )
    parser.add_argument(
        "--init_model_from",
        dest="init_model_from",
        type=str,
        default="",
        help="initialize the model parameters from some specific checkpoint?",
    )
    parser.add_argument(
        "--trainer_name",
        dest="trainer_name",
        type=str,
        default="caption_trainer",
        help="A name for the trainer",
    )
    parser.add_argument(
        "--image_encoding_size",
        dest="image_encoding_size",
        type=int,
        default=256,
        help="size of the image encoding",
    )
    parser.add_argument(
        "--image_features_size",
        dest="image_features_size",
        type=int,
        default=4096,
        help="size of extracted image features",
    )
    parser.add_argument(
        "--word_encoding_size",
        dest="word_encoding_size",
        type=int,
        default=256,
        help="size of word encoding",
    )
    parser.add_argument(
        "--hidden_size",
        dest="hidden_size",
        type=int,
        default=256,
        help="size of hidden layer in generator RNNs",
    )
    parser.add_argument(
        "-c",
        "--regc",
        dest="regc",
        type=float,
        default=1e-8,
        help="regularization strength",
    )
    parser.add_argument(
        "-l",
        "--learning_rate",
        dest="learning_rate",
        type=float,
        default=1e-3,
        help="solver learning rate",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        dest="batch_size",
        type=int,
        default=16,
        help="batch size",
    )
    parser.add_argument(
        "--grad_clip",
        dest="grad_clip",
        type=float,
        default=1,
        help=(
            "clip gradients (normalized by batch size)? elementwise. if positive,"
            " at what threshold?"
        ),
    )
    parser.add_argument(
        "--max_length",
        dest="max_length",
        type=int,
        default=18,
        help="The maximum number of words to be generated in a sentence",
    )
    parser.add_argument(
        "--word_count_threshold",
        dest="word_count_threshold",
        type=int,
        default=5,
        help=(
            "if a word occurs less than this number of times in training data, it is"
            " discarded"
        ),
    )
    parser.add_argument(
        "--epochs",
        dest="epochs",
        type=int,
        default=5,
        help=("number of epochs to train for"),
    )
    parser.add_argument(
        "-ee",
        "--evaluate_every",
        dest="evaluate_every",
        type=float,
        default=1.0,
        help="in units of epochs, how often do we evaluate on val set?",
    )

    return parser.parse_args()


def main(args):
    """main application driver"""
    logger = Logger("main", level=logging.INFO)
    data_provider = BasicDataProvider(
        args.dataset, args.word_count_threshold, args.batch_size, args.max_length
    )
    logger.info("Created data provider")
    wordtoix, ixtoword, word_counts, num_sents, _ = create_vocabulary(
        data_provider.iter_sentences("train"), args.word_count_threshold
    )
    bias_init_vector = compute_bias_init_vector(ixtoword, word_counts, num_sents)
    model = CaptionModel(
        args.word_encoding_size,
        args.hidden_size,
        args.image_features_size,
        wordtoix,
        ixtoword,
        args.max_length,
    )
    logger.info("Create model")
    model.word_decoder.update_parameters({"bias": bias_init_vector})
    logger.info("Reinitialized bias word decoder bias vector")
    if args.init_model_from:
        checkpoint = pickle.load(open(args.init_model_from, "rb"))
        model.set_parameters(checkpoint)
        logger.info(f"Loaded checkpoint from {args.init_model_from}")
    trainer = Trainer(
        ixtoword=ixtoword,
        grad_clip=args.grad_clip,
        regc=args.regc,
        evaluate_every=args.evaluate_every,
        name=args.trainer_name,
    )
    logger.info("Created trainer")
    trainer.fit(model, data_provider, epochs=args.epochs, lr=args.learning_rate)
    logger.info("Done!")


if __name__ == "__main__":
    args = parse_args()
    main(args)
