"""Predict from a previously generated song model."""
import argparse
import json
import os
import pickle

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text

from lyrics import util


def softmax_sampling(probabilities, randomness, seed=None):
    """Returns the index of the highest value from a softmax vector,
    with a bit of randomness based on the probabilities returned.

    """
    if seed:
        np.random.seed(seed)
    if randomness == 0:
        return np.argmax(probabilities)
    probabilities = np.asarray(probabilities).astype("float64")
    probabilities = np.log(probabilities) / randomness
    exp_probabilities = np.exp(probabilities)
    probabilities = exp_probabilities / np.sum(exp_probabilities)
    return np.argmax(np.random.multinomial(1, probabilities, 1))


def generate_lyrics(model, tokenizer, text_seed, song_length, randomness=0, seed=None):
    """Generate a new lyrics based on the given model, tokenizer, etc.

    Returns the final output as both a vector and a string.

    """
    # The sequence length is the second dimension of the input shape. If the
    # input shape is (None,), the model uses the transformer network which
    # takes a string as input!
    input_shape = model.inputs[0].shape
    seq_length = -1
    if len(input_shape) >= 2:
        print("Using integer sequences")
        seq_length = int(input_shape[1])
    else:
        print("Using string sequences")

    # Create a reverse lookup index for integers to words
    rev = {v: k for k, v in tokenizer.word_index.items()}

    spacer = "" if tokenizer.char_level else " "

    text_output = tokenizer.texts_to_sequences([text_seed])[0]
    text_output_str = spacer.join(rev.get(word) for word in text_output)
    while len(text_output) < song_length:
        if seq_length != -1:
            padded = tf.keras.preprocessing.sequence.pad_sequences(
                [text_output], maxlen=seq_length, padding="post"
            )
        else:
            padded = np.array([text_output_str])
        next_word = model.predict_on_batch(padded)
        next_word = softmax_sampling(next_word[0], randomness, seed=seed)
        text_output.append(next_word)
        text_output_str += f"{spacer}{rev.get(next_word)}"
    return text_output, text_output_str


def load_model(model_filename):
    return tf.keras.models.load_model(
        model_filename, custom_objects={"KerasLayer": hub.KerasLayer}
    )


def lyrics(args):
    model = load_model(args.model)

    tokenizer = util.load_tokenizer(args.tokenizer)

    print(f'Generating lyrics from "{args.text}"...')
    seed = (
        args.random_seed
        if args.random_seed
        else np.random.randint(np.iinfo(np.int32).max)
    )

    raw, text = generate_lyrics(
        model, tokenizer, args.text, args.length, args.randomness, seed=seed
    )
    if args.print_raw:
        print(raw)
    print(text)
    print()
    print(f"Random seed (for reproducibility): {seed}")


def export(args):
    import tensorflowjs as tfjs

    model = load_model(args.model)

    os.makedirs("./export/js", exist_ok=True)

    with open(args.tokenizer, "rb") as handle:
        tokenizer = pickle.load(handle)
        with open("./export/js/words.json", "w") as f:
            f.write(json.dumps(tokenizer.word_index))

    tfjs.converters.save_keras_model(model, "./export/js")


def cli():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    lyrics_parser = subparsers.add_parser(
        "lyrics", help="Make a new lyric based on the given trained lyrics model"
    )
    lyrics_parser.add_argument("model", help="The path to the Keras model to load")
    lyrics_parser.add_argument(
        "tokenizer", help="The path to the pickled tokenizer used for words"
    )
    lyrics_parser.add_argument(
        "--length",
        default=50,
        type=int,
        help="The maximum length (in characters) for the lyrics",
    )
    lyrics_parser.add_argument(
        "--text",
        default="hello there",
        help="The starting text for the lyrics. Different start provide different outcomes",
    )
    lyrics_parser.add_argument(
        "--randomness",
        default=0.0,
        type=float,
        help="""Probability variance (sometimes called "temperature") to apply
        when selecting words.  Can be larger than 1, but makes the most sense
        between 0 and 1.""",
    )
    lyrics_parser.add_argument(
        "--print-raw",
        action="store_true",
        help="Whether or not to print the raw song vector",
    )
    lyrics_parser.add_argument(
        "--random-seed",
        type=int,
        help="""Set a specific random seed for lyrics generation. Allows for
        reproducible results.""",
    )
    lyrics_parser.set_defaults(func=lyrics)

    export_parser = subparsers.add_parser(
        "export",
        help="Export a model and tokenizer to a format that Tensorflowjs can understand",
    )
    export_parser.add_argument("model", help="The path to the Keras model to export")
    export_parser.add_argument(
        "tokenizer", help="The path to the pickled tokenizer used for words"
    )
    export_parser.set_defaults(func=export)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    cli()
