"""Embedding utilities."""
import argparse
import csv
import datetime
import multiprocessing

import numpy as np
import pandas as pd
import tensorflow as tf
from gensim.models import Word2Vec

from . import config, util

_num_workers = 1
try:
    _num_workers = multiprocessing.cpu_count()
except:
    pass


def create_word2vec(
    data_dir="./data",
    songdata_file=config.SONGDATA_FILE,
    artists=config.ARTISTS,
    embedding_dim=config.EMBEDDING_DIM,
    name_suffix="",
):
    songs = util.load_songdata(songdata_file=songdata_file, artists=artists)
    songs = util.prepare_songs(songs)

    sequences = []
    for song in songs:
        sequences.append(tf.keras.preprocessing.text.text_to_word_sequence(song))

    # Initializing the model also starts training
    print(
        "Training Word2Vec on {} sequences with {} workers".format(
            len(sequences), _num_workers
        )
    )
    now = datetime.datetime.now()
    model = Word2Vec(sequences, size=embedding_dim, workers=_num_workers, min_count=1)
    print("Took {}".format(datetime.datetime.now() - now))

    # Save the model both in the gensim format but also as a text file, similar
    # to the glove embeddings
    model.save(f"{data_dir}/word2vec{name_suffix}.model")
    with open(f"{data_dir}/word2vec{name_suffix}.txt", "w") as f:
        writer = csv.writer(f, delimiter=" ", quoting=csv.QUOTE_MINIMAL)
        for word in model.wv.vocab:
            word_vector = model.wv[word]
            writer.writerow([word] + ["{:.5f}".format(i) for i in word_vector])

    return model


def create_embedding_mappings(embedding_file=config.EMBEDDING_FILE):
    """Create a lookup dictionary for word embeddings from the given embeddings file."""
    print("Loading embedding mapping file {}".format(embedding_file))
    embedding = pd.read_table(
        embedding_file, sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE
    )
    mapping = {}

    print("Creating embedding mappings for faster lookup")
    for row in embedding.itertuples():
        mapping[row[0]] = list(row[1:])

    return mapping


def create_embedding_matrix(
    tokenizer,
    embedding_mapping,
    embedding_dim=config.EMBEDDING_DIM,
    max_num_words=config.MAX_NUM_WORDS,
):
    """Create an embedding matrix from the given keras tokenizer and embedding mapping dictionary.

    The embedding matrix can be used as weights for an embedding layer in Keras.

    The function ensures that that only the top N words get selected for the embedding.

    """
    print("Creating embedding matrix")

    # Create embedding matrix, add an extra row for the out of vocabulary vector
    embedding_matrix = np.zeros((max_num_words + 1, embedding_dim))

    num_words = 0
    num_words_found = 0
    num_words_ignored = 0

    now = datetime.datetime.now()
    for word, i in tokenizer.word_index.items():
        if i > max_num_words:
            num_words_ignored += 1
            continue
        num_words += 1
        if word in embedding_mapping:
            embedding_matrix[i] = embedding_mapping[word]
            num_words_found += 1
    print("Took {}".format(datetime.datetime.now() - now))
    print(
        "Found {} words in mapping ({:.1%})".format(
            num_words_found, num_words_found / num_words
        )
    )
    print("{} words were ignored because they are infrequent".format(num_words_ignored))

    return embedding_matrix


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name-suffix", default="", help="Name suffix for the embedding file."
    )
    parser.add_argument(
        "--songdata-file",
        default=config.SONGDATA_FILE,
        help="Use a custom songdata file",
    )
    parser.add_argument(
        "--artists",
        default=config.ARTISTS,
        help="""
            A list of artists to use. Use '*' (quoted) to include everyone.
            The default is a group of rock artists.
        """,
        nargs="*",
    )
    args = parser.parse_args()
    artists = args.artists if args.artists != ["*"] else []
    create_word2vec(
        name_suffix=args.name_suffix, artists=artists, songdata_file=args.songdata_file
    )
