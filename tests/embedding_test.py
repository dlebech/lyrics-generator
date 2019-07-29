import os
import tempfile

import numpy as np
import pytest
import tensorflow as tf

from lyrics import embedding, util


@pytest.fixture
def embedding_mapping():
    return {"woof": [0.1, 0.2, 0.3]}


def test_create_word2vec(export_dir, songfile):
    w2v = embedding.create_word2vec(
        export_dir,  # Use export directory fixture as data directory. It's ok :-)
        songdata_file=songfile,
        artists=None,
        embedding_dim=5,
    )

    # It excludes newlines for this example
    assert len(w2v.wv.vocab) == 3
    assert len(w2v.wv["woof"]) == 5

    with open(export_dir + "/word2vec.txt") as f:
        print(f.read())


def test_create_embedding_mappings(embedding_file):
    """It should create a dictionary of embedding mappings."""
    embedding_mapping = embedding.create_embedding_mappings(
        embedding_file=embedding_file
    )
    assert embedding_mapping == {"woof": [0.1, 0.2, 0.3]}


def test_create_embedding_matrix(songs, embedding_mapping):
    """It should create a dictionary of embedding mappings."""
    num_words = 2
    tokenizer = util.prepare_tokenizer(songs, num_words=num_words)

    embedding_matrix = embedding.create_embedding_matrix(
        tokenizer, embedding_mapping, max_num_words=num_words, embedding_dim=3
    )

    # Only woof is known
    np.testing.assert_array_equal(
        embedding_matrix,
        [
            [0, 0, 0],  # OOV
            [0, 0, 0],  # \n
            [0.1, 0.2, 0.3]  # woof
            # [0, 0, 0], # meow, absent, because we only choose 2 words
            # [0, 0, 0], # chorus, absent, same reason
        ],
    )
