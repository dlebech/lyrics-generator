"""End to end train testing."""
import os

import numpy as np

from lyrics import train


# Created by Gene Lyrica One :-)
# Modified slightly from original
longer_song = """
hello world is a dream
i know when i been like your love
and i can't go home

i cannot cry
i don't want to see
i don't know why i can't run
i got me
i got me
i got me
"""


def test_prepare_data_transform_words():
    """It should prepare song data and transform words"""
    songs = [longer_song]
    x, y, seq_length, num_words, tokenizer = train.prepare_data(
        songs, transform_words=True
    )

    # Average length is five words
    # Median length is also five words
    # We should thus expect a sequence length of 23 (4 sentences + 3 newline character)
    # Note that the default max repeats 2 removes the last sentence
    assert seq_length == 23

    # There are 25 words plus the newline character...
    assert num_words == 26

    # The newline character should be in the tokenizer's word index
    assert "\n" in tokenizer.word_index

    # "Cannot" should not exist anymore because of the transformed words
    assert "cannot" not in tokenizer.word_index
    assert "Cannot" not in tokenizer.word_index
    assert "can't" in tokenizer.word_index

    # The first X will contain just the "hello" word and the target would be "world"
    hello = tokenizer.word_index["hello"]
    world = tokenizer.word_index["world"]
    assert x[0][0] == hello
    assert y[0] == world

    # The last sequence should end with i can't run \n i got me i got because
    # the last "me" is a target and the the repeat sentence should be deleted.
    assert tokenizer.sequences_to_texts(x[-1:])[0].endswith("i can't run \n i got me \n i got")


def test_prepare_data_use_full_sentences():
    """It should prepare song data and use full sentences"""
    songs = [longer_song]
    x, y, seq_length, num_words, tokenizer = train.prepare_data(
        songs, transform_words=True, use_full_sentences=True
    )

    # Basic assumption are the same as the previous test.
    assert seq_length == 23
    assert num_words == 26

    # The first X will contain no zeros
    hello = tokenizer.word_index["hello"]
    world = tokenizer.word_index["world"]
    cant = tokenizer.word_index["can't"]
    me = tokenizer.word_index["me"]
    assert np.all(x[0])

    # "can't" is the 24th word so with a sequence length of 23, it should be the first output.
    assert x[0][0] == hello
    assert x[0][1] == world
    assert y[0] == cant

    # "world" should be the first word in the second input sentence.
    assert x[1][0] == world

    # "me" is the last word so it should be the output of the last sequence...
    assert y[-1] == me


def test_prepare_data_use_strings():
    """It should prepare song data and return sentences strings"""
    songs = [longer_song]
    x, y, seq_length, num_words, tokenizer = train.prepare_data(
        songs, transform_words=True, use_full_sentences=True, use_strings=True
    )

    # Basic assumption are the same as the previous test.
    assert seq_length == 23
    assert num_words == 26

    # "can't" is the 24th word so with a sequence length of 23, it should be the first output.
    cant = tokenizer.word_index["can't"]
    assert y[0] == cant

    # Make sure we have strings
    assert type(x[0]) == str
    assert (
        x[0]
        == "hello world is a dream \n i know when i been like your love \n and i can't go home \n \n i"
    )


def test_prepare_data_num_lines():
    """It should prepare song data and take a specific number of lines"""
    songs = [longer_song]
    x, y, seq_length, num_words, tokenizer = train.prepare_data(
        songs, num_lines_to_include=5
    )

    # Average length is five words
    # Median length is also five words
    # We should thus expect a sequence length of 29 (5 sentences + 4 newline character)
    assert seq_length == 29


def test_prepare_data_char_level():
    """It should prepare song data and character level data"""
    songs = [longer_song]
    x, y, seq_length, num_words, tokenizer = train.prepare_data(
        songs, char_level=True
    )

    # The vocabulary is slightly smaller than the above test because it consists
    # of characters.
    assert num_words == 23

    # The first X will contain no zeros
    h = tokenizer.word_index["h"]
    e = tokenizer.word_index["e"]
    assert x[0][0] == h
    assert y[0] == e

    # "me" is the last word so it should be the output of the last sequence...
    assert y[-1] == e


def test_train(export_dir, embedding_file, songfile):
    """It should train and save a model and tokenizer."""
    train.train(
        max_epochs=1,
        export_dir=export_dir,
        songdata_file=songfile,
        artists=["cat", "dog"],
        embedding_file=embedding_file,
        embedding_dim=3,
        save_freq="epoch",
    )
    assert os.path.exists("{}/model.h5".format(export_dir))
    assert os.path.exists("{}/tokenizer.pickle".format(export_dir))
