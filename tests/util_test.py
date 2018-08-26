"""Test the utility module."""
import os
import pickle
import shutil
import tempfile

import pandas as pd
import pytest
import tensorflow as tf

from lyrics import util


def test_pickle_load_tokenizer(export_dir, songs):
    """It should pickle and unpickle a tokenizer."""
    tokenizer = tf.keras.preprocessing.text.Tokenizer() 
    tokenizer.fit_on_texts(songs)
    util.pickle_tokenizer(tokenizer, export_dir)
    tokenizer = util.load_tokenizer('{}/tokenizer.pickle'.format(export_dir))
    assert 'woof' in tokenizer.word_index


def test_load_songdata(songfile):
    """It should return an array of songs."""
    songs = util.load_songdata(songdata_file=songfile, artists=['cat', 'dog'])
    assert len(songs) == 2
    assert songs[0] == '\nmeow\nmeow'
    assert songs[1] == 'woof\n\nchorus\nwoof\n'


def test_load_songdata_limit_artists(songfile):
    """It should only return the requested artists."""
    songs = util.load_songdata(songdata_file=songfile, artists=['dog'])
    assert len(songs) == 1
    assert songs[0] == 'woof\n\nchorus\nwoof\n'


def test_prepare_songs(songs_raw):
    """It should strip newlines at beginning and end but preserve newlines in the middle."""
    songs = util.prepare_songs(songs_raw)
    assert songs[0] == 'meow \n meow'
    assert songs[1] == 'woof \n  \n chorus \n woof'


def test_prepare_tokenizer(songs):
    """It should tokenize newlines and include all words."""
    tokenizer = util.prepare_tokenizer(songs)
    assert len(tokenizer.word_index) == 4
    assert tokenizer.word_index == {
        '\n': 1,
        'woof': 2,
        'meow': 3,
        'chorus': 4
    }

    sentences = tokenizer.texts_to_sequences(songs)

    # The songs fixture has been carefully crafted, didn't you notice? :-)
    # 0 is reserved, 1 is newline, 2 is woof, 3 is meow, 4 is chorus
    assert sentences[0] == [3, 1, 3]
    assert sentences[1] == [2, 1, 1, 4, 1, 2, 2]


def test_prepare_tokenizer_limit_words(songs):
    """It should tokenize newlines."""
    tokenizer = util.prepare_tokenizer(songs, num_words=2)

    # So interestingly, keras keeps track of all words. It's not until turning
    # sentences into sequences that the num_words parameter kicks in
    assert len(tokenizer.word_index) == 4

    sentences = tokenizer.texts_to_sequences(songs)

    # 0 is reserved, 1 is newline, 2 is woof, the others are not included so they will be 0
    assert sentences[0] == [1]
    assert sentences[1] == [2, 1, 1, 1, 2, 2]