"""Shared Utility functions."""
import csv
import datetime
import pickle
import re

import pandas as pd
import tensorflow as tf

from . import config


ing_matcher = re.compile(r"(\win)'(?!\w)(\s)?")

# Map a bunch of words to their shorter form so they will be treated as a
# single token. They need to be sorted such that they don't conflict with each
# other.
sorted_word_pairs = [
    # Make sure to not catch edge cases such as the island
    (re.compile(r"(?<!\w)he is"), "he's"),
    ("she is", "she's"),
    ("cannot", "can't"),
    ("they are", "they're"),
    ("we are", "we're"),
]


def pickle_tokenizer(tokenizer, export_dir):
    with open("{}/tokenizer.pickle".format(export_dir), "wb") as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_tokenizer(tokenizer_path):
    with open(tokenizer_path, "rb") as f:
        return pickle.load(f)


def load_songdata(songdata_file=config.SONGDATA_FILE, artists=config.ARTISTS):
    print("Loading song data from {}".format(songdata_file))
    songdata = pd.read_csv(songdata_file)

    # Find all songs from the selected artists
    if artists:
        songdata = songdata[songdata.artist.isin(artists)]

    return songdata.text.values


def _clean_song(song, transform_words=False):
    song = song.lower().strip("\n").replace("\n", " \n ")
    if not transform_words:
        return song

    song = ing_matcher.sub(r"\1g\2", song)
    for pair in sorted_word_pairs:
        # Long to short
        song = re.sub(pair[0], pair[1], song)
    return song


def prepare_songs(songs, transform_words=False):
    """Do pre-cleaning of all songs in the given array."""
    # Put whitespace around each newline character so something like \nhello is
    # not treated as a word but newline characters are still preserved by
    # themselves
    # Also fix endings with in' by adding a g. This reduces the vocabulary size.
    print("Preparing proper newlines")
    if transform_words:
        print("... also transforming words")
    now = datetime.datetime.now()
    songs = [_clean_song(song, transform_words=transform_words) for song in songs]
    print("Took {}".format(datetime.datetime.now() - now))

    return songs


def prepare_tokenizer(songs, num_words=config.MAX_NUM_WORDS):
    """Prepare the song tokenizer. Uses Keras' tokenizer"""
    # Create tokenizer and remove newline character from the filters so it's treated as a word
    # Use +1 in the number of words to include the OOV (0) word
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=num_words + 1)
    tokenizer.filters = tokenizer.filters.replace("\n", "")

    # Fit on the texts and convert the data to integer sequences
    print("Fitting tokenizer to texts")
    now = datetime.datetime.now()
    tokenizer.fit_on_texts(songs)
    print("Took {}".format(datetime.datetime.now() - now))
    return tokenizer
