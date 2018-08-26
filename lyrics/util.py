"""Shared Utility functions."""
import csv
import datetime
import pickle

import pandas as pd
import tensorflow as tf

from . import config


def pickle_tokenizer(tokenizer, export_dir):
    with open('{}/tokenizer.pickle'.format(export_dir), 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_tokenizer(tokenizer_path):
    with open(tokenizer_path, 'rb') as f:
        return pickle.load(f)


def load_songdata(songdata_file=config.SONGDATA_FILE, artists=config.ARTISTS):
    print('Loading song data from {}'.format(songdata_file))
    songdata = pd.read_csv(songdata_file)

    # Find all songs from the selected artists
    if artists:
        songdata = songdata[songdata.artist.isin(artists)]

    return songdata.text.values


def prepare_songs(songs):
    """Do pre-cleaning of all songs in the given array."""
    # Put whitespace around each newline character so something like \nhello is
    # not treated as a word but newline characters are still preserved by
    # themselves
    print('Preparing proper newlines')
    now = datetime.datetime.now()
    songs = [song.strip('\n').replace('\n', ' \n ') for song in songs]
    print('Took {}'.format(datetime.datetime.now() - now))
    return songs


def prepare_tokenizer(songs, num_words=config.MAX_NUM_WORDS):
    """Prepare the song tokenizer. Uses Keras' tokenizer"""
    # Create tokenizer and remove newline character from the filters so it's treated as a word
    # Use +1 in the number of words to include the OOV (0) word
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=num_words + 1) 
    tokenizer.filters = tokenizer.filters.replace('\n', '')

    # Fit on the texts and convert the data to integer sequences
    print('Fitting tokenizer to texts')
    now = datetime.datetime.now()
    tokenizer.fit_on_texts(songs)
    print('Took {}'.format(datetime.datetime.now() - now))
    return tokenizer