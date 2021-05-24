"""Shared Utility functions."""
import datetime
import pickle
import re

import pandas as pd
import tensorflow as tf
from better_profanity import profanity

from . import config

star_matcher = re.compile(r"\*")

# Match present participle shortened with apostrophes like singin', workin'.
ing_matcher = re.compile(r"(\win)'(?!\w)(\s)?")

# Map a bunch of words to their shorter form so they will be treated as a
# single token. They need to be sorted such that they don't conflict with each
# other.
sorted_word_pairs = [
    # Make sure to not catch edge cases such as the island
    (re.compile(r"(?<!\w)he is(?!\w)"), "he's"),
    (re.compile(r"she is(?!\w)"), "she's"),
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


def load_songdata(songdata_file, artists):
    print("Loading song data from {}".format(songdata_file))
    songdata = pd.read_csv(songdata_file)
    songdata = songdata[~songdata.text.isnull()]

    # Find all songs from the selected artists
    if artists:
        songdata = songdata[songdata.artist.isin(artists)]

    return songdata.text.values


def _remove_repeats(song, max_repeats):
    seqs = song.split("\n")
    seq_index = 0
    final_song_sequences = []

    while seq_index < len(seqs):
        seq = seqs[seq_index].strip()
        final_song_sequences.append(seq)
        seq_index += 1
        counter = 0

        while seq_index < len(seqs):
            next_seq = seqs[seq_index].strip()
            if seq != next_seq:
                break

            counter += 1

            if counter < max_repeats:
                final_song_sequences.append(next_seq)
            seq_index += 1

    return " \n ".join(final_song_sequences)


def _clean_song(song, transform_words, max_repeats, profanity_censor=False):
    song = star_matcher.sub("", song)
    song = song.lower().strip("\n")
    song = _remove_repeats(song, max_repeats)

    if profanity_censor:
        song = profanity.censor(song)

    if not transform_words:
        return song

    # Replace singin' with singing etc.
    song = ing_matcher.sub(r"\1g\2", song)

    # Replace cannot with can't etc.
    for pair in sorted_word_pairs:
        song = re.sub(pair[0], pair[1], song)

    return song


def prepare_songs(
    songs, transform_words=False, max_repeats=config.MAX_REPEATS, profanity_censor=False
):
    """Do pre-cleaning of all songs in the given array."""
    # Put whitespace around each newline character so something like \nhello is
    # not treated as a word but newline characters are still preserved by
    # themselves
    # Also fix endings with in' by adding a g. This reduces the vocabulary size.
    print("Preparing proper newlines")
    if transform_words:
        print("... also transforming words")
    if profanity_censor:
        print("... also censoring profanity")
    now = datetime.datetime.now()
    songs = [
        _clean_song(
            song,
            transform_words=transform_words,
            max_repeats=max_repeats,
            profanity_censor=profanity_censor,
        )
        for song in songs
    ]
    print("Took {}".format(datetime.datetime.now() - now))

    return songs


def prepare_tokenizer(songs, num_words=config.MAX_NUM_WORDS, char_level=False):
    """Prepare the song tokenizer. Uses Keras' tokenizer"""
    # Create tokenizer and remove newline character from the filters so it's treated as a word
    # Use +1 in the number of words to include the OOV (0) word
    tokenizer = tf.keras.preprocessing.text.Tokenizer(
        num_words=num_words + 1, char_level=char_level
    )
    # Allow * and newlines. The stars are used for profanity censoring, if that
    # option is chosen during data preparation.
    tokenizer.filters = tokenizer.filters.replace("\n", "").replace("*", "")

    # Fit on the texts and convert the data to integer sequences
    print("Fitting tokenizer to texts")
    now = datetime.datetime.now()
    tokenizer.fit_on_texts(songs)
    print("Took {}".format(datetime.datetime.now() - now))
    return tokenizer
