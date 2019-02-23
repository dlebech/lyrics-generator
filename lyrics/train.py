"""Train a song generating model."""
import argparse
import csv
import datetime
import os
import statistics

import numpy as np
import pandas as pd
import tensorflow as tf

from . import config, embedding, util


def prepare_data(songs, transform_words=False):
    songs = util.prepare_songs(songs, transform_words=transform_words)
    tokenizer = util.prepare_tokenizer(songs)

    num_words = min(config.MAX_NUM_WORDS, len(tokenizer.word_index))

    print('Encoding all songs to integer sequences')
    now = datetime.datetime.now()
    songs_encoded = tokenizer.texts_to_sequences(songs)
    print('Took {}'.format(datetime.datetime.now() - now))
    print()

    # Find the newline integer
    newline_int = tokenizer.word_index['\n']

    # Calculate the average length of each sentence before a newline is seen.
    # This is probably between 5 and 10 words for most songs.
    # It will guide the verse structure.
    line_lengths = []
    print('Find the average line length for all songs')
    now = datetime.datetime.now()
    for song_encoded in songs_encoded:
        # Find the index of the newline character.
        # For double newlines (between verses), the distance will be 1
        newline_indexes = np.where(np.array(song_encoded) == newline_int)[0]
        lengths = [
            newline_indexes[i] - newline_indexes[i-1]
            for i in range(1, len(newline_indexes))
            if newline_indexes[i] - newline_indexes[i-1] > 1
        ]
        line_lengths.extend(lengths)
    print('Took {}'.format(datetime.datetime.now() - now))
    print()

    median_seq_length = statistics.median(line_lengths)
    mean_seq_length = statistics.mean(line_lengths)
    print('Median/mean line length from {} lines: {}/{}'
          .format(len(line_lengths), median_seq_length, mean_seq_length))
    print()

    # Prepare input data based on the median sequence length
    # Take two average linex (hence the multiplication by 2)
    seq_length = int(round(median_seq_length)) * 2

    # Prepare data for training
    X, y = [], []
    print('Creating test data')
    now = datetime.datetime.now()
    for song_encoded in songs_encoded:
        for i in range(1, len(song_encoded)):
            seq = song_encoded[:i]
            # Manually pad/slice the sequences to the proper length
            # This avoids an expensive call to pad_sequences afterwards.
            if len(seq) < seq_length:
                zeros = [0]*(seq_length - len(seq))
                zeros.extend(seq)
                seq = zeros
            seq = seq[-seq_length:]
            X.append(seq)
            y.append(song_encoded[i])
    print('Took {}'.format(datetime.datetime.now() - now))
    print()

    return X, y, seq_length, num_words, tokenizer


def create_model(seq_length, num_words, embedding_matrix, embedding_dim=config.EMBEDDING_DIM):
    # The + 1 accounts for the OOV token
    actual_num_words = num_words + 1

    inp = tf.keras.layers.Input(shape=(seq_length,))
    x = tf.keras.layers.Embedding(
        input_dim=actual_num_words, 
        output_dim=embedding_dim,
        input_length=seq_length,
        weights=[embedding_matrix],
        mask_zero=True)(inp)
    x = tf.keras.layers.GRU(128, return_sequences=True)(x)
    x = tf.keras.layers.GRU(128, dropout=0.2, recurrent_dropout=0.2)(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outp = tf.keras.layers.Dense(actual_num_words, activation='softmax')(x)

    model = tf.keras.models.Model(inputs=[inp], outputs=[outp])
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='rmsprop',
        metrics=['accuracy']
    )
    model.summary()
    return model


def train(epochs=100,
          export_dir=None,
          songdata_file=config.SONGDATA_FILE,
          artists=config.ARTISTS,
          embedding_file=config.EMBEDDING_FILE,
          embedding_dim=config.EMBEDDING_DIM,
          transform_words=False):
    if export_dir is None:
        export_dir = './export/{}'.format(datetime.datetime.now().isoformat(timespec='seconds'))
        os.makedirs(export_dir, exist_ok=True)

    embedding_mapping = embedding.create_embedding_mappings(embedding_file=embedding_file)
    songs = util.load_songdata(songdata_file=songdata_file, artists=artists)
    print('Will use {} songs from {} artists'.format(len(songs), len(artists)))
    
    X, y, seq_length, num_words, tokenizer = prepare_data(
        songs,
        transform_words=transform_words)

    # Make sure tokenizer is pickled, in case we need to
    util.pickle_tokenizer(tokenizer, export_dir)

    embedding_matrix = embedding.create_embedding_matrix(
        tokenizer,
        embedding_mapping,
        embedding_dim=embedding_dim,
        max_num_words=num_words)

    model = create_model(seq_length, num_words, embedding_matrix, embedding_dim=embedding_dim)

    # Run the training
    model.fit(np.array(X),
              np.array(y),
              batch_size=256,
              epochs=epochs,
              callbacks=[
                  tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10, verbose=1),
                  tf.keras.callbacks.ModelCheckpoint(
                      '{}/model.h5'.format(export_dir),
                      monitor='loss',
                      save_best_only=True,
                      verbose=1)
              ])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--embedding-file',
        default=config.EMBEDDING_FILE,
        help='Use a custom embedding file')
    parser.add_argument(
        '--transform-words',
        action='store_true',
        help="""
            To clean the song texts a little bit more than normal by e.g.
            transforming certain words like runnin' to running.
        """)
    args = parser.parse_args()
    train(embedding_file=args.embedding_file, transform_words=args.transform_words)