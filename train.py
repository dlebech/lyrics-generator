"""Train a song generating model."""
import csv
import datetime
import os
import pickle
import statistics

import numpy as np
import pandas as pd
import tensorflow as tf


EMBEDDING_DIM = 50
MAX_NUM_WORDS = 20000
SONGDATA_FILE = './data/songdata.csv'
EMBEDDING_FILE = './data/glove.6B.{}d.txt'.format(EMBEDDING_DIM)

# Sample rock artists (this was based on a random top 20 I found online)
# Artists are confirmed to exist in the dataset
artists = [
    'The Beatles',
    'Rolling Stones',
    'Pink Floyd',
    'Queen',
    'Who', # The Who
    'Jimi Hendrix',
    'Doors', # The Doors
    'Nirvana',
    'Eagles',
    'Aerosmith',
    'Creedence Clearwater Revival',
    "Guns N' Roses",
    'Black Sabbath',
    'U2',
    'David Bowie',
    'Beach Boys',
    'Van Halen',
    'Bob Dylan',
    'Eric Clapton',
    'Red Hot Chili Peppers',
]


def pickle_tokenizer(tokenizer, export_dir):
    with open('{}/tokenizer.pickle'.format(export_dir), 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_data(songdata_file=SONGDATA_FILE, embedding_file=EMBEDDING_FILE):
    print('Loading song data from {}'.format(songdata_file))
    songdata = pd.read_csv(songdata_file)

    print('Loading glove embeddings')
    glove = pd.read_table(EMBEDDING_FILE, sep=' ', index_col=0, header=None, quoting=csv.QUOTE_NONE)
    glove_mapping = {}

    print('Creating glove mappings for faster lookup')
    for row in glove.itertuples():
        glove_mapping[row[0]] = list(row[1:])

    # Find all songs from the selected artists
    songs = songdata[songdata.artist.isin(artists)].text

    print('Will use {} songs from {} artists'.format(len(songs), len(artists)))
    return songs, glove_mapping


def create_embedding_matrix(tokenizer, glove_mapping):
    # Create embedding matrix
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))

    print('Finding embedding vectors')
    now = datetime.datetime.now()
    for word, i in tokenizer.word_index.items():
        if i >= MAX_NUM_WORDS:
            continue
        if word in glove_mapping:
            embedding_matrix[i] = glove_mapping[word]
    print('Took {}'.format(datetime.datetime.now() - now))
    print()
    return embedding_matrix


def prepare_data(songs, glove_mapping):
    # Put whitespace around each newline character so something like \nhello is not treated as a word
    print('Preparing proper newlines')
    now = datetime.datetime.now()
    songs = [song.strip('\n').replace('\n', ' \n ') for song in songs]
    print('Took {}'.format(datetime.datetime.now() - now))
    print()

    # Create tokenizer and remove newline character from the filters so it's treated as a word
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=MAX_NUM_WORDS)
    tokenizer.filters = tokenizer.filters.replace('\n', '')

    # Fit on the texts and convert the data to integer sequences
    print('Fitting tokenizer to texts')
    now = datetime.datetime.now()
    tokenizer.fit_on_texts(songs)
    print('Took {}'.format(datetime.datetime.now() - now))
    print()

    # The number of word (features) is the length of the vocabulary + 1
    # to account for the missing 0 in the tokenizer
    num_words = min(MAX_NUM_WORDS, len(tokenizer.word_index) + 1)

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


def create_model(seq_length, num_words, embedding_matrix):
    inp = tf.keras.layers.Input(shape=(seq_length,))
    x = tf.keras.layers.Embedding(
        input_dim=num_words,
        output_dim=EMBEDDING_DIM,
        input_length=seq_length,
        weights=[embedding_matrix],
        mask_zero=True)(inp)
    x = tf.keras.layers.GRU(128, return_sequences=True)(x)
    x = tf.keras.layers.GRU(128, dropout=0.2, recurrent_dropout=0.2)(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outp = tf.keras.layers.Dense(num_words, activation='softmax')(x)

    model = tf.keras.models.Model(inputs=[inp], outputs=[outp])
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='rmsprop',
        metrics=['accuracy']
    )
    model.summary()
    return model


if __name__ == '__main__':
    export_dir = './export/{}'.format(datetime.datetime.now().isoformat(timespec='seconds'))
    os.makedirs(export_dir, exist_ok=True)

    songs, glove_mapping = load_data()
    X, y, seq_length, num_words, tokenizer = prepare_data(songs, glove_mapping)

    # Make sure tokenizer is pickled, in case we need to
    pickle_tokenizer(tokenizer, export_dir)

    embedding_matrix = create_embedding_matrix(tokenizer, glove_mapping)

    model = create_model(seq_length, num_words, embedding_matrix)

    # Run the training
    model.fit(np.array(X),
              np.array(y),
              batch_size=256,
              epochs=100,
              callbacks=[
                  tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10, verbose=1),
                  tf.keras.callbacks.ModelCheckpoint(
                      '{}/model.h5'.format(export_dir),
                      monitor='loss',
                      save_best_only=True,
                      verbose=1)
              ])
