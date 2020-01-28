"""Train a song generating model."""
import argparse
import csv
import datetime
import os
import statistics

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub

from . import config, embedding, util


def prepare_data(
    songs, transform_words=False, use_full_sentences=False, use_strings=False
):
    """Prepare songs for training, including tokenizing and word preprocessing.

    Parameters
    ----------
    songs : list
        A list of song strings
    transform_words : bool
        Whether or not to transform certain words such as cannot -> can't
    use_full_sentences : bool
        Whether or not to only create full sentences, i.e. sentences where
        all the tokenized words are non-zero.
    use_strings : bool
        Whether or not to return sequences as normal strings or lists of integers

    Returns
    -------
    X : list
        Input sentences
    y : list
        Predicted words
    seq_length : int
        The length of each sequence
    num_words : int
        Number of words in the vocabulary
    tokenizer : object
        The Keras preproceessing tokenizer used for transforming sentences.

    """
    songs = util.prepare_songs(songs, transform_words=transform_words)
    tokenizer = util.prepare_tokenizer(songs)

    num_words = min(config.MAX_NUM_WORDS, len(tokenizer.word_index))

    print("Encoding all songs to integer sequences")
    if use_full_sentences:
        print("Note: Will only use full integer sequences!")
    now = datetime.datetime.now()
    songs_encoded = tokenizer.texts_to_sequences(songs)
    print("Took {}".format(datetime.datetime.now() - now))
    print()

    # Find the newline integer
    newline_int = tokenizer.word_index["\n"]

    # Calculate the average length of each sentence before a newline is seen.
    # This is probably between 5 and 10 words for most songs.
    # It will guide the verse structure.
    line_lengths = []
    print("Find the average line length for all songs")
    now = datetime.datetime.now()
    for song_encoded in songs_encoded:
        # Find the indices of the newline characters.
        # For double newlines (between verses), the distance will be 1 so these
        # distances are ignored...

        # Note: np.where() returns indices when used only with a condition.
        # Thus, these indices can be used to measure the distance between
        # newlines.
        newline_indexes = np.where(np.array(song_encoded) == newline_int)[0]

        lengths = [
            # Exclude the newline itself by subtracting 1 at the end...
            newline_indexes[i] - newline_indexes[i - 1] - 1
            for i in range(1, len(newline_indexes))
            if newline_indexes[i] - newline_indexes[i - 1] > 1
        ]
        line_lengths.extend(lengths)

        # There are no newlines at the beginning and end of the song, so add those line lengths
        line_lengths.append(
            newline_indexes[0]
        )  # The length of the first line is just the index of the newline...
        line_lengths.append(len(song_encoded) - newline_indexes[-1] - 1)

    print("Took {}".format(datetime.datetime.now() - now))
    print()

    median_seq_length = statistics.median(line_lengths)
    mean_seq_length = statistics.mean(line_lengths)
    print(
        "Median/mean line length from {} lines: {}/{}".format(
            len(line_lengths), median_seq_length, mean_seq_length
        )
    )
    print()

    # Prepare input data based on the median sequence length
    # Take 4 average lines (hence the multiplication by 4)
    # And assume a newline character between each (hence the + 3)
    seq_length = int(round(median_seq_length)) * 4 + 3

    # Prepare data for training
    X, y = [], []
    print("Creating test data")
    now = datetime.datetime.now()
    for song_encoded in songs_encoded:
        start_index = seq_length if use_full_sentences else 1
        for i in range(start_index, len(song_encoded)):
            seq = song_encoded[:i]
            # Manually pad/slice the sequences to the proper length
            # This avoids an expensive call to pad_sequences afterwards.
            if len(seq) < seq_length:
                zeros = [0] * (seq_length - len(seq))
                zeros.extend(seq)
                seq = zeros
            seq = seq[-seq_length:]
            X.append(seq)
            y.append(song_encoded[i])
    print("Took {}".format(datetime.datetime.now() - now))
    print()

    if use_strings:
        X = tokenizer.sequences_to_texts(X)

    return X, y, seq_length, num_words, tokenizer


def create_model(
    seq_length,
    num_words,
    embedding_matrix,
    embedding_dim=config.EMBEDDING_DIM,
    embedding_not_trainable=False,
):
    # The + 1 accounts for the OOV token
    actual_num_words = num_words + 1

    inp = tf.keras.layers.Input(shape=(seq_length,))
    x = tf.keras.layers.Embedding(
        input_dim=actual_num_words,
        output_dim=embedding_dim,
        input_length=seq_length,
        weights=[embedding_matrix],
        mask_zero=True,
        name="song_embedding",
    )(inp)
    x = tf.keras.layers.GRU(128, return_sequences=True)(x)
    x = tf.keras.layers.GRU(128, dropout=0.2, recurrent_dropout=0.2)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outp = tf.keras.layers.Dense(actual_num_words, activation="softmax")(x)

    model = tf.keras.models.Model(inputs=[inp], outputs=[outp])

    if embedding_not_trainable:
        model.get_layer("song_embedding").trainable = False

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="rmsprop",
        metrics=["accuracy"],
    )
    model.summary()
    return model


def create_transformer_model(
    num_words, transformer_network, trainable=True,
):
    inp = tf.keras.layers.Input(shape=[], dtype=tf.string)
    x = hub.KerasLayer(
        "https://tfhub.dev/google/universal-sentence-encoder/4",
        trainable=trainable,
        input_shape=[],
        dtype=tf.string,
    )(inp)
    x = tf.keras.layers.Dense(64, activation="relu")(x)

    # The + 1 accounts for the OOV token which can sometimes be present as the target word
    outp = tf.keras.layers.Dense(num_words + 1, activation="softmax")(x)

    model = tf.keras.models.Model(inputs=inp, outputs=outp)

    model.compile(
        loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"],
    )
    model.summary()
    return model


def train(
    epochs=100,
    export_dir=None,
    songdata_file=config.SONGDATA_FILE,
    artists=config.ARTISTS,
    embedding_file=config.EMBEDDING_FILE,
    embedding_dim=config.EMBEDDING_DIM,
    embedding_not_trainable=False,
    transform_words=False,
    use_full_sentences=False,
    transformer_network=None,
):
    if export_dir is None:
        export_dir = "./export/{}".format(
            datetime.datetime.now().isoformat(timespec="seconds")
        )
        os.makedirs(export_dir, exist_ok=True)

    songs = util.load_songdata(songdata_file=songdata_file, artists=artists)
    print(f"Will use {len(songs)} songs from {len(artists)} artists")

    X, y, seq_length, num_words, tokenizer = prepare_data(
        songs,
        transform_words=transform_words,
        use_full_sentences=use_full_sentences,
        use_strings=bool(transformer_network),
    )
    util.pickle_tokenizer(tokenizer, export_dir)

    model = None

    if transformer_network:
        print(f"Using transformer network '{transformer_network}'")
        model = create_transformer_model(
            num_words, transformer_network, trainable=not embedding_not_trainable
        )
    else:
        print(f"Using precreated embeddings from {embedding_file}")
        embedding_mapping = embedding.create_embedding_mappings(
            embedding_file=embedding_file
        )
        embedding_matrix = embedding.create_embedding_matrix(
            tokenizer,
            embedding_mapping,
            embedding_dim=embedding_dim,
            max_num_words=num_words,
        )
        model = create_model(
            seq_length,
            num_words,
            embedding_matrix,
            embedding_dim=embedding_dim,
            embedding_not_trainable=embedding_not_trainable,
        )

    # Run the training
    model.fit(
        np.array(X),
        np.array(y),
        batch_size=256,
        epochs=epochs,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor="loss", patience=10, verbose=1),
            tf.keras.callbacks.ModelCheckpoint(
                "{}/model.h5".format(export_dir),
                monitor="loss",
                save_best_only=True,
                verbose=1,
            ),
        ],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--embedding-file",
        default=config.EMBEDDING_FILE,
        help="Use a custom embedding file",
    )
    parser.add_argument(
        "--embedding-not-trainable",
        action="store_true",
        help="""
            Whether the embedding weights are trainable or locked to the
            vectors of the embedding file. It is only recommend to set this
            flag if the embedding file contains vectors for the full
            vocabulary of the songs.
        """,
    )
    parser.add_argument(
        "--transform-words",
        action="store_true",
        help="""
            To clean the song texts a little bit more than normal by e.g.
            transforming certain words like runnin' to running.
        """,
    )
    parser.add_argument(
        "--use-full-sentences",
        action="store_true",
        help="""
            Use only full sentences as training input to the model, i.e. no
            single-word vectors will be used for training. This decreases the
            training data, and avoids putting emphasis on single starting
            words in a song.
        """,
    )
    parser.add_argument(
        "--transformer-network",
        help="""
            Use a transformer architecture like the universal sentence encoder
            rather than a recurrent neural network.
        """,
        choices=["use"],
    )
    args = parser.parse_args()
    train(
        embedding_file=args.embedding_file,
        transform_words=args.transform_words,
        use_full_sentences=args.use_full_sentences,
        embedding_not_trainable=args.embedding_not_trainable,
        transformer_network=args.transformer_network,
    )
