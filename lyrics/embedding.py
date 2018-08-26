"""Create a Word2Vec embedding from song data."""
import csv
import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
from gensim.models import Word2Vec

from . import config, util


def create_word2vec():
    songs = util.load_songdata()
    songs = songs.iloc[0:5]
    #tf.keras.preprocessing.
    #seq = text_to_word_sequence(text, self.filters, self.lower, self.split)
    model = Word2Vec(songs, workers=4, min_count=1, max_vocab_size=None)
    print(model.wv.vocab)
    #print(model.wv['crazy'])


def create_embedding_mappings(embedding_file=config.EMBEDDING_FILE):
    """Create a lookup dictionary for word embeddings from the given embeddings file."""
    print('Loading embedding mapping file')
    glove = pd.read_table(embedding_file, sep=' ', index_col=0, header=None, quoting=csv.QUOTE_NONE)
    mapping = {}

    print('Creating embedding mappings for faster lookup')
    for row in glove.itertuples():
        mapping[row[0]] = list(row[1:])

    return mapping


def create_embedding_matrix(tokenizer,
                            embedding_mapping,
                            embedding_dim=config.EMBEDDING_DIM,
                            max_num_words=config.MAX_NUM_WORDS):
    """Create an embedding matrix from the given keras tokenizer and embedding mapping dictionary.

    The embedding matrix can be used as weights for an embedding layer in Keras.

    The function ensures that that only the top N words get selected for the embedding.

    """
    # Create embedding matrix, add an extra row for the out of vocabulary vector
    embedding_matrix = np.zeros((max_num_words + 1, embedding_dim))

    print('Finding embedding vectors')
    now = datetime.datetime.now()
    for word, i in tokenizer.word_index.items():
        if i > max_num_words:
            continue
        if word in embedding_mapping:
            embedding_matrix[i] = embedding_mapping[word]
    print('Took {}'.format(datetime.datetime.now() - now))
    return embedding_matrix


if __name__ == '__main__':
    create_word2vec()