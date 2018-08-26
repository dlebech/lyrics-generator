"""Configuration and parameters."""

MAX_NUM_WORDS = 20000
SONGDATA_FILE = './data/songdata.csv'

# The default embedding dimension matches the glove filename
EMBEDDING_DIM = 50
EMBEDDING_FILE = './data/glove.6B.50d.txt'

# Sample rock artists (this was based on a random top 20 I found online)
# Artists are confirmed to exist in the dataset
ARTISTS = [
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