"""Configuration and parameters."""

BATCH_SIZE = 256
MAX_EPOCHS = 100

MAX_NUM_WORDS = 20000
SONGDATA_FILE = "./data/songdata.csv"
NUM_LINES_TO_INCLUDE = 4
MAX_REPEATS = 2
SAVE_FREQUENCY = 10
EARLY_STOPPING_PATIENCE = 5

# The default embedding dimension matches the glove filename
EMBEDDING_DIM = 50
EMBEDDING_FILE = "./data/glove.6B.50d.txt"

# Sample rock artists (this was based on a random top 20 I found online)
# Artists are confirmed to exist in the dataset
ARTISTS = [
    "The Beatles",
    "Rolling Stones",
    "Pink Floyd",
    "Queen",
    "Who",  # The Who
    "Jimi Hendrix",
    "Doors",  # The Doors
    "Nirvana",
    "Eagles",
    "Aerosmith",
    "Creedence Clearwater Revival",
    "Guns N' Roses",
    "Black Sabbath",
    "U2",
    "David Bowie",
    "Beach Boys",
    "Van Halen",
    "Bob Dylan",
    "Eric Clapton",
    "Red Hot Chili Peppers",
]
