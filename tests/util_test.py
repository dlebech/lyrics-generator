"""Test the utility module."""
import tensorflow as tf

from lyrics import util


def test_pickle_load_tokenizer(export_dir, songs):
    """It should pickle and unpickle a tokenizer."""
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(songs)
    util.pickle_tokenizer(tokenizer, export_dir)
    tokenizer = util.load_tokenizer("{}/tokenizer.pickle".format(export_dir))
    assert "woof" in tokenizer.word_index


def test_load_songdata(songfile):
    """It should return an array of songs."""
    songs = util.load_songdata(songdata_file=songfile, artists=["cat", "dog"])
    assert len(songs) == 2
    assert songs[0] == "\nmeow\nmeow"
    assert songs[1] == "woof\n\nchorus\nwoof\n"


def test_load_songdata_limit_artists(songfile):
    """It should only return the requested artists."""
    songs = util.load_songdata(songdata_file=songfile, artists=["dog"])
    assert len(songs) == 1
    assert songs[0] == "woof\n\nchorus\nwoof\n"


def test_load_songdata_all_artists(songfile):
    """It should only return the requested artists."""
    songs = util.load_songdata(songdata_file=songfile, artists=[])
    assert len(songs) == 2


def test_prepare_songs(songs_raw):
    """It should strip newlines at beginning and end but preserve newlines in the middle."""
    songs = util.prepare_songs(songs_raw)
    assert songs[0] == "meow \n meow"
    assert songs[1] == "woof \n  \n chorus \n woof"


def test_prepare_songs_max_repeats(songs_raw):
    """It should strip newlines at beginning and end but preserve newlines in the middle."""
    songs = util.prepare_songs(["repeat\nrepeat\nrepeat"])
    assert songs[0] == "repeat \n repeat"

    songs = util.prepare_songs(
        ["once\n twice\nrepeat \n repeat\nrepeat \n and again\n and again\n and again"]
    )
    assert songs[0] == "once \n twice \n repeat \n repeat \n and again \n and again"


def test_prepare_songs_transform_words():
    """It should replace in' with ing."""
    raw_songs = [
        "I am runnin' and singin'",
        "She is runnin', all over the place",
        "runnin'\nall over the place",
        "She is runnin' and he is singin'",
        "I cannot listen to what they are singin'",
        "The island we are on is nice",
        "I ain't havin' this ain't",
        "Ain't!",
        "She isn't goin'",
        "He isn't talkin'",
    ]
    songs = util.prepare_songs(raw_songs)
    assert songs[0] == "i am runnin' and singin'"
    assert songs[1] == "she is runnin', all over the place"
    assert songs[2] == "runnin' \n all over the place"

    songs = util.prepare_songs(raw_songs, transform_words=True)
    assert songs[0] == "i am running and singing"
    assert songs[1] == "she's running, all over the place"
    assert songs[2] == "running \n all over the place"
    assert songs[3] == "she's running and he's singing"
    assert songs[4] == "i can't listen to what they're singing"
    assert songs[5] == "the island we're on is nice"
    assert songs[6] == "i ain't having this ain't"
    assert songs[7] == "ain't!"
    assert songs[8] == "she isn't going"
    assert songs[9] == "he isn't talking"


def test_prepare_songs_profanity_censor():
    """It should remove profanity."""
    songs = util.prepare_songs(["ok shit go *"], profanity_censor=True)
    assert songs[0] == "ok **** go"


def test_prepare_tokenizer(songs):
    """It should tokenize newlines and include all words."""
    tokenizer = util.prepare_tokenizer(songs)
    assert len(tokenizer.word_index) == 4
    assert tokenizer.word_index == {"\n": 1, "woof": 2, "meow": 3, "chorus": 4}

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


def test_prepare_tokenizer_char_level(songs):
    """It should tokenize at character level."""
    tokenizer = util.prepare_tokenizer(songs, char_level=True)
    # 12 characters = ['\n', ' ', 'c', 'e', 'f', 'h', 'm', 'o', 'r', 's', 'u', 'w']
    assert len(tokenizer.word_index) == 12


def test_prepare_tokenizer_profanity():
    """It should accept the profanity censorship."""
    songs = ["ok ok ok **** go go"]
    tokenizer = util.prepare_tokenizer(["ok ok ok **** go go"])
    assert len(tokenizer.word_index) == 3
    assert tokenizer.word_index == {"ok": 1, "go": 2, "****": 3}
    sentences = tokenizer.texts_to_sequences(songs)
    assert sentences[0] == [1, 1, 1, 3, 2, 2]
