"""Pytest configuration."""
import csv
import os
import tempfile
import shutil

import pytest


@pytest.fixture
def export_dir():
    export_dir = tempfile.mkdtemp(prefix="export_")
    yield export_dir
    shutil.rmtree(export_dir, ignore_errors=True)


@pytest.fixture
def songs_raw():
    return ["\nmeow\nmeow", "woof\n\nchorus\nwoof\n"]


@pytest.fixture
def songfile(songs_raw):
    _, songfile = tempfile.mkstemp(prefix="songs_", suffix=".csv")
    with open(songfile, "w", newline="") as f:
        writer = csv.writer(f, delimiter=",", quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["artist", "text"])
        writer.writerow(["cat", songs_raw[0]])
        writer.writerow(["dog", songs_raw[1]])
    yield songfile
    os.remove(songfile)


@pytest.fixture()
def songs():
    return ["meow \n meow", "woof \n  \n chorus \n woof woof"]


@pytest.fixture
def embedding_file():
    _, embedding_file = tempfile.mkstemp(prefix="embedding_", suffix=".txt")
    with open(embedding_file, "w") as f:
        f.write("woof 0.1 0.2 0.3")
    yield embedding_file
    os.remove(embedding_file)
