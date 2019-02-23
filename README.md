# Lyrics Generator

[![Build Status](https://travis-ci.com/dlebech/lyrics-generator.svg?branch=master)](https://travis-ci.com/dlebech/lyrics-generator)
[![codecov](https://codecov.io/gh/dlebech/lyrics-generator/branch/master/graph/badge.svg)](https://codecov.io/gh/dlebech/lyrics-generator)

This is a small experiment in generating lyrics with a recurrent neural network, trained with Keras and Tensorflow.

It works in the browser with Tensorflow.js! Try it [here](https://davidlebech.com/lyrics/).

## Train the model

### Install dependencies

```shell
pip install -r requirements.txt
```

The requirement file has been reduced in size so if any of the scripts fail,
just install the missing packages :-)

### Get the data

- Download the [songdata dataset](https://www.kaggle.com/mousehead/songlyrics).
  - Save the `songdata.csv` file in a `data` sub-directory.
- Download the [Glove embeddings](http://nlp.stanford.edu/data/glove.6B.zip)
  - Save the `glove.6B.50d.txt` file in a `data` sub-directory.
  - Alternatively, you can create your a word2vec embedding (see below)
  
### (Optional) Create a word2vec embedding matrix

If you have the `songdata.csv` file from above, you can simply create the
word2vec vectors like this:

```shell
python -m lyrics.embedding
```

Perhaps there will be a proper CLI command for this in the future, perhaps not :-)

### Run the training

```shell
python -m lyrics.train -h
```

This command by default takes care of all the training. Warning: it takes a
very long time on a normal CPU!

Check `-h` for options. For example, if you want to use a different embedding
than the glove embedding:

```shell
python -m lyrics.train --embedding-file ./embeddings.txt
```

The embeddings are still assumed to be 50 dimensional.


## Create new lyrics

```shell
python cli.py lyrics model.h5 tokenizer.pickle
```

Try `python cli.py lyrics -h` to find out more

## Export to Tensorflow JS (used for the app)

```shell
python cli.py export model.h5 tokenizer.pickle
```

This creates a sub-directory `export` with the relevant files (can be used for the app)

## Single-page "app" for creating lyrics

The `lyrics-tfjs` sub-directory has a simple web-page that can be used to
create lyrics in the browser. The code expects data to be found in a `data/`
sub-directory. This includes the `words.json` file, `model.json` and any extra
files generated by the Tensorflow export.

[Demo](https://davidlebech.com/lyrics/).

## Development

Make sure to get all dependencies:

```shell
pip install -r requirements_dev.txt
```

### Testing

```shell
python -m pytest --cov=lyrics tests/
```
