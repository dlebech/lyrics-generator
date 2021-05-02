# Lyrics Generator

![build status](https://github.com/dlebech/lyrics-generator/actions/workflows/python-app.yml/badge.svg)
[![codecov](https://codecov.io/gh/dlebech/lyrics-generator/branch/master/graph/badge.svg)](https://codecov.io/gh/dlebech/lyrics-generator)

This is a small experiment in generating lyrics with a recurrent neural network, trained with Keras and Tensorflow 2.

It works in the browser with Tensorflow.js! Try it [here](https://davidlebech.com/lyrics/).

The model can be trained at both word- and character level which each has their own pros and cons.

## Train the model

### Install dependencies

```shell
pip install -r requirements.txt
```

The requirement file has been reduced in size so if any of the scripts fail,
just install the missing packages :-)

### Get the data

- Create a song dataset. See ["Create your own song dataset"](#create-your-own-song-dataset) below.
  - Save the dataset as `songdata.csv` file in a `data` sub-directory.
  - Alternatively, you can name it anything you like and use the `--songdata-file` parameter when training.
- Download the [Glove embeddings](http://nlp.stanford.edu/data/glove.6B.zip)
  - Save the `glove.6B.50d.txt` file in a `data` sub-directory.
  - Alternatively, you can create your a word2vec embedding (see below)

### Create your own song dataset

The code expects an input dataset to be stored at `date/songdata.csv` by default (this can be changed in `config.py` or via CLI parameter `--songdata-file`).

The file should be in CSV format with the following columns (case sensitive):
- `artist`
  - A string, e.g. "The Beatles"
- `text`
  - A string with the entire lyrics for one song, including newlines.

A sample dataset with a simple text is provided in `sample.csv`. To test things are working, you can train using that file:

```shell
python -m lyrics.train --songdata-file sample.csv --early-stopping-patience 50 --artists '*'
```
  
### (Optional) Create a word2vec embedding matrix

If you have the `songdata.csv` file from above, you can simply create the
word2vec vectors like this:

```shell
python -m lyrics.embedding --name-suffix myembedding
```

This will create `word2vec_myembedding.model` and `word2vec_myembedding.txt`
files in the default data directory `data/`. Use `-h` to see other options
like artists and custom songdata file.

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

The output model and tokenizer is stored in a timestamped folder like `export/2020-01-01T010203` by default.

**Note**: During experimentation, I found that raising the batch size to something like 2048 speeds up processing, but it depends on your hardware resources whether this is feasible of course.

#### Training on GPU

The requirements.txt file refers to the CPU version of Tensorflow but manually
uninstalling and installing the GPU version should work fine.

However, it might be a bit easier with Docker and [nvidia-docker](https://github.com/NVIDIA/nvidia-docker),
so follow the instructions there to install it first, and then:

```shell
docker build -t lyrics-gpu .
docker run --rm -it --gpus all -v $PWD:/tf/src -u $(id -u):$(id -g) lyrics-gpu bash
```

Then run the normal commands from there, e.g. `python -m lyrics.train`.

Tip: You might want to use the parameter `--gpu-speedup`

Tip: If you get a cryptic Tensorflow error like `errors_impl.CancelledError:  [_Derived_]RecvAsync is cancelled.` while training on GPU, try pre-prending the train command with `TF_FORCE_GPU_ALLOW_GROWTH=true`, e.g.:
```shell
TF_FORCE_GPU_ALLOW_GROWTH=true python -m lyrics.train --transform-words --num-lines-to-include=10 --artists '*' --gpu-speedup
```

#### Use transformer network

To use the universal sentence encoder architecture:

```shell
python -m lyrics.train --embedding-not-trainable --transformer-network use
```

**Note** This model is not going to work in Tensorflow JS currently, so it
should only be used from the command-line.

#### Character-level predictions

In the default training mode, the model predicts the next word, given a sequence of words. Changing the model to predict the next character can be done using the `--char-level` flag.

```shell
python -m lyrics.train --char-level
```

## Create new lyrics

```shell
python -m cli lyrics model.h5 tokenizer.pickle
```

Try `python -m cli lyrics -h` to find out more

## Export to Tensorflow JS (used for the app)

**Note**: Make sure to use the `--tfjs-compatible` flag during training!

```shell
python -m cli export model.h5 tokenizer.pickle
```

This creates a sub-directory `export/js` with the relevant files (can be used
for the app).

## Single-page "app" for creating lyrics

**Note**: Make sure to use the `--tfjs-compatible` flag during training!

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
