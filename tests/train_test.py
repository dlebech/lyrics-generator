"""End to end train testing."""
import os

from lyrics import train


def test_train(export_dir, embedding_file, songfile):
    """It should train and save a model and tokenizer."""
    train.train(
        epochs=1,
        export_dir=export_dir,
        songdata_file=songfile,
        artists=['cat', 'dog'],
        embedding_file=embedding_file,
        embedding_dim=3)
    assert os.path.exists('{}/model.h5'.format(export_dir))
    assert os.path.exists('{}/tokenizer.pickle'.format(export_dir))