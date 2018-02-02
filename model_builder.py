import tensorflow as tf
import collections

import data_pipeline
from model import NMTModel



def create_or_load_model(hparams, model, sess):
    latest_ckpt = tf.train.latest_checkpoint(hparams.model_dir)
    if latest_ckpt:
        model.saver.restore(sess, latest_ckpt)
    else:
        sess.run(tf.global_variables_initializer())

    return model


class TrainModel(collections.namedtuple("TrainModel", ("graph", "model", "iterator"))):
    pass


def create_train_model(hparams):
    graph = tf.Graph()

    with graph.as_default():
        iterator = data_pipeline.get_batched_iterator(
            hparams,
            src_loc=hparams.data_dir + 'primary.csv',
            tgt_loc=hparams.data_dir + 'secondary.csv',
            weights_loc=hparams.data_dir + 'weights.csv')

        model = NMTModel(
            hparams,
            iterator=iterator,
            mode='TRAIN')

    return TrainModel(
        graph=graph,
        model=model,
        iterator=iterator)


class EvalModel(collections.namedtuple("EvalModel", ("graph", "model", "iterator"))):
    pass


def create_eval_model(hparams):
    graph = tf.Graph()

    with graph.as_default():
        iterator = data_pipeline.get_batched_iterator(
            hparams,
            src_loc=hparams.data_dir + 'primary.csv',
            tgt_loc=hparams.data_dir + 'secondary.csv',
            weights_loc=hparams.data_dir + 'weights.csv')

        model = NMTModel(
            hparams,
            iterator=iterator,
            mode='EVAL')

    return EvalModel(
        graph=graph,
        model=model,
        iterator=iterator)


class InferModel(collections.namedtuple("InferModel", ("graph", "model", "iterator"))):
    pass


def create_infer_model(hparams):
    graph = tf.Graph()

    with graph.as_default():
        iterator = data_pipeline.get_batched_iterator(
            hparams,
            src_loc=hparams.data_dir + 'primary.csv',
            tgt_loc=hparams.data_dir + 'secondary.csv',
            weights_loc=hparams.data_dir + 'weights.csv')

        model = NMTModel(
            hparams,
            iterator=iterator,
            mode='INFER')

    return InferModel(
        graph=graph,
        model=model,
        iterator=iterator)
