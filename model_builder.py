import tensorflow as tf
import collections

from tensorflow.python.ops import lookup_ops

import data_pipeline
import models


def create_or_load_model(hparams, model, sess):
    latest_ckpt = tf.train.latest_checkpoint(hparams.model_dir)
    if latest_ckpt:
        model.saver.restore(sess, latest_ckpt)
    else:
        sess.run(tf.global_variables_initializer())

    sess.run(tf.tables_initializer())
    global_step = model.global_step.eval(session=sess)
    return model, global_step


class TrainModel(collections.namedtuple("TrainModel", ("graph", "model", "iterator"))):
    pass


def create_train_model(hparams):
    graph = tf.Graph()

    src_vocab_loc = hparams.data_dir + 'primary_vocab.txt'
    tgt_vocab_loc = hparams.data_dir + 'secondary_vocab.txt'
    src_loc = hparams.data_dir + 'primary_train.csv'
    tgt_loc = hparams.data_dir + 'secondary_train.csv'
    weights_loc = hparams.data_dir + 'weights_train.csv'

    with graph.as_default():
        src_vocab_table, tgt_vocab_table = data_pipeline.make_vocab_tables(src_vocab_loc, tgt_vocab_loc)

        src_data = tf.data.TextLineDataset(src_loc)
        tgt_data = tf.data.TextLineDataset(tgt_loc)
        weight_data = tf.data.TextLineDataset(weights_loc)

        iterator = data_pipeline.get_iterator(hparams,
                                              src_data, tgt_data, weight_data,
                                              src_vocab_table, tgt_vocab_table)

        model = models.NMTModel(
            hparams,
            iterator=iterator,
            mode='TRAIN',
            src_vocab_table=src_vocab_table,
            tgt_vocab_table=tgt_vocab_table)

    return TrainModel(
        graph=graph,
        model=model,
        iterator=iterator)


class EvalModel(collections.namedtuple("EvalModel", ("graph", "model", "iterator"))):
    pass


def create_eval_model(hparams):
    graph = tf.Graph()

    src_vocab_loc = hparams.data_dir + 'primary_vocab.txt'
    tgt_vocab_loc = hparams.data_dir + 'secondary_vocab.txt'
    src_loc = hparams.data_dir + 'primary_test.csv'
    tgt_loc = hparams.data_dir + 'secondary_test.csv'
    weights_loc = hparams.data_dir + 'weights_test.csv'

    with graph.as_default():
        src_vocab_table, tgt_vocab_table = data_pipeline.make_vocab_tables(src_vocab_loc, tgt_vocab_loc)

        src_data = tf.data.TextLineDataset(src_loc)
        tgt_data = tf.data.TextLineDataset(tgt_loc)
        weight_data = tf.data.TextLineDataset(weights_loc)

        iterator = data_pipeline.get_iterator(hparams,
                                              src_data, tgt_data, weight_data,
                                              src_vocab_table, tgt_vocab_table)

        model = models.NMTModel(
            hparams,
            iterator=iterator,
            mode='EVAL',
            src_vocab_table=src_vocab_table,
            tgt_vocab_table=tgt_vocab_table)

    return EvalModel(
        graph=graph,
        model=model,
        iterator=iterator)


class InferModel(collections.namedtuple("InferModel", ("graph", "model", "iterator"))):
    pass


def create_infer_model(hparams):
    graph = tf.Graph()

    src_vocab_loc = hparams.data_dir + 'primary_vocab.txt'
    tgt_vocab_loc = hparams.data_dir + 'secondary_vocab.txt'
    src_loc = hparams.data_dir + 'primary_test.csv'
    tgt_loc = hparams.data_dir + 'secondary_test.csv'
    weights_loc = hparams.data_dir + 'weights_test.csv'

    with graph.as_default():
        src_vocab_table, tgt_vocab_table = data_pipeline.make_vocab_tables(src_vocab_loc, tgt_vocab_loc)

        src_data = tf.data.TextLineDataset(src_loc)
        tgt_data = tf.data.TextLineDataset(tgt_loc)
        weight_data = tf.data.TextLineDataset(weights_loc)

        iterator = data_pipeline.get_iterator(hparams,
                                              src_data, tgt_data, weight_data,
                                              src_vocab_table, tgt_vocab_table)

        model = models.NMTModel(
            hparams,
            iterator=iterator,
            mode='INFER',
            src_vocab_table=src_vocab_table,
            tgt_vocab_table=tgt_vocab_table)

    return InferModel(
        graph=graph,
        model=model,
        iterator=iterator)


class PredModel(collections.namedtuple("PredModel", ("graph", "model", "iterator", "src_placeholder"))):
    pass


def create_pred_model(hparams):
    graph = tf.Graph()

    src_vocab_loc = hparams.data_dir + 'primary_vocab.txt'
    tgt_vocab_loc = hparams.data_dir + 'secondary_vocab.txt'

    with graph.as_default():
        src_vocab_table, tgt_vocab_table = data_pipeline.make_vocab_tables(src_vocab_loc, tgt_vocab_loc)
        reverse_tgt_vocab_table = lookup_ops.index_to_string_table_from_file(tgt_vocab_loc)

        src_placeholder = tf.placeholder(shape=[None], dtype=tf.string)
        src_data = tf.data.Dataset.from_tensor_slices(src_placeholder)

        iterator = data_pipeline.get_infer_iterator(
                hparams,
                src_data=src_data,
                src_vocab_table=src_vocab_table)

        model = models.NMTModel(
                hparams,
                iterator=iterator,
                mode='PRED',
                src_vocab_table=src_vocab_table,
                tgt_vocab_table=tgt_vocab_table,
                reverse_tgt_vocab_table=reverse_tgt_vocab_table)

    return PredModel(
        graph=graph,
        model=model,
        iterator=iterator,
        src_placeholder=src_placeholder)
