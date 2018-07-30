import numpy as np
import tensorflow as tf
import os
import csv

import hparams_setup
import model_builder
from utils import io
from utils import metrics
from utils import preprocess
from gui import GUI

np.set_printoptions(linewidth=10000, threshold=1000000000)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

hparams = hparams_setup.get_hparams()

preprocess.clear_previous_run(hparams)
#preprocess.prep_nmt_dataset(hparams)


# INITIALIZE MODELS & SESSIONS

train_model = model_builder.create_train_model(hparams)
test_model = model_builder.create_test_model(hparams)
test2_model = model_builder.create_test2_model(hparams)
validate_model = model_builder.create_validate_model(hparams)
pred_model = model_builder.create_pred_model(hparams)

train_sess = tf.Session(graph=train_model.graph)
test_sess = tf.Session(graph=test_model.graph)
test2_sess = tf.Session(graph=test2_model.graph)
validate_sess = tf.Session(graph=validate_model.graph)
pred_sess = tf.Session(graph=pred_model.graph)

# DEFINE MODEL STEPS & LOGGING

def train_log(global_step):
    print('TRAIN STEP >>> @ Train Step {}: Completed with loss {}'.format(global_step, loss))

    summary_writer = tf.summary.FileWriter(os.path.join(hparams.model_dir, 'train'), train_model.graph)
    summary_writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='loss', simple_value=loss)]), global_step)
    summary_writer.close()

def test_step_log():
    with test_model.graph.as_default():
        loaded_test_model, global_step = model_builder.create_or_load_model(hparams, test_model.model, test_sess)

    test_sess.run(test_model.iterator.initializer)
    loss, src, tgts, ids = loaded_test_model.eval(test_sess)

    io.print_example(ids, src, tgts, hparams.test_max_printouts)
    print('TEST STEP >>> @ Train Step {}: Completed with loss {}'.format(global_step, loss))
    q8 = np.round(metrics.q8_infer_accuracy(preds=ids, targets=tgts), 4) * 100
    q3 = np.round(metrics.q3_infer_accuracy(preds=ids, targets=tgts), 4) * 100
    print('Q8: {}'.format(q8))
    print('Q3: {}'.format(q3))

    summary_writer = tf.summary.FileWriter(os.path.join(hparams.model_dir, 'test'), test_model.graph)
    summary_writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='loss', simple_value=loss)]), global_step)
    summary_writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='q8', simple_value=q8)]), global_step)
    summary_writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='q3', simple_value=q3)]), global_step)
    summary_writer.close()

test_frag_num = 0
test_frags = list()
with open(hparams.data_dir + "test/test_frag.csv", 'r+') as test_frag_file:
    frag_reader = csv.reader(test_frag_file)
    for frag_len in frag_reader:
        test_frags.append(int(frag_len[0]))


def test2_step_log(test_frag_num):
    with test2_model.graph.as_default():
        loaded_test2_model, global_step = model_builder.create_or_load_model(hparams, test2_model.model, test2_sess)

    test2_sess.run(test2_model.iterator.initializer)
    src, tgts, ids = loaded_test2_model.infer(test2_sess)
    if hparams.beam_search:
        ids = ids.transpose([2, 0, 1])   # Change from [batch_size, time_steps, beam_width] to [beam_width, batch_size, time_steps]
        ids = ids[0]  # Only use top 1 prediction from top K

    print(src)
    print(tgts)
    print(ids)

    new_src, new_tgts, new_ids, test_frag_num = metrics.stitch(src, tgts, ids, hparams.fragment_radius, test_frags, test_frag_num)

    print(new_src)
    print(new_tgts)
    print(new_ids)

    print('TEST2 STEP >>> @ Train Step {}'.format(global_step))
    io.print_example(new_ids, new_src, new_tgts, hparams.test2_max_printouts)
    fragq8 = np.round(metrics.q8_infer_accuracy(preds=ids, targets=tgts), 4) * 100
    fragq3 = np.round(metrics.q3_infer_accuracy(preds=ids, targets=tgts), 4) * 100
    q8 = np.round(metrics.q8_infer_accuracy(preds=new_ids, targets=new_tgts), 4) * 100
    q3 = np.round(metrics.q3_infer_accuracy(preds=new_ids, targets=new_tgts), 4) * 100
    print('FragQ8: {}'.format(fragq8))
    print('FragQ3: {}'.format(fragq3))
    print('Q8: {}'.format(q8))
    print('Q3: {}'.format(q3))

    summary_writer = tf.summary.FileWriter(os.path.join(hparams.model_dir, 'test2'), test2_model.graph)
    summary_writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='fragq8', simple_value=q8)]), global_step)
    summary_writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='fragq3', simple_value=q3)]), global_step)
    summary_writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='q8', simple_value=q8)]), global_step)
    summary_writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='q3', simple_value=q3)]), global_step)
    summary_writer.close()

    return test_frag_num


# LOAD & START TRAINING

with train_model.graph.as_default():
    loaded_train_model, global_step = model_builder.create_or_load_model(hparams, train_model.model, train_sess)
print('MODE >>> Training ({} out of {} steps)'.format(global_step, hparams.num_train_steps))
epoch = 0
train_sess.run(train_model.iterator.initializer)

while global_step < hparams.num_train_steps:
    try:
        _, loss, global_step = loaded_train_model.train(train_sess)

        if global_step % hparams.train_log_freq == 0:
            train_log(global_step)

        if global_step % hparams.test_log_freq == 0:
            loaded_train_model.saver.save(train_sess, hparams.model_dir, global_step)
            test_step_log()

        if global_step % hparams.test2_log_freq == 0:
            loaded_train_model.saver.save(train_sess, hparams.model_dir, global_step)
            test_frag_num = test2_step_log(test_frag_num)

    except tf.errors.OutOfRangeError:
        print('Epoch {} completed.'.format(epoch))
        train_sess.run(train_model.iterator.initializer)
        epoch += 1

# PREDICTION LOGIC

print()
print('MODE >>> Prediction')

with pred_sess as sess:
    with pred_model.graph.as_default():
        loaded_pred_model, _ = model_builder.create_or_load_model(hparams, pred_model.model, sess)

    def callback(text):
        src = ''
        for ch in text:
            src = src + ch + ','
        sess.run(pred_model.iterator.initializer, feed_dict={pred_model.src_placeholder: [src]})
        ids = loaded_pred_model.pred(pred_sess)
        if hparams.beam_search:
            ids = ids.transpose([2, 0, 1])   # Change from [batch_size, time_steps, beam_width] to [beam_width, batch_size, time_steps]
            ids = ids[0]  # Only use top 1 prediction from top K
        ids = ids[0]
        ids = ''.join(ids)
        gui.set_out_text(ids)

    gui = GUI(50)
    gui.set_callback(callback)
    gui.run()

# VALIDATE (DO NOT USE)

validate_frag_num = 0
validate_frags = list()
with open(hparams.data_dir + "validate/validate_frag.csv", 'r+') as validate_frag_file:
    frag_reader = csv.reader(validate_frag_file)
    for frag_len in frag_reader:
        validate_frags.append(int(frag_len[0]))


def validate():
    with validate_model.graph.as_default():
        loaded_validate_model, global_step = model_builder.create_or_load_model(hparams, validate_model.model, validate_sess)

    validate_sess.run(validate_model.iterator.initializer)
    src, tgts, ids = loaded_validate_model.infer(validate_sess)
    if hparams.beam_search:
        ids = ids.transpose([2, 0, 1])   # Change from [batch_size, time_steps, beam_width] to [beam_width, batch_size, time_steps]
        ids = ids[0]  # Only use top 1 prediction from top K

    new_src, new_tgts, new_ids, validate_frag_num = metrics.do_stitching(src, tgts, ids, hparams.fragment_radius, validate_frags, validate_frag_num)

    # do something or other

    return validate_frag_num