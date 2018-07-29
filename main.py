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


# INITIALIZE MODELS

train_model = model_builder.create_train_model(hparams)
eval_model = model_builder.create_eval_model(hparams)
eval2_model = model_builder.create_eval2_model(hparams)
infer_model = model_builder.create_infer_model(hparams)
pred_model = model_builder.create_pred_model(hparams)

# INITALIZE SESSIONS

train_sess = tf.Session(graph=train_model.graph)
eval_sess = tf.Session(graph=eval_model.graph)
eval2_sess = tf.Session(graph=eval2_model.graph)
infer_sess = tf.Session(graph=infer_model.graph)
pred_sess = tf.Session(graph=pred_model.graph)

# STITCHING LOGIC

def stitch(frags):
    candidates = list()
    for _ in frags:
        candidates.append(list())
    for i, frag in enumerate(frags):
        if i < hparams.fragment_radius:
            for j in range(i + hparams.fragment_radius):
                candidates[j].append(frag[j])
        elif i >= len(frags) - hparams.fragment_radius:
            for j in range(hparams.fragment_radius + len(frags) - i):
                candidates[i + j - hparams.fragment_radius].append(frag[j])
        else:
            for j in range(hparams.fragment_radius * 2):
                candidates[i + j - hparams.fragment_radius].append(frag[j])

    stitched = list()
    for candidate in candidates:
        print("candidate: {}".format(candidate))
        stitched.append(np.argmax(np.bincount(candidate)))
    return stitched

# TRAIN LOGIC

def train_log(global_step):
    print('TRAIN STEP >>> @ Train Step {}: Completed with loss {}'.format(global_step, loss))

    summary_writer = tf.summary.FileWriter(os.path.join(hparams.model_dir, 'train'), train_model.graph)
    loss_summary = tf.Summary(value=[tf.Summary.Value(tag='loss', simple_value=loss)])
    summary_writer.add_summary(loss_summary, global_step)
    summary_writer.close()

# EVAL (TEST 1) LOGIC

def eval_step_log():
    with eval_model.graph.as_default():
        loaded_eval_model, global_step = model_builder.create_or_load_model(hparams, eval_model.model, eval_sess)

    eval_sess.run(eval_model.iterator.initializer)
    loss, src, tgts, ids = loaded_eval_model.eval(eval_sess)

    io.print_example(ids, src, tgts, hparams.eval_max_printouts)
    print('EVAL STEP >>> @ Train Step {}: Completed with loss {}'.format(global_step, loss))
    print('Q3: {}'.format(np.round(metrics.q3_infer_accuracy(preds=ids, targets=tgts), 4) * 100))
    print('Q8: {}'.format(np.round(metrics.q8_infer_accuracy(preds=ids, targets=tgts), 4) * 100))

    summary_writer = tf.summary.FileWriter(os.path.join(hparams.model_dir, 'eval'), eval_model.graph)
    loss_summary = tf.Summary(value=[tf.Summary.Value(tag='loss', simple_value=loss)])
    summary_writer.add_summary(loss_summary, global_step)
    summary_writer.close()

# EVAL2 (TEST 2) LOGIC

# Load frag lengths
test_frag_num = 0
test_frags = list();
with open(hparams.data_dir + "test/test_frag.csv", 'r+') as test_frag_file:
    frag_reader = csv.reader(test_frag_file)
    for frag_len in frag_reader:
        test_frags.append(int(frag_len[0]))

def eval2_step_log(test_frag_num):
    with eval2_model.graph.as_default():
        loaded_eval2_model, global_step = model_builder.create_or_load_model(hparams, eval2_model.model, eval2_sess)

    eval2_sess.run(eval2_model.iterator.initializer)
    src, tgts, ids = loaded_eval2_model.infer(eval2_sess)
    if hparams.beam_search:
        ids = ids.transpose([2, 0, 1])   # Change from [batch_size, time_steps, beam_width] to [beam_width, batch_size, time_steps]
        ids = ids[0]  # Only use top 1 prediction from top K

    new_src = list()
    new_tgts = list()
    new_ids = list()

    i = 0
    while i < len(ids):
        j = 0
        k = test_frag_num + i
        while k > 0:
            k -= test_frags[j]
            j += 1
        if k == 0 and len(ids) - i >= test_frags[j]: # Start of protein and all needed frags are present
            new_src.append(stitch(src[i:i+test_frags[j]]))
            new_tgts.append(stitch(tgts[i:i+test_frags[j]]))
            new_ids.append(stitch(ids[i:i+test_frags[j]]))
            i += test_frags[j]
        else:
            i += 1
    test_frag_num += i

    if len(new_src) > 0:
        src = new_src
        tgts = new_tgts
        ids = new_ids

    accuracy = np.round(metrics.q8_infer_accuracy(preds=ids, targets=tgts), 4) * 100
    io.print_example(ids, src, tgts, hparams.infer_max_printouts)
    print('EVAL2 STEP >>> @ Train Step {}: Completed with {}% correct'.format(global_step, accuracy))
    print('Q3: {}'.format(np.round(metrics.q3_infer_accuracy(preds=ids, targets=tgts), 4) * 100))

    summary_writer = tf.summary.FileWriter(os.path.join(hparams.model_dir, 'eval2'), eval2_model.graph)
    loss_summary = tf.Summary(value=[tf.Summary.Value(tag='accuracy', simple_value=accuracy)])
    summary_writer.add_summary(loss_summary, global_step)
    summary_writer.close()

    return accuracy, test_frag_num


# INFER (VALIDATE) LOGIC

# Load frag lengths
validate_frag_num = 0
infer_frags = list();
with open(hparams.data_dir + "validate/validate_frag.csv", 'r+') as infer_frag_file:
    frag_reader = csv.reader(infer_frag_file)
    for frag_len in frag_reader:
        infer_frags.append(int(frag_len[0]))

def infer_step_log(validate_frag_num):
    with infer_model.graph.as_default():
        loaded_infer_model, global_step = model_builder.create_or_load_model(hparams, infer_model.model, infer_sess)

    infer_sess.run(infer_model.iterator.initializer)
    src, tgts, ids = loaded_infer_model.infer(infer_sess)
    if hparams.beam_search:
        ids = ids.transpose([2, 0, 1])   # Change from [batch_size, time_steps, beam_width] to [beam_width, batch_size, time_steps]
        ids = ids[0]  # Only use top 1 prediction from top K

    new_src = list()
    new_tgts = list()
    new_ids = list()

    i = 0
    while i < len(ids):
        j = 0
        k = validate_frag_num + i
        while k > 0:
            k -= infer_frags[j]
            j += 1
        if k == 0 and len(ids) - i >= infer_frags[j]: # Start of protein and all needed frags are present
            new_src.append(stitch(src[i:i+infer_frags[j]]))
            new_tgts.append(stitch(tgts[i:i+infer_frags[j]]))
            new_ids.append(stitch(ids[i:i+infer_frags[j]]))
            i += infer_frags[j]
        else:
            i += 1
    validate_frag_num += i

    if len(new_src) > 0:
        src = new_src
        tgts = new_tgts
        ids = new_ids

    accuracy = np.round(metrics.q8_infer_accuracy(preds=ids, targets=tgts), 4) * 100
    io.print_example(ids, src, tgts, hparams.infer_max_printouts)
    print('INFER STEP >>> @ Train Step {}: Completed with {}% correct'.format(global_step, accuracy))
    print('Q3: {}'.format(np.round(metrics.q3_infer_accuracy(preds=ids, targets=tgts), 4) * 100))

    summary_writer = tf.summary.FileWriter(os.path.join(hparams.model_dir, 'infer'), infer_model.graph)
    loss_summary = tf.Summary(value=[tf.Summary.Value(tag='accuracy', simple_value=accuracy)])
    summary_writer.add_summary(loss_summary, global_step)
    summary_writer.close()

    return accuracy, validate_frag_num

# LOAD TRAIN MODEL

with train_model.graph.as_default():
    loaded_train_model, global_step = model_builder.create_or_load_model(hparams, train_model.model, train_sess)

# START SESSION

print('MODE >>> Training ({} out of {} steps)'.format(global_step, hparams.num_train_steps))
epoch = 0
train_sess.run(train_model.iterator.initializer)

# MAIN LOOP

while global_step < hparams.num_train_steps:
    try:
        _, loss, global_step = loaded_train_model.train(train_sess)

        if global_step % hparams.train_log_freq == 0:
            train_log(global_step)

        if global_step % hparams.eval_log_freq == 0:
            loaded_train_model.saver.save(train_sess, hparams.model_dir, global_step)
            eval_step_log()

        if global_step % hparams.eval2_log_freq == 0:
            loaded_train_model.saver.save(train_sess, hparams.model_dir, global_step)
            _, validate_frag_num = eval2_step_log(test_frag_num)

        if global_step % hparams.infer_log_freq == 0:
            loaded_train_model.saver.save(train_sess, hparams.model_dir, global_step)
            _, validate_frag_num = infer_step_log(validate_frag_num)

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
