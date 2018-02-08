import numpy as np
import tensorflow as tf
import os

import hparams_setup
import utils
import model_builder


np.set_printoptions(linewidth=10000, threshold=1000000000)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

hparams = hparams_setup.get_hparams()
utils.clear_previous_runs(data_dir=hparams.data_dir, model_dir=None, log_dir=None)
utils.prep_nmt_dataset(hparams)

train_model = model_builder.create_train_model(hparams)
eval_model = model_builder.create_eval_model(hparams)
infer_model = model_builder.create_infer_model(hparams)
pred_model = model_builder.create_pred_model(hparams)

train_sess = tf.Session(graph=train_model.graph)
eval_sess = tf.Session(graph=eval_model.graph)
infer_sess = tf.Session(graph=infer_model.graph)
pred_sess = tf.Session(graph=pred_model.graph)


def train_log(global_step):
    print('TRAIN STEP >>> @ Train Step {}: Completed with loss {}'.format(global_step, loss))

    summary_writer = tf.summary.FileWriter(os.path.join(hparams.log_dir, 'train'), train_model.graph)
    loss_summary = tf.Summary(value=[tf.Summary.Value(tag='loss', simple_value=loss)])
    summary_writer.add_summary(loss_summary, global_step)
    summary_writer.close()


def eval_step_log():
    with eval_model.graph.as_default():
        loaded_eval_model, global_step = model_builder.create_or_load_model(hparams, eval_model.model, eval_sess)

    eval_sess.run(eval_model.iterator.initializer)
    loss, src, tgts, ids = loaded_eval_model.eval(eval_sess)

    utils.print_example(ids, src, tgts, hparams.eval_max_printouts)
    print('EVAL STEP >>> @ Train Step {}: Completed with loss {}'.format(global_step, loss))
    utils.print_common_mistake(ids, src, tgts)

    summary_writer = tf.summary.FileWriter(os.path.join(hparams.log_dir, 'eval'), eval_model.graph)
    loss_summary = tf.Summary(value=[tf.Summary.Value(tag='loss', simple_value=loss)])
    summary_writer.add_summary(loss_summary, global_step)
    summary_writer.close()


def infer_step_log():
    with infer_model.graph.as_default():
        loaded_infer_model, global_step = model_builder.create_or_load_model(hparams, infer_model.model, infer_sess)

    infer_sess.run(infer_model.iterator.initializer)
    src, tgts, ids = loaded_infer_model.infer(infer_sess)
    if hparams.beam_search:
        ids = ids.transpose([2, 0, 1])   # Change from [batch_size, time_steps, beam_width] to [beam_width, batch_size, time_steps]
        ids = ids[0]  # Only use top 1 prediction from top K
    accuracy = np.round(utils.percent_infer_accuracy(preds=ids, targets=tgts), 4) * 100

    utils.print_example(ids, src, tgts, hparams.infer_max_printouts)
    print('INFER STEP >>> @ Train Step {}: Completed with {}% correct'.format(global_step, accuracy))

    summary_writer = tf.summary.FileWriter(os.path.join(hparams.log_dir, 'infer'), infer_model.graph)
    loss_summary = tf.Summary(value=[tf.Summary.Value(tag='accuracy', simple_value=accuracy)])
    summary_writer.add_summary(loss_summary, global_step)
    summary_writer.close()
    return accuracy


with train_model.graph.as_default():
    loaded_train_model, global_step = model_builder.create_or_load_model(hparams, train_model.model, train_sess)

print('MODE >>> Training ({} out of {} steps)'.format(global_step, hparams.num_train_steps))
epoch = 0
train_sess.run(train_model.iterator.initializer)

n = 0
total_loss = 0
square_sum = 0
stdev = 0

m = 0
total_accuracy = 0
accuracy_square_sum = 0
accuracy_stdev = 0

while global_step < hparams.num_train_steps:

    try:
        _, loss, global_step = loaded_train_model.train(train_sess)

        n += 1
        total_loss, square_sum, stdev = utils.update_standard_deviation(total_loss, square_sum, n, loss)

        if global_step % hparams.train_log_freq == 0:
            train_log(global_step)
            print('Standard deviation: {}'.format(stdev))

        if global_step % hparams.eval_log_freq == 0:
            loaded_train_model.saver.save(train_sess, hparams.model_dir, global_step)
            eval_step_log()

        if global_step % hparams.infer_log_freq == 0:
            m += 1
            accuracy = loaded_train_model.saver.save(train_sess, hparams.model_dir, global_step)
            total_accuracy, accuracy_square_sum, accuracy_stdev = utils.update_standard_deviation(total_accuracy, accuracy_square_sum, m, accuracy)
            print('Accuracy standard deviation: {}'.format(accuracy_stdev))
            infer_step_log()

    except tf.errors.OutOfRangeError:
        print('Epoch {} completed.'.format(epoch))
        train_sess.run(train_model.iterator.initializer)
        epoch += 1

print()
print('MODE >>> Prediction')

while True:
    src = utils.get_inference_input()

    with pred_sess as sess:
        with pred_model.graph.as_default():
            loaded_pred_model, _ = model_builder.create_or_load_model(hparams, pred_model.model, sess)

        sess.run(pred_model.iterator.initializer, feed_dict={pred_model.src_placeholder: [src]})
        ids = loaded_pred_model.pred(pred_sess)
        if hparams.beam_search:
            ids = ids.transpose([2, 0, 1])   # Change from [batch_size, time_steps, beam_width] to [beam_width, batch_size, time_steps]
            ids = ids[0]  # Only use top 1 prediction from top K
        src = src.split(',')+['-1']

        utils.print_example(ids, [src])
        print()
