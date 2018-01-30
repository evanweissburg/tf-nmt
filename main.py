import numpy as np
import tensorflow as tf
import itertools
import os

import hparams_setup
import utils
import model_builder

np.set_printoptions(linewidth=10000, threshold=1000000000)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

hparams = hparams_setup.get_hparams()
utils.clear_previous_runs(model_dir=None, data_dir=hparams.data_dir, log_dir=hparams.log_dir)
utils.make_dataset(max_len=hparams.max_len, max_size=hparams.dataset_max_size, data_dir=hparams.data_dir,
                   max_weight=hparams.max_weight, delta_weight=hparams.delta_weight, min_weight=hparams.min_weight)

train_model = model_builder.create_train_model(hparams)
eval_model = model_builder.create_eval_model(hparams)
infer_model = model_builder.create_infer_model(hparams)

train_sess = tf.Session(graph=train_model.graph)
eval_sess = tf.Session(graph=eval_model.graph)
infer_sess = tf.Session(graph=infer_model.graph)

with train_model.graph.as_default():
    loaded_train_model = model_builder.create_or_load_model(hparams, train_model.model, train_sess)

num_batches = 0
for epoch in range(hparams.epochs):
    train_sess.run(train_model.iterator.initializer)
    epoch_loss = 0
    for batch in itertools.count():
        try:
            _, loss = loaded_train_model.train(train_sess)
            epoch_loss += loss

            if batch % hparams.train_log_freq == 0:
                print('TRAIN STEP >>> Epoch {}: Batch {} completed with loss {}'.format(epoch, batch, loss))

                summary_writer = tf.summary.FileWriter(os.path.join(hparams.log_dir, 'train'), train_model.graph)
                loss_summary = tf.Summary(value=[tf.Summary.Value(tag='loss', simple_value=loss)])
                summary_writer.add_summary(loss_summary, epoch*num_batches+batch)
                summary_writer.close()

            if batch % hparams.eval_log_freq == 0:
                loaded_train_model.saver.save(train_sess, hparams.model_dir)
                with eval_model.graph.as_default():
                    loaded_eval_model = model_builder.create_or_load_model(hparams, eval_model.model, eval_sess)

                eval_sess.run(eval_model.iterator.initializer)
                loss, src, tgts, ids = loaded_eval_model.eval(eval_sess)

                print('EVAL BEGIN >>> Epoch {}: Batch {} completed with loss {}'.format(epoch, batch, loss))
                utils.print_prots(ids, src, tgts, hparams.eval_max_printouts)
                print('EVAL END >>> Epoch {}: Batch {} completed with loss {}'.format(epoch, batch, loss))

                summary_writer = tf.summary.FileWriter(os.path.join(hparams.log_dir, 'eval'), eval_model.graph)
                loss_summary = tf.Summary(value=[tf.Summary.Value(tag='loss', simple_value=loss)])
                summary_writer.add_summary(loss_summary, epoch*num_batches+batch)
                summary_writer.close()

            if batch % hparams.infer_log_freq == 0:
                loaded_train_model.saver.save(train_sess, hparams.model_dir)
                with infer_model.graph.as_default():
                    loaded_infer_model = model_builder.create_or_load_model(hparams, infer_model.model, infer_sess)

                infer_sess.run(infer_model.iterator.initializer)
                src, tgts, ids = loaded_infer_model.infer(infer_sess)
                if hparams.beam_search:
                    ids = ids.transpose([2, 0, 1])   # Change from [batch_size, time_steps, beam_width] to [beam_width, batch_size, time_steps]
                    ids = ids[0]  # Only use top 1 prediction from top K
                accuracy = np.round(utils.percent_infer_accuracy(preds=ids, targets=tgts), 4) * 100

                print('INFER BEGIN >>> Epoch {}: Batch {} completed with {}% correct'.format(epoch, batch, accuracy))
                utils.print_prots(ids, src, tgts, hparams.infer_max_printouts)
                print('INFER END >>> Epoch {}: Batch {} completed with {}% correct'.format(epoch, batch, accuracy))

                summary_writer = tf.summary.FileWriter(os.path.join(hparams.log_dir, 'infer'), infer_model.graph)
                loss_summary = tf.Summary(value=[tf.Summary.Value(tag='accuracy', simple_value=accuracy)])
                summary_writer.add_summary(loss_summary, epoch*num_batches+batch)
                summary_writer.close()

        except tf.errors.OutOfRangeError:
            num_batches = batch + 1
            print('Epoch {} completed with average loss {}'.format(epoch, epoch_loss/num_batches))
            break
