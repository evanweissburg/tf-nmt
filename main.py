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
utils.clear_previous_runs(model_dir=hparams.model_dir, data_dir=hparams.data_dir)
utils.make_dataset(max_len=hparams.max_len, max_size=hparams.dataset_max_size, data_dir=hparams.data_dir)

train_model = model_builder.create_train_model(hparams)
eval_model = model_builder.create_eval_model(hparams)
infer_model = model_builder.create_infer_model(hparams)

train_sess = tf.Session(graph=train_model.graph)
eval_sess = tf.Session(graph=eval_model.graph)
infer_sess = tf.Session(graph=infer_model.graph)

with train_model.graph.as_default():
    loaded_train_model = model_builder.create_or_load_model(hparams, train_model.model, train_sess)

for epoch in range(hparams.epochs):
    train_sess.run(train_model.iterator.initializer)
    epoch_loss = 0
    for batch in itertools.count():
        try:
            _, loss = loaded_train_model.train(train_sess)
            epoch_loss += loss

            if batch % hparams.train_print_freq == 0:
                print('TRAIN STEP >>> Epoch {}: Batch {} completed with loss {}'.format(epoch, batch, loss))

            if batch % hparams.eval_print_freq == 0:
                loaded_train_model.saver.save(train_sess, hparams.model_dir)
                with eval_model.graph.as_default():
                    loaded_eval_model = model_builder.create_or_load_model(hparams, eval_model.model, eval_sess)
                eval_sess.run(eval_model.iterator.initializer)
                loss, src, tgts, logits = loaded_eval_model.eval(eval_sess)
                print('EVAL BEGIN >>> Epoch {}: Batch {} completed with loss {}'.format(epoch, batch, loss))
                utils.print_prots(logits, src, tgts, hparams.eval_max_printouts)
                print('EVAL END >>> Epoch {}: Batch {} completed with loss {}'.format(epoch, batch, loss))

            if batch % hparams.infer_print_freq == 0:
                loaded_train_model.saver.save(train_sess, hparams.model_dir)
                with infer_model.graph.as_default():
                    loaded_infer_model = model_builder.create_or_load_model(hparams, infer_model.model, infer_sess)
                infer_sess.run(infer_model.iterator.initializer)
                src, tgts, logits = loaded_infer_model.infer(infer_sess)
                print('INFER BEGIN >>> Epoch {}: Batch {} completed'.format(epoch, batch, loss))
                utils.print_prots(logits, src, tgts, hparams.infer_max_printouts)
                print('INFER END >>> Epoch {}: Batch {} completed'.format(epoch, batch, loss))

        except tf.errors.OutOfRangeError:
            print('Epoch {} completed with average loss {}'.format(epoch, epoch_loss/(batch+1)))
            break
