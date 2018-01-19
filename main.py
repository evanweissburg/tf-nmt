import numpy as np
import tensorflow as tf
import itertools

import utils
import model_builder

np.set_printoptions(linewidth=10000, threshold=1000000000)

SAVE_MODEL_DIRECTORY = '/Users/ianbulovic/Documents/Other/tf-nmt-models/tf-nmt-models'

# Sets calculation frequency (modulo per batch) and quantity of output
TRAIN_PRINT_FREQ = 10
EVAL_PRINT_FREQ = 100
EVAL_MAX_PRINTOUTS = 5
INFER_PRINT_FREQ = 100
INFER_MAX_PRINTOUTS = 5

# Standard hparams
EPOCHS = 2000
LEARNING_RATE = 0.001
NUM_UNITS = 50
BATCH_SIZE = 500
MAX_GRADIENT_NORM = 5.0

# VSize/EmSize
SRC_VOCAB_SIZE = 27  # A-Z + padding
TGT_VOCAB_SIZE = 11  # 8 + padding + sos + eos
SRC_EMBED_SIZE = 15
TGT_EMBED_SIZE = 10

# Integer sos, eos, pad tokens
START_TOKEN = 1
END_TOKEN = 2
SRC_PADDING = 0
TGT_PADDING = 0

# Misc
SHUFFLE_SEED = 0
SHUFFLE_BUFFER_SIZE = 10000
NUM_BUCKETS = 1
MAX_LEN = None

hparams = tf.contrib.training.HParams(model_dir=SAVE_MODEL_DIRECTORY, l_rate=LEARNING_RATE, num_units=NUM_UNITS,
                                      batch_size=BATCH_SIZE, max_gradient_norm=MAX_GRADIENT_NORM,
                                      src_vsize=SRC_VOCAB_SIZE, tgt_vsize=TGT_VOCAB_SIZE, src_emsize=SRC_EMBED_SIZE,
                                      tgt_emsize=TGT_EMBED_SIZE, sos=START_TOKEN, eos=END_TOKEN, src_pad=SRC_PADDING,
                                      tgt_pad=TGT_PADDING, shuffle_seed=SHUFFLE_SEED,
                                      shuffle_buffer_size=SHUFFLE_BUFFER_SIZE, num_buckets=NUM_BUCKETS, max_len=MAX_LEN)

# Clear SAVE_MODEL_DIRECTORY before each run (fresh start)
import shutil
shutil.rmtree(hparams.model_dir, ignore_errors=True)

train_model = model_builder.create_train_model(hparams)
eval_model = model_builder.create_eval_model(hparams)
infer_model = model_builder.create_infer_model(hparams)

train_sess = tf.Session(graph=train_model.graph)
eval_sess = tf.Session(graph=eval_model.graph)
infer_sess = tf.Session(graph=infer_model.graph)

with train_model.graph.as_default():
    loaded_train_model = model_builder.create_or_load_model(hparams, train_model.model, train_sess)
NUM_BUCKETS = 1
MAX_LEN = None
for epoch in range(EPOCHS):
    train_sess.run(train_model.iterator.initializer)
    epoch_loss = 0
    for batch in itertools.count():
        try:
            _, loss = loaded_train_model.train(train_sess)
            epoch_loss += loss

            if batch % TRAIN_PRINT_FREQ == 0:
                print('TRAIN STEP >>> Epoch {}: Batch {} completed with loss {}'.format(epoch, batch, loss))

            if batch % EVAL_PRINT_FREQ == 0:
                loaded_train_model.saver.save(train_sess, hparams.model_dir)
                with eval_model.graph.as_default():
                    loaded_eval_model = model_builder.create_or_load_model(hparams, eval_model.model, eval_sess)
                eval_sess.run(eval_model.iterator.initializer)
                loss, src, tgts, logits = loaded_eval_model.eval(eval_sess)
                print('EVAL BEGIN >>> Epoch {}: Batch {} completed with loss {}'.format(epoch, batch, loss))
                utils.print_prots(logits, src, tgts, EVAL_MAX_PRINTOUTS)
                print('EVAL END >>> Epoch {}: Batch {} completed with loss {}'.format(epoch, batch, loss))

            if batch % INFER_PRINT_FREQ == 0:
                loaded_train_model.saver.save(train_sess, hparams.model_dir)
                with infer_model.graph.as_default():
                    loaded_infer_model = model_builder.create_or_load_model(hparams, infer_model.model, infer_sess)
                infer_sess.run(infer_model.iterator.initializer)
                src, tgts, logits = loaded_infer_model.infer(infer_sess)
                print('INFER BEGIN >>> Epoch {}: Batch {} completed'.format(epoch, batch, loss))
                utils.print_prots(logits, src, tgts, INFER_MAX_PRINTOUTS)
                print('INFER END >>> Epoch {}: Batch {} completed'.format(epoch, batch, loss))

        except tf.errors.OutOfRangeError:
            print('Epoch {} completed with average loss {}'.format(epoch, epoch_loss/(batch+1)))
            break
