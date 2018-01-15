import numpy as np
import os
import tensorflow as tf
import itertools

import model_builder
import hyper_params

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
tf.set_random_seed(0)
np.set_printoptions(linewidth=10000, threshold=1000000000)

SAVE_MODEL_DIRECTORY = '/home/nave01314/Documents/tf-nmt-models/'
try:
    import shutil
    shutil.rmtree(SAVE_MODEL_DIRECTORY)
except Exception:
    print('no folder')

TRAIN_PRINT_FREQ = 5       # how many batches between evaluating loss
EVAL_PRINT_FREQ = 100
EVAL_MAX_PRINTOUTS = 5
INFER_PRINT_FREQ = 500
INFER_MAX_PRINTOUTS = 5

EPOCHS = 2000
LEARNING_RATE = 0.001
NUM_UNITS = 10
BATCH_SIZE = 400
MAX_GRADIENT_NORM = 1

SRC_VOCAB_SIZE = 27  # A-Z + padding
TGT_VOCAB_SIZE = 11  # 8 + padding + sos + eos
SRC_EMBED_SIZE = 15
TGT_EMBED_SIZE = 10

START_TOKEN = 1
END_TOKEN = 2
SRC_PADDING = 0
TGT_PADDING = 0
SHUFFLE_SEED = 0
SHUFFLE_BUFFER_SIZE = 10000
NUM_BUCKETS = 10
MAX_LEN = None

hparams = hyper_params.HParams(SAVE_MODEL_DIRECTORY, LEARNING_RATE, NUM_UNITS, BATCH_SIZE, MAX_GRADIENT_NORM, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, SRC_EMBED_SIZE, TGT_EMBED_SIZE, START_TOKEN, END_TOKEN, SRC_PADDING, TGT_PADDING, SHUFFLE_SEED, SHUFFLE_BUFFER_SIZE, NUM_BUCKETS, MAX_LEN)

train_model = model_builder.create_train_model(hparams)
eval_model = model_builder.create_eval_model(hparams)
infer_model = model_builder.create_infer_model(hparams)

train_sess = tf.Session(graph=train_model.graph)
eval_sess = tf.Session(graph=eval_model.graph)
infer_sess = tf.Session(graph=infer_model.graph)

with train_model.graph.as_default():
    loaded_train_model = model_builder.create_or_load_model(hparams, train_model.model, train_sess)

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
                loss, src, targetsin, targetsout, logits = loaded_eval_model.eval(eval_sess)
                print('EVAL BEGIN >>> Epoch {}: Batch {} completed with loss {}'.format(epoch, batch, loss))
                predictions = np.argmax(logits, axis=2)
                for i in range(min(len(targetsout), EVAL_MAX_PRINTOUTS)):
                    frmt = '{:>3}'*len(targetsout[i])
                    print('>>> START PROTEIN <<<')
                    print('Targetin   :' + frmt.format(*targetsin[i]))
                    print('Targetout  :' + frmt.format(*targetsout[i]))
                    print('Prediction :' + frmt.format(*predictions[i]))
                    print('Source     :' + frmt.format(*np.insert(src[i], list(src[i]).index(0), [-1]) if src[i][-1] == 0 else np.append(src[i], [-1])))
                print('EVAL END >>> Epoch {}: Batch {} completed with loss {}'.format(epoch, batch, loss))
            if batch % INFER_PRINT_FREQ == 0:
                loaded_train_model.saver.save(train_sess, hparams.model_dir)
                with infer_model.graph.as_default():
                    loaded_infer_model = model_builder.create_or_load_model(hparams, infer_model.model, infer_sess)
                infer_sess.run(infer_model.iterator.initializer)
                src, targetsin, targetsout, logits = loaded_infer_model.infer(infer_sess)
                print('INFER BEGIN >>> Epoch {}: Batch {} completed'.format(epoch, batch, loss))
                predictions = np.argmax(logits, axis=2)
                for i in range(min(len(targetsout), INFER_MAX_PRINTOUTS)):
                    frmt = '{:>3}'*len(targetsout[i])
                    print('>>> START PROTEIN <<<')
                    print('Targetin   :' + frmt.format(*targetsin[i]))
                    print('Targetout  :' + frmt.format(*targetsout[i]))
                    print('Prediction :' + frmt.format(*predictions[i]))
                    print('Source     :' + frmt.format(*np.insert(src[i], list(src[i]).index(0), [-1]) if src[i][-1] == 0 else np.append(src[i], [-1])))
                print('INFER END >>> Epoch {}: Batch {} completed'.format(epoch, batch, loss))
        except tf.errors.OutOfRangeError:
            print('Epoch {} completed with average loss {}'.format(epoch, epoch_loss/(batch+1)))
            break
