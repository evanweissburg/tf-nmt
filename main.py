import numpy as np
import os
import tensorflow as tf
import itertools

import data_pipeline
import nmt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
tf.set_random_seed(0)
np.set_printoptions(linewidth=10000, threshold=1000000000)

PRINT_FREQ = 1       # how often should loss be evaluated
EPOCHS = 20
LEARNING_RATE = 0.01
NUM_UNITS = 20
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

train_graph = tf.Graph()

with train_graph.as_default():
    train_iterator = data_pipeline.get_batched_iterator(BATCH_SIZE, START_TOKEN, END_TOKEN, SRC_PADDING, TGT_PADDING, SHUFFLE_SEED, SHUFFLE_BUFFER_SIZE)
    train_model = nmt.TrainingModel(train_iterator, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, SRC_EMBED_SIZE, TGT_EMBED_SIZE,
                                    NUM_UNITS, BATCH_SIZE, MAX_GRADIENT_NORM, LEARNING_RATE)
    initializer = tf.global_variables_initializer()

train_sess = tf.Session(graph=train_graph)

train_sess.run(initializer)
train_sess.run(train_iterator.initializer)

for epoch in range(EPOCHS):
    epoch_loss = 0
    for batch in itertools.count():
        try:
            _, loss = train_model.train(train_sess)
            epoch_loss += loss
            if batch % 100 == 0:
                print('Batch {} completed with loss {}'.format(batch, loss))
        except tf.errors.OutOfRangeError:
            print('Epoch {} completed with average loss {}'.format(epoch, epoch_loss/(batch+1)))
            break

# What print outs can look like (for eval later)
#   outputs, targets, sources = sess.run([logits, target_out, source])
#   predictions = np.argmax(outputs, axis=2)
#   for i in range(len(targets)):
#       frmt = '{:>3}'*len(targets[i])
#       print('>>> START PROTEIN <<<')
#       print('Target     :' + frmt.format(*targets[i]))
#       print('Prediction :' + frmt.format(*predictions[i]))
#       print('Source     :' + frmt.format(*np.insert(sources[i], list(sources[i]).index(0), [-1]) if sources[i][-1] == 0 else np.append(sources[i], [-1])))

