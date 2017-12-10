import numpy as np
import os
import tensorflow as tf
from tensorflow.python.layers import core as layers_core

import data_pipeline as pipeline

print('Beginning run...')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
np.random.seed(0)
np.set_printoptions(linewidth=10000)

print('Creating Tensorflow graph...')

PRINT_FREQ = 100       # how often should loss be evaluated
PRINT_EXAMPLES = 5     # num of example proteins to print out every time loss is evaluated

EPOCHS = 100000
LEARNING_RATE = 0.0001
NUM_UNITS = 20
BATCH_SIZE = 5
MAX_GRADIENT_NORM = 1

SRC_VOCAB_SIZE = 27  # A-Z + padding (0)
TGT_VOCAB_SIZE = 11  # 8 + padding (0) + sos + eos
SRC_EMBED_SIZE = 15
TGT_EMBED_SIZE = 10

START_TOKEN = 1
END_TOKEN = 2

# Data pipeline
batched_iterator, source, source_lengths, target_in, target_out, target_lengths = pipeline.get_batched_iterator(BATCH_SIZE, START_TOKEN, END_TOKEN)

# Lookup embeddings
embedding_encoder = tf.get_variable("embedding_encoder", [SRC_VOCAB_SIZE, SRC_EMBED_SIZE])
encoder_emb_inp = tf.nn.embedding_lookup(embedding_encoder, source)
embedding_decoder = tf.get_variable("embedding_decoder", [TGT_VOCAB_SIZE, TGT_EMBED_SIZE])
decoder_emb_inp = tf.nn.embedding_lookup(embedding_decoder, target_in)

# Build and run Encoder LSTM
encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(NUM_UNITS)
encoder_outputs, encoder_state = tf.nn.dynamic_rnn(encoder_cell, encoder_emb_inp, sequence_length=source_lengths, dtype=tf.float32)

# Build and run Decoder LSTM with TrainingHelper and output projection layer
decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(NUM_UNITS)
projection_layer = layers_core.Dense(TGT_VOCAB_SIZE, use_bias=False)
helper = tf.contrib.seq2seq.TrainingHelper(decoder_emb_inp, sequence_length=target_lengths)
decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, encoder_state, output_layer=projection_layer)
outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder)
logits = outputs.rnn_output

# Calculate loss
crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_out, logits=logits)
target_weights = tf.sequence_mask(target_lengths, maxlen=tf.shape(target_out)[1], dtype=logits.dtype)
train_loss = tf.reduce_sum(crossent * target_weights / BATCH_SIZE)

# Calculate and clip gradients
params = tf.trainable_variables()
gradients = tf.gradients(train_loss, params)
clipped_gradients, _ = tf.clip_by_global_norm(gradients, MAX_GRADIENT_NORM)

# Optimize model
optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
update_step = optimizer.apply_gradients(zip(clipped_gradients, params))

print('Tensorflow graph created.')

with tf.Session() as sess:
    print('Initializing Tensorflow variables.')
    sess.run(tf.global_variables_initializer())
    sess.run(batched_iterator.initializer)

    for epoch in range(EPOCHS):
        _ = sess.run([update_step])
        if epoch % PRINT_FREQ == 0:
            loss, outputs, targets, sources = sess.run([train_loss, logits, target_out, source])
            predictions = np.argmax(outputs, axis=2)
            for i in range(PRINT_EXAMPLES):
                frmt = '{:>3}'*len(targets[i])
                print('>>> START PROTEIN <<<')
                print('Target     :' + frmt.format(*targets[i]))
                print('Prediction :' + frmt.format(*predictions[i]))
                print('Source     :' + frmt.format(*np.insert(sources[i], list(sources[i]).index(0), [-1]) if sources[i][-1] == 0 else np.append(sources[i], [-1])))
            print('Epoch %s, Loss %s\n' % (epoch, loss))

print('Program finished successfully.')
