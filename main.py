import random
import numpy as np
import os
import tensorflow as tf
from tensorflow.python.layers import core as layers_core

import utils

print('Beginning run...')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
random.seed(0)
np.random.seed(0)

print('Creating Tensorflow graph...')

EPOCHS = 100
LEARNING_RATE = 0.01
NUM_UNITS = 10
BATCH_SIZE = 2
MAX_GRADIENT_NORM = 1

SRC_VOCAB_SIZE = 20
TGT_VOCAB_SIZE = 8
SRC_EMBED_SIZE = 10
TGT_EMBED_SIZE = 5

# Placeholders
encoder_inputs = tf.placeholder(tf.int32, shape=[max_encoder_time, BATCH_SIZE])
decoder_inputs = tf.placeholder(tf.int32, shape=[max_decoder_time, BATCH_SIZE])
decoder_outputs = tf.placeholder(tf.int32, shape=[max_decoder_time, BATCH_SIZE])

# Look up embedding:
#   encoder_inputs: [max_time, batch_size]
#   encoder_emb_inp: [max_time, batch_size, embedding_size]
# Encoder Embeddings
embedding_encoder = tf.get_variable("embedding_encoder", [SRC_VOCAB_SIZE, SRC_EMBED_SIZE], ...)
encoder_emb_inp = tf.nn.embedding_lookup(embedding_encoder, encoder_inputs)
# Decoder Embeddings
embedding_decoder = tf.get_variable("embedding_encoder", [TGT_VOCAB_SIZE, TGT_EMBED_SIZE], ...)
decoder_emb_inp = tf.nn.embedding_lookup(embedding_decoder, decoder_inputs)

# Build first RNN cell
encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(NUM_UNITS)
# Run Dynamic RNN
#   encoder_outpus: [max_time, batch_size, num_units]
#   encoder_state: [batch_size, num_units]
encoder_outputs, encoder_state = tf.nn.dynamic_rnn(encoder_cell, encoder_emb_inp, sequence_length=source_sequence_length, time_major=True)

# Build second RNN cell
decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(NUM_UNITS)
# Build a projection layer (will be placed on top of Decoder LSTM)
projection_layer = layers_core.Dense(TGT_VOCAB_SIZE, use_bias=False)
# Helper (for training)
helper = tf.contrib.seq2seq.TrainingHelper(decoder_emb_inp, decoder_lengths, time_major=True)
# Decoder
decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, encoder_state, output_layer=projection_layer)
# Dynamic decoding
outputs, _ = tf.contrib.seq2seq.dynamic_decode(decoder, ...)
logits = outputs.rnn_output

crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=decoder_outputs, logits=logits)
train_loss = (tf.reduce_sum(crossent * target_weights) / BATCH_SIZE)

# Calculate and clip gradients
params = tf.trainable_variables()
gradients = tf.gradients(train_loss, params)
clipped_gradients, _ = tf.clip_by_global_norm(gradients, MAX_GRADIENT_NORM)

# Optimization
optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
update_step = optimizer.apply_gradients(zip(clipped_gradients, params))

print('Tensorflow graph created.')

with tf.Session() as sess:
    primary, secondary = utils.read_integerized_input()

    print('Initializing Tensorflow variables.')
    sess.run(tf.global_variables_initializer())

    for epoch in range(EPOCHS):
        prim_inputs, sec_inputs, sec_targets = utils.get_batch(primary, secondary)
        feed_dict = {encoder_inputs: prim_inputs, decoder_inputs: sec_inputs, decoder_outputs: sec_targets}
        _, loss = sess.run([update_step, train_loss], feed_dict=feed_dict)
        print('Epoch %s, Loss %s' % (epoch, loss))

print('Program finished successfully.')