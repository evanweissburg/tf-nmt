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
np.set_printoptions(linewidth=10000)

print('Creating Tensorflow graph...')

PRINT_FREQ = 100       # how often should loss be evaluated
PRINT_EXAMPLES = 5     # num of example proteins to print out every time loss is evaluated

EPOCHS = 10000
LEARNING_RATE = 0.001
NUM_UNITS = 10
BATCH_SIZE = 5
MAX_GRADIENT_NORM = 1

SRC_VOCAB_SIZE = 24
TGT_VOCAB_SIZE = 9
SRC_EMBED_SIZE = 10
TGT_EMBED_SIZE = 5

# Data pipeline
if not (os.path.exists('primary.csv') and os.path.exists('secondary.csv')):
    utils.integerize_raw_data()
source_dataset = tf.data.TextLineDataset('primary.csv')
target_dataset = tf.data.TextLineDataset('secondary.csv')
source_dataset = source_dataset.map(lambda protein: tf.string_to_number(tf.string_split([protein], delimiter=',').values, tf.int32))
target_dataset = target_dataset.map(lambda protein: tf.string_to_number(tf.string_split([protein], delimiter=',').values, tf.int32))
source_dataset = source_dataset.map(lambda words: (words, tf.size(words)))
target_dataset = target_dataset.map(lambda words: (words, tf.size(words)))
dataset = tf.data.Dataset.zip((source_dataset, target_dataset))
batched_dataset = dataset.padded_batch(BATCH_SIZE,
                                       padded_shapes=((tf.TensorShape([None]), tf.TensorShape([])),   # source, lengths
                                                      (tf.TensorShape([None]), tf.TensorShape([]))),  # targets, lengths
                                       padding_values=((0, 0),   # Source padding, size padding (no size padding -- type is int)
                                                       (0, 0)))  # Target padding, size padding (see above)
batched_iterator = batched_dataset.make_initializable_iterator()
((source, source_lengths), (target, target_lengths)) = batched_iterator.get_next()

# Lookup embeddings
embedding_encoder = tf.get_variable("embedding_encoder", [SRC_VOCAB_SIZE, SRC_EMBED_SIZE])
encoder_emb_inp = tf.nn.embedding_lookup(embedding_encoder, source)
embedding_decoder = tf.get_variable("embedding_decoder", [TGT_VOCAB_SIZE, TGT_EMBED_SIZE])
decoder_emb_inp = tf.nn.embedding_lookup(embedding_decoder, target)

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
crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target, logits=logits)
target_weights = tf.sequence_mask(target_lengths, maxlen=tf.shape(target)[1], dtype=logits.dtype)
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
            loss, outputs, targets, sources = sess.run([train_loss, logits, target, source])
            predictions = np.argmax(outputs, axis=2)
            for i in range(PRINT_EXAMPLES):
                print('>>> START PROTEIN <<<\n Target     :%s\n Prediction :%s\n Source     :%s' % (targets[i], predictions[i], sources[i]))
            print('Epoch %s, Loss %s\n' % (epoch, loss))

print('Program finished successfully.')
