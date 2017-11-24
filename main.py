import random
import numpy as np
import os
import tensorflow as tf

import utils
import embed

print('Beginning run...')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
random.seed(0)
np.random.seed(0)

primary, secondary = utils.read_integerized_input()
embed.generate_embeddings(primary, batch_size=10, epochs=100000, embedding_size=10, learning_rate=1.0,
                          vocabulary_size=20, window_radius=3, repetition=1)
embed.generate_embeddings(secondary, batch_size=10, epochs=100000, embedding_size=5, learning_rate=1.0,
                          vocabulary_size=8, window_radius=3, repetition=1)

print('Run complete.')

# LEARNING_RATE = 0.01
# HIDDEN_SIZE = 50
#
# src_vocab_size = 20
# tgt_vocab_size = 8
# batch_size = 10
# max_gradient_norm = 1
# embedding_size = 10
#
# labels, primary, secondary = utils.read_input("ss.txt")
#
# encoder_inputs = primary[0]
# decoder_inputs = secondary[0]
#
# # Embedding
# embedding_encoder = tf.get_variable("embedding_encoder", [src_vocab_size, embedding_size], ...)
# # Look up embedding:
# #   encoder_inputs: [max_time, batch_size]
# #   encoder_emb_inp: [max_time, batch_size, embedding_size]
# encoder_emb_inp = tf.nn.embedding_lookup(embedding_encoder, encoder_inputs)
# # New code
# embedding_decoder = tf.get_variable("embedding_decoder", [src_vocab_size, embedding_size], ...)
# decoder_emb_inp = tf.nn.embedding_lookup(embedding_encoder, decoder_inputs)
#
# # Build RNN cell
# encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)
#
# # Run Dynamic RNN
# #   encoder_outpus: [max_time, batch_size, num_units]
# #   encoder_state: [batch_size, num_units]
# encoder_outputs, encoder_state = tf.nn.dynamic_rnn(encoder_cell, encoder_emb_inp, sequence_length=source_sequence_length, time_major=True)
#
# # Build RNN cell
# decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)
#
# # Helper
# helper = tf.contrib.seq2seq.TrainingHelper(decoder_emb_inp, decoder_lengths, time_major=True)
# # Decoder
# decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, encoder_state, output_layer=projection_layer)
# # Dynamic decoding
# outputs, _ = tf.contrib.seq2seq.dynamic_decode(decoder, ...)
# logits = outputs.rnn_output
#
# projection_layer = layers_core.Dense(tgt_vocab_size, use_bias=False)
#
# crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=decoder_outputs, logits=logits)
# train_loss = (tf.reduce_sum(crossent * target_weights) / batch_size)
#
# # Calculate and clip gradients
# params = tf.trainable_variables()
# gradients = tf.gradients(train_loss, params)
# clipped_gradients, _ = tf.clip_by_global_norm(
#     gradients, max_gradient_norm)
#
# # Optimization
# optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
# update_step = optimizer.apply_gradients(
#     zip(clipped_gradients, params))