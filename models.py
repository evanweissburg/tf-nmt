import tensorflow as tf
from tensorflow.python.layers import core as layers_core


class NMTModel:
    def __init__(self, hparams, iterator, mode):
        source, target_in, target_out, source_lengths, target_lengths = iterator.get_next()

        # Lookup embeddings
        embedding_encoder = tf.get_variable("embedding_encoder", [hparams.src_vsize, hparams.src_emsize])
        encoder_emb_inp = tf.nn.embedding_lookup(embedding_encoder, source)
        embedding_decoder = tf.get_variable("embedding_decoder", [hparams.tgt_vsize, hparams.tgt_emsize])
        decoder_emb_inp = tf.nn.embedding_lookup(embedding_decoder, target_in)

        # Build and run Encoder LSTM
        encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(hparams.num_units)
        encoder_outputs, encoder_state = tf.nn.dynamic_rnn(encoder_cell, encoder_emb_inp, sequence_length=source_lengths, dtype=tf.float32)

        # Build and run Decoder LSTM with TrainingHelper and output projection layer
        decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(hparams.num_units)
        projection_layer = layers_core.Dense(hparams.tgt_vsize, use_bias=False)
        helper = tf.contrib.seq2seq.TrainingHelper(decoder_emb_inp, sequence_length=target_lengths)
        decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, encoder_state, output_layer=projection_layer)
        outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder)
        logits = outputs.rnn_output

        if mode is 'TRAIN' or mode is 'EVAL':  # then calculate loss
            crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_out, logits=logits)
            target_weights = tf.sequence_mask(target_lengths, maxlen=tf.shape(target_out)[1], dtype=logits.dtype)
            self.loss = tf.reduce_sum((crossent * target_weights) / hparams.batch_size)

        if mode is 'TRAIN':  # then calculate/clip gradients, then optimize model
            params = tf.trainable_variables()
            gradients = tf.gradients(self.loss, params)
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, hparams.max_gradient_norm)

            optimizer = tf.train.AdamOptimizer(hparams.l_rate)
            self.update_step = optimizer.apply_gradients(zip(clipped_gradients, params))

        # Designate a saver operation
        self.saver = tf.train.Saver(tf.global_variables())

    def train(self, sess):
        return sess.run([self.update_step, self.loss])

    def eval(self, sess):
        return sess.run([self.loss])

    def infer(self, sess):
        return sess.run([])
