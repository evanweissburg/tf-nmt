import tensorflow as tf
from tensorflow.python.layers import core as layers_core


class NMTModel:
    def __init__(self, hparams, iterator, mode):
        tf.set_random_seed(hparams.graph_seed)
        source, target_in, target_out, weights, source_lengths, target_lengths = iterator.get_next()
        true_batch_size = tf.size(source_lengths)

        # Lookup embeddings
        embedding_encoder = tf.get_variable("embedding_encoder", [hparams.src_vsize, hparams.src_emsize])
        encoder_emb_inp = tf.nn.embedding_lookup(embedding_encoder, source)
        embedding_decoder = tf.get_variable("embedding_decoder", [hparams.tgt_vsize, hparams.tgt_emsize])
        decoder_emb_inp = tf.nn.embedding_lookup(embedding_decoder, target_in)

        # Build and run Encoder LSTM
        encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(hparams.num_units)
        encoder_outputs, encoder_state = tf.nn.dynamic_rnn(encoder_cell, encoder_emb_inp, sequence_length=source_lengths, dtype=tf.float32)

        # Build Decoder cell
        decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(hparams.num_units)

        if hparams.attention:
            attention_mechanism = tf.contrib.seq2seq.LuongAttention(hparams.num_units, encoder_outputs, memory_sequence_length=source_lengths)
            decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism, attention_layer_size=hparams.num_units)
            decoder_initial_state = decoder_cell.zero_state(hparams.batch_size, tf.float32).clone(cell_state=encoder_state)
        else:
            decoder_initial_state = encoder_state

        # Add Helper and ProjectionLayer and run Decoder LSTM
        projection_layer = layers_core.Dense(hparams.tgt_vsize, use_bias=False)
        if mode is 'TRAIN' or mode is 'EVAL':  # then decode using TrainingHelper
            helper = tf.contrib.seq2seq.TrainingHelper(decoder_emb_inp, sequence_length=target_lengths)
        elif mode is 'INFER':  # then decode using Beam Search
            helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding_decoder, tf.fill([true_batch_size], hparams.sos), hparams.eos)
        decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, decoder_initial_state, output_layer=projection_layer)
        outputs, _, self.test = tf.contrib.seq2seq.dynamic_decode(decoder, maximum_iterations=tf.reduce_max(target_lengths))
        logits = outputs.rnn_output

        if mode is 'TRAIN' or mode is 'EVAL':  # then calculate loss
            crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_out, logits=logits)
            target_weights = tf.sequence_mask(target_lengths, maxlen=tf.shape(target_out)[1], dtype=logits.dtype)
            self.loss = tf.reduce_sum((crossent * target_weights * weights)) / tf.cast(true_batch_size, tf.float32)

        if mode is 'TRAIN':  # then calculate/clip gradients, then optimize model
            params = tf.trainable_variables()
            gradients = tf.gradients(self.loss, params)
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, hparams.max_gradient_norm)

            optimizer = tf.train.AdamOptimizer(hparams.l_rate)
            self.update_step = optimizer.apply_gradients(zip(clipped_gradients, params))

        if mode is 'EVAL' or mode is 'INFER':  # then allow access to input/output tensors to printout
            self.src = source
            self.tgt = target_out
            self.preds = tf.argmax(logits, axis=2)

        # Designate a saver operation
        self.saver = tf.train.Saver(tf.global_variables())

    def train(self, sess):
        return sess.run([self.update_step, self.loss])

    def eval(self, sess):
        return sess.run([self.loss, self.src, self.tgt, self.preds])

    def infer(self, sess):
        return sess.run([self.src, self.tgt, self.preds])  # tgt should not exist (temporary debugging only)

