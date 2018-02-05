import tensorflow as tf
from tensorflow.python.layers import core as layers_core


class NMTModel:
    def __init__(self, hparams, iterator, mode, src_vocab_table, tgt_vocab_table, reverse_tgt_vocab_table=None):
        tf.set_random_seed(hparams.graph_seed)
        self.global_step = tf.Variable(0, trainable=False)
        self.src_vocab_table = src_vocab_table
        self.tgt_vocab_table = tgt_vocab_table
        self.reverse_tgt_vocab_table = reverse_tgt_vocab_table

        if mode is not 'PRED':
            self.src, self.tgt_in, self.tgt_out, self.key_weights, self.src_len, self.tgt_len = iterator.get_next()
        else:
            self.src, self.src_len = iterator.get_next()
            mode = 'INFER'
        self.batch_size = tf.size(self.src_len)

        self.embedding_encoder = tf.get_variable("embedding_encoder", [hparams.src_vsize, hparams.src_emsize])
        self.encoder_emb_inp = tf.nn.embedding_lookup(self.embedding_encoder, self.src)
        self.embedding_decoder = tf.get_variable("embedding_decoder", [hparams.tgt_vsize, hparams.tgt_emsize])
        if mode is not 'INFER':
            self.decoder_emb_inp = tf.nn.embedding_lookup(self.embedding_decoder, self.tgt_in)

        encoder_outputs, encoder_state = self.make_and_run_encoder(hparams)

        decoder_cell, decoder_state = self.make_decoder(hparams, mode, encoder_outputs, encoder_state)

        decoder = self.construct_decoder(hparams, mode, decoder_cell, decoder_state)

        logits, self.ids = self.run_decoder(hparams, mode, decoder)

        if mode is 'TRAIN' or mode is 'EVAL':
            loss, self.char_loss = self.calculate_loss(logits)

        if mode is 'TRAIN':
            self.update_step = self.optimize_model(hparams, loss)

        self.saver = tf.train.Saver(tf.global_variables())

    def train(self, sess):
        return sess.run([self.update_step, self.char_loss, self.global_step])

    def eval(self, sess):
        return sess.run([self.char_loss, self.src, self.tgt_out, self.ids])

    def infer(self, sess):
        return sess.run([self.src, self.tgt_out, self.ids])

    def pred(self, sess):
        dssp = self.reverse_tgt_vocab_table.lookup(tf.to_int64(self.ids))
        return sess.run(dssp).astype('U13')

    def _make_rnn_block(self, num_units, num_layers):
        """Build a block of RNNcells or a single RNNCell.
        :return: rnn_cell"""

        if num_layers > 1:
            cell_list = []
            for i in range(num_layers):
                single_cell = tf.contrib.rnn.BasicLSTMCell(num_units)
                cell_list.append(single_cell)
            return tf.contrib.rnn.MultiRNNCell(cell_list)

        else:
            return tf.contrib.rnn.BasicLSTMCell(num_units)

    def make_and_run_encoder(self, hparams):
        """Build and run an Encoder cell.
        :return: encoder_outputs, encoder_state"""

        if hparams.bidir_encoder:
            forward_cell = self._make_rnn_block(num_units=hparams.num_units, num_layers=hparams.num_layers/2)
            backward_cell = self._make_rnn_block(num_units=hparams.num_units, num_layers=hparams.num_layers/2)
            bi_outputs, encoder_state = tf.nn.bidirectional_dynamic_rnn(
                forward_cell, backward_cell, self.encoder_emb_inp,
                sequence_length=self.src_len, dtype=tf.float32)
            encoder_outputs = tf.concat(bi_outputs, -1)
        else:
            encoder_cell = self._make_rnn_block(num_units=hparams.num_units, num_layers=hparams.num_layers)
            encoder_outputs, encoder_state = tf.nn.dynamic_rnn(encoder_cell, self.encoder_emb_inp, sequence_length=self.src_len, dtype=tf.float32)

        return encoder_outputs, encoder_state

    def make_decoder(self, hparams, mode, encoder_outputs, encoder_state):
        """Build a Decoder cell and get its initial memory state.
        :return: decoder_cell, decoder_initial_state"""

        decoder_cell = self._make_rnn_block(hparams.num_units, hparams.num_layers)

        source_sequence_len = self.src_len
        batch_size = self.batch_size
        if mode is 'INFER' and hparams.beam_search:
            encoder_state = tf.contrib.seq2seq.tile_batch(encoder_state, multiplier=hparams.beam_width)
            if hparams.attention:
                encoder_outputs = tf.contrib.seq2seq.tile_batch(encoder_outputs, multiplier=hparams.beam_width)
                source_sequence_len = tf.contrib.seq2seq.tile_batch(source_sequence_len, multiplier=hparams.beam_width)
                batch_size *= hparams.beam_width

        if hparams.attention:
            attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units=hparams.num_units,
                                                                    memory=encoder_outputs,
                                                                    memory_sequence_length=source_sequence_len,
                                                                    scale=True)
            decoder_cell = tf.contrib.seq2seq.AttentionWrapper(cell=decoder_cell,
                                                               attention_mechanism=attention_mechanism,
                                                               attention_layer_size=hparams.num_units)
            decoder_initial_state = decoder_cell.zero_state(batch_size, tf.float32).clone(cell_state=encoder_state)

        else:
            decoder_initial_state = encoder_state

        return decoder_cell, decoder_initial_state

    def construct_decoder(self, hparams, mode, decoder_cell, decoder_state):
        """Finish the Decoder by selecting a decoding-style and adding a projection layer.
        :return: decoder"""

        projection_layer = layers_core.Dense(units=hparams.tgt_vsize, use_bias=False)
        tgt_sos_id = tf.cast(self.tgt_vocab_table.lookup(tf.constant(hparams.tgt_sos)), tf.int32)
        tgt_eos_id = tf.cast(self.tgt_vocab_table.lookup(tf.constant(hparams.tgt_eos)), tf.int32)

        if mode is 'TRAIN' or mode is 'EVAL' or (mode is 'INFER' and not hparams.beam_search):
            if mode is 'TRAIN' or mode is 'EVAL':
                helper = tf.contrib.seq2seq.TrainingHelper(inputs=self.decoder_emb_inp,
                                                           sequence_length=self.tgt_len)

            elif mode is 'INFER' and not hparams.beam_search:
                helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding=self.embedding_decoder,
                                                                  start_tokens=tf.fill([self.batch_size], tgt_sos_id),
                                                                  end_token=tgt_eos_id)

            decoder = tf.contrib.seq2seq.BasicDecoder(cell=decoder_cell,
                                                      helper=helper,
                                                      initial_state=decoder_state,
                                                      output_layer=projection_layer)

        else:
            decoder = tf.contrib.seq2seq.BeamSearchDecoder(cell=decoder_cell,
                                                           embedding=self.embedding_decoder,
                                                           start_tokens=tf.fill([self.batch_size], tgt_sos_id),
                                                           end_token=tgt_eos_id,
                                                           initial_state=decoder_state,
                                                           beam_width=hparams.beam_width,
                                                           output_layer=projection_layer,
                                                           length_penalty_weight=hparams.length_penalty_weight)

        return decoder

    def run_decoder(self, hparams, mode, decoder):
        """Run the decoder cell.
        :return: logits, ids"""

        outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, maximum_iterations=tf.reduce_max(self.src_len))
        logits = outputs.rnn_output if mode is not 'INFER' or not hparams.beam_search else tf.no_op()
        ids = outputs.sample_id if mode is not 'INFER' or not hparams.beam_search else outputs.predicted_ids

        return logits, ids

    def calculate_loss(self, logits):
        """Calculate Softmaxed Cross-Entropy loss; additionally find character-wise loss.
        :return: loss, characterwise_loss"""

        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.tgt_out, logits=logits)
        target_weights = tf.sequence_mask(self.tgt_len, maxlen=tf.shape(self.tgt_out)[1], dtype=logits.dtype)
        masked_loss = crossent * target_weights * self.key_weights

        loss = tf.reduce_sum(masked_loss) / tf.cast(self.batch_size, tf.float32)
        char_loss = tf.reduce_sum(tf.reduce_sum(masked_loss, axis=1) / tf.cast(self.tgt_len, tf.float32)) / tf.cast(self.batch_size, tf.float32)

        return loss, char_loss

    def optimize_model(self, hparams, loss):
        """Calculate gradients, clip gradients, apply gradients.
        :return: update_step"""

        params = tf.trainable_variables()
        gradients = tf.gradients(loss, params)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, hparams.max_gradient_norm)

        optimizer = tf.train.AdamOptimizer(hparams.l_rate)
        update_step = optimizer.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)

        return update_step


