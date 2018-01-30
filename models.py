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
            if mode is 'INFER' and hparams.beam_search:
                encoder_outputs = tf.contrib.seq2seq.tile_batch(encoder_outputs, multiplier=hparams.beam_width)
                source_lengths = tf.contrib.seq2seq.tile_batch(source_lengths, multiplier=hparams.beam_width)
                encoder_state = tf.contrib.seq2seq.tile_batch(encoder_state, multiplier=hparams.beam_width)
                batch_size = true_batch_size * hparams.beam_width
            else:
                batch_size = true_batch_size
            attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units=hparams.num_units,
                                                                    memory=encoder_outputs,
                                                                    memory_sequence_length=source_lengths,
                                                                    scale=True)
            decoder_cell = tf.contrib.seq2seq.AttentionWrapper(cell=decoder_cell,
                                                               attention_mechanism=attention_mechanism,
                                                               attention_layer_size=hparams.num_units)
            decoder_initial_state = decoder_cell.zero_state(batch_size, tf.float32).clone(cell_state=encoder_state)
        else:
            decoder_initial_state = encoder_state

        # Add Helper (if needed) and ProjectionLayer and run Decoder LSTM
        projection_layer = layers_core.Dense(units=hparams.tgt_vsize, use_bias=False)

        if mode is 'TRAIN' or mode is 'EVAL':
            helper = tf.contrib.seq2seq.TrainingHelper(inputs=decoder_emb_inp,
                                                       sequence_length=target_lengths)
        elif mode is 'INFER' and not hparams.beam_search:
            helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding=embedding_decoder,
                                                              start_tokens=tf.fill([true_batch_size], hparams.sos),
                                                              end_token=hparams.eos)

        if mode is 'TRAIN' or mode is 'EVAL' or (mode is 'INFER' and not hparams.beam_search):
            decoder = tf.contrib.seq2seq.BasicDecoder(cell=decoder_cell,
                                                      helper=helper,
                                                      initial_state=decoder_initial_state,
                                                      output_layer=projection_layer)

        elif mode is 'INFER' and hparams.beam_search:
            decoder = tf.contrib.seq2seq.BeamSearchDecoder(cell=decoder_cell,
                                                           embedding=embedding_decoder,
                                                           start_tokens=tf.fill([true_batch_size], hparams.sos),
                                                           end_token=hparams.eos,
                                                           initial_state=decoder_initial_state,
                                                           beam_width=hparams.beam_width,
                                                           output_layer=projection_layer,
                                                           length_penalty_weight=hparams.length_penalty_weight)

        outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, maximum_iterations=tf.reduce_max(target_lengths))
        logits = outputs.rnn_output if mode is not 'INFER' or not hparams.beam_search else tf.no_op()
        ids = outputs.sample_id if mode is not 'INFER' or not hparams.beam_search else outputs.predicted_ids

        # Calculate loss
        if mode is 'TRAIN' or mode is 'EVAL':
            crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_out, logits=logits)
            target_weights = tf.sequence_mask(target_lengths, maxlen=tf.shape(target_out)[1], dtype=logits.dtype)
            masked_loss = crossent * target_weights * weights
            self.loss = tf.reduce_sum(masked_loss) / tf.cast(true_batch_size, tf.float32)
            self.char_loss = tf.reduce_sum(tf.reduce_sum(masked_loss, axis=1) / tf.cast(target_lengths, tf.float32)) / tf.cast(true_batch_size, tf.float32)

        # Calculate/clip gradients, then optimize model
        if mode is 'TRAIN':
            params = tf.trainable_variables()
            gradients = tf.gradients(self.loss, params)
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, hparams.max_gradient_norm)

            optimizer = tf.train.AdamOptimizer(hparams.l_rate)
            self.update_step = optimizer.apply_gradients(zip(clipped_gradients, params))

        # Allow access to input/output tensors to printout
        if mode is 'EVAL' or mode is 'INFER':
            self.src = source
            self.tgt = target_out
            self.ids = ids

        self.saver = tf.train.Saver(tf.global_variables())

    def train(self, sess):
        return sess.run([self.update_step, self.char_loss])

    def eval(self, sess):
        return sess.run([self.char_loss, self.src, self.tgt, self.ids])

    def infer(self, sess):
        return sess.run([self.src, self.tgt, self.ids])


