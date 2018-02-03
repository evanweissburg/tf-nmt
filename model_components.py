import tensorflow as tf
from tensorflow.python.layers import core as layers_core


def _make_rnn_block(num_units, num_layers):
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


def make_and_run_encoder(hparams, encoder_emb_inp, source_lengths):
    """Build and run an Encoder cell.
    :return: encoder_outputs, encoder_state"""

    if hparams.bidir_encoder:
        forward_cell = _make_rnn_block(num_units=hparams.num_units, num_layers=hparams.num_layers/2)
        backward_cell = _make_rnn_block(num_units=hparams.num_units, num_layers=hparams.num_layers/2)
        bi_outputs, encoder_state = tf.nn.bidirectional_dynamic_rnn(
            forward_cell, backward_cell, encoder_emb_inp,
            sequence_length=source_lengths, dtype=tf.float32)
        encoder_outputs = tf.concat(bi_outputs, -1)
    else:
        encoder_cell = _make_rnn_block(num_units=hparams.num_units, num_layers=hparams.num_layers)
        encoder_outputs, encoder_state = tf.nn.dynamic_rnn(encoder_cell, encoder_emb_inp, sequence_length=source_lengths, dtype=tf.float32)

    return encoder_outputs, encoder_state


def make_decoder(hparams, mode, encoder_outputs, encoder_state, source_lengths, batch_size):
    """Build a Decoder cell and get its initial memory state.
    :return: decoder_cell, decoder_initial_state"""

    decoder_cell = _make_rnn_block(hparams.num_units, hparams.num_layers)

    if hparams.attention:
        if mode is 'INFER' and hparams.beam_search:
            encoder_outputs = tf.contrib.seq2seq.tile_batch(encoder_outputs, multiplier=hparams.beam_width)
            encoder_state = tf.contrib.seq2seq.tile_batch(encoder_state, multiplier=hparams.beam_width)
            source_lengths = tf.contrib.seq2seq.tile_batch(source_lengths, multiplier=hparams.beam_width)
            batch_size *= hparams.beam_width

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

    return decoder_cell, decoder_initial_state


def construct_decoder(hparams, mode, decoder_cell, decoder_state, decoder_emb_inp, target_lengths, embedding_decoder, batch_size):
    """Finish the Decoder by selecting a decoding-style and adding a projection layer.
    :return: decoder"""

    projection_layer = layers_core.Dense(units=hparams.tgt_vsize, use_bias=False)

    if mode is 'TRAIN' or mode is 'EVAL':
        helper = tf.contrib.seq2seq.TrainingHelper(inputs=decoder_emb_inp,
                                                   sequence_length=target_lengths)
    elif mode is 'INFER' and not hparams.beam_search:
        helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding=embedding_decoder,
                                                          start_tokens=tf.fill([batch_size], hparams.sos),
                                                          end_token=hparams.eos)

    if mode is 'TRAIN' or mode is 'EVAL' or (mode is 'INFER' and not hparams.beam_search):
        decoder = tf.contrib.seq2seq.BasicDecoder(cell=decoder_cell,
                                                  helper=helper,
                                                  initial_state=decoder_state,
                                                  output_layer=projection_layer)

    elif mode is 'INFER' and hparams.beam_search:
        decoder = tf.contrib.seq2seq.BeamSearchDecoder(cell=decoder_cell,
                                                       embedding=embedding_decoder,
                                                       start_tokens=tf.fill([batch_size], hparams.sos),
                                                       end_token=hparams.eos,
                                                       initial_state=decoder_state,
                                                       beam_width=hparams.beam_width,
                                                       output_layer=projection_layer,
                                                       length_penalty_weight=hparams.length_penalty_weight)

    return decoder


def run_decoder(hparams, mode, decoder, target_lengths):
    """Run the decoder cell.
    :return: logits, ids"""

    outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, maximum_iterations=tf.reduce_max(target_lengths))
    logits = outputs.rnn_output if mode is not 'INFER' or not hparams.beam_search else tf.no_op()
    ids = outputs.sample_id if mode is not 'INFER' or not hparams.beam_search else outputs.predicted_ids

    return logits, ids


def calculate_loss(logits, target_out, target_lengths, key_weights, batch_size):
    """Calculate Softmaxed Cross-Entropy loss; additionally find character-wise loss.
    :return: loss, characterwise_loss"""

    crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_out, logits=logits)
    target_weights = tf.sequence_mask(target_lengths, maxlen=tf.shape(target_out)[1], dtype=logits.dtype)
    masked_loss = crossent * target_weights * key_weights
    loss = tf.reduce_sum(masked_loss) / tf.cast(batch_size, tf.float32)
    char_loss = tf.reduce_sum(tf.reduce_sum(masked_loss, axis=1) / tf.cast(target_lengths, tf.float32)) / tf.cast(batch_size, tf.float32)

    return loss, char_loss


def optimize_model(hparams, loss, global_step):
    """Calculate gradients, clip gradients, apply gradients.
    :return: update_step"""

    params = tf.trainable_variables()
    gradients = tf.gradients(loss, params)
    clipped_gradients, _ = tf.clip_by_global_norm(gradients, hparams.max_gradient_norm)

    optimizer = tf.train.AdamOptimizer(hparams.l_rate)
    update_step = optimizer.apply_gradients(zip(clipped_gradients, params), global_step=global_step)

    return update_step


