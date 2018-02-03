import tensorflow as tf
import model_components as my_comps


class NMTModel:
    def __init__(self, hparams, iterator, mode):
        tf.set_random_seed(hparams.graph_seed)
        self.global_step = tf.Variable(0, trainable=False)

        source, target_in, target_out, key_weights, source_lengths, target_lengths = iterator.get_next()
        true_batch_size = tf.size(source_lengths)

        embedding_encoder = tf.get_variable("embedding_encoder", [hparams.src_vsize, hparams.src_emsize])
        encoder_emb_inp = tf.nn.embedding_lookup(embedding_encoder, source)
        embedding_decoder = tf.get_variable("embedding_decoder", [hparams.tgt_vsize, hparams.tgt_emsize])
        decoder_emb_inp = tf.nn.embedding_lookup(embedding_decoder, target_in)

        encoder_outputs, encoder_state = my_comps.make_and_run_encoder(hparams, encoder_emb_inp, source_lengths)

        decoder_cell, decoder_state = my_comps.make_decoder(hparams, mode, encoder_outputs, encoder_state,
                                                            source_lengths, true_batch_size)

        decoder = my_comps.construct_decoder(hparams, mode, decoder_cell, decoder_state,
                                             decoder_emb_inp, target_lengths, embedding_decoder, true_batch_size)

        logits, ids = my_comps.run_decoder(hparams, mode, decoder, target_lengths)

        if mode is 'TRAIN' or mode is 'EVAL':
            self.loss, self.char_loss = my_comps.calculate_loss(logits, target_out, target_lengths,
                                                                key_weights, true_batch_size)

        if mode is 'TRAIN':
            self.update_step = my_comps.optimize_model(hparams, self.loss, self.global_step)

        if mode is 'EVAL' or mode is 'INFER':
            self.src = source
            self.tgt = target_out
            self.ids = ids

        self.saver = tf.train.Saver(tf.global_variables())

    def train(self, sess):
        return sess.run([self.update_step, self.char_loss, self.global_step])

    def eval(self, sess):
        return sess.run([self.char_loss, self.src, self.tgt, self.ids])

    def infer(self, sess):
        return sess.run([self.src, self.tgt, self.ids])


