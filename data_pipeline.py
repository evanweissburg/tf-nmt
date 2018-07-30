import tensorflow as tf

from tensorflow.python.ops import lookup_ops


def make_vocab_tables(src_vocab_loc, tgt_vocab_loc):
    src_vocab_table = lookup_ops.index_table_from_file(src_vocab_loc)
    tgt_vocab_table = lookup_ops.index_table_from_file(tgt_vocab_loc)
    return src_vocab_table, tgt_vocab_table


def get_infer_iterator(hparams, src_data, src_vocab_table):
    src_eos_id = tf.cast(src_vocab_table.lookup(tf.constant(hparams.src_eos)), tf.int32)

    src_data = src_data.map(lambda src: tf.string_split([src], delimiter=',').values)

    src_data = src_data.map(lambda src: tf.cast(src_vocab_table.lookup(src), tf.int32))

    src_data = src_data.map(lambda src: (src, tf.size(src)))

    def batching_func(x):
        return x.padded_batch(
            batch_size=1,
            padded_shapes=(
                tf.TensorShape([None]),
                tf.TensorShape([])),
            padding_values=(
                src_eos_id,
                0))

    batched_data = batching_func(src_data)

    return batched_data.make_initializable_iterator()


def get_iterator(hparams, src_data, tgt_data, weight_data, src_vocab_table, tgt_vocab_table, stitching):
    src_eos_id = tf.cast(src_vocab_table.lookup(tf.constant(hparams.src_eos)), tf.int32)
    tgt_sos_id = tf.cast(tgt_vocab_table.lookup(tf.constant(hparams.tgt_sos)), tf.int32)
    tgt_eos_id = tf.cast(tgt_vocab_table.lookup(tf.constant(hparams.tgt_eos)), tf.int32)

    dataset = tf.data.Dataset.zip((src_data, tgt_data, weight_data))

    dataset = dataset.shuffle(hparams.shuffle_buffer_size, reshuffle_each_iteration=True)

    dataset = dataset.map(lambda src, tgt, weights:
                          (tf.string_split([src], delimiter=',').values,
                           tf.string_split([tgt], delimiter=',').values,
                           tf.string_to_number(tf.string_split([weights], delimiter=',').values)))

    dataset = dataset.map(lambda src, tgt, weights:
                          (tf.cast(src_vocab_table.lookup(src), tf.int32),
                           tf.cast(tgt_vocab_table.lookup(tgt), tf.int32),
                           tf.cast(weights, tf.float32)))

    def fragment(src, tgt, weights):
        src = tf.ones([tf.shape(src)[0], 1], tf.int32) * src
        tgt = tf.ones([tf.shape(src)[0], 1], tf.int32) * tgt
        weights = tf.ones([tf.shape(src)[0], 1], tf.float32) * weights

        matrix = tf.ones_like(src, tf.int32)
        matrix = tf.matrix_band_part(matrix, hparams.fragment_radius-1, hparams.fragment_radius-1)

        src *= matrix
        tgt *= matrix
        weights *= tf.cast(matrix, tf.float32)

        return tf.data.Dataset.from_tensor_slices((src, tgt, weights))

    if stitching:
        dataset = dataset.take(1).flat_map(fragment)
        dataset = dataset.map(lambda src, tgt, weights:
                              (tf.transpose(tf.gather(src, tf.where(tf.not_equal(src, 0))))[0],
                               tf.transpose(tf.gather(tgt, tf.where(tf.not_equal(src, 0))))[0],
                               tf.transpose(tf.gather(weights, tf.where(tf.not_equal(src, 0))))[0]))

    dataset = dataset.map(lambda src, tgt, weights:
                          (tf.concat((src, [src_eos_id]), axis=0),
                           tf.concat(([tgt_sos_id], tgt), axis=0),
                           tf.concat((tgt, [tgt_eos_id]), axis=0),
                           tf.concat((weights, [1.0]), axis=0)))

    dataset = dataset.map(lambda source, target_in, target_out, weights:
                          (source, target_in, target_out, weights,
                           tf.size(source), tf.size(target_in)))

    def batch(x):
        return x.padded_batch(hparams.batch_size,
                              padded_shapes=(tf.TensorShape([None]), tf.TensorShape([None]), tf.TensorShape([None]), tf.TensorShape([None]), tf.TensorShape([]), tf.TensorShape([])),
                              padding_values=(src_eos_id, tgt_eos_id, tgt_eos_id, 0.0, 0, 0))

    if hparams.num_buckets > 1:
        def key_func(unused_1, unused_2, unused_3, unused_4, src_len, tgt_len):
            bucket_width = (hparams.max_len + hparams.num_buckets - 1) // hparams.num_buckets
            bucket_id = tf.maximum(src_len // bucket_width, tgt_len // bucket_width)
            return tf.cast(tf.minimum(hparams.num_buckets, bucket_id), tf.int64)  # all extra long src go to last bucket

        def reduce_func(unused, windowed_data):
            return batch(windowed_data)

        batched_data = dataset.apply(tf.contrib.data.group_by_window(key_func=key_func, reduce_func=reduce_func, window_size=hparams.batch_size))

    else:
        batched_data = batch(dataset)

    return batched_data.make_initializable_iterator()
