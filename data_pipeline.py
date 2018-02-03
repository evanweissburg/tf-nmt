import tensorflow as tf


def get_batched_iterator(hparams, src_loc, tgt_loc, weights_loc):
    source_dataset = tf.data.TextLineDataset(src_loc)
    target_dataset = tf.data.TextLineDataset(tgt_loc)
    weights_dataset = tf.data.TextLineDataset(weights_loc)
    dataset = tf.data.Dataset.zip((source_dataset, target_dataset, weights_dataset))
    dataset = dataset.shuffle(hparams.shuffle_buffer_size, hparams.shuffle_seed, reshuffle_each_iteration=True)

    dataset = dataset.map(lambda source, target, weights:
                          (tf.string_to_number(tf.string_split([source], delimiter=',').values, tf.int32),
                           tf.string_to_number(tf.string_split([target], delimiter=',').values, tf.int32),
                           tf.string_to_number(tf.string_split([weights], delimiter=',').values, tf.float32)))
    dataset = dataset.map(lambda source, target, weights:
                          (source,
                           tf.concat(([hparams.sos], target), axis=0),
                           tf.concat((target, [hparams.eos]), axis=0),
                           tf.concat((weights, [1.0]), axis=0)))
    dataset = dataset.map(lambda source, target_in, target_out, weights: (source, target_in, target_out, weights, tf.size(source), tf.size(target_in)))

    def batch(x):
        return x.padded_batch(hparams.batch_size,
                              padded_shapes=(tf.TensorShape([None]), tf.TensorShape([None]), tf.TensorShape([None]), tf.TensorShape([None]), tf.TensorShape([]), tf.TensorShape([])),
                              padding_values=(hparams.src_pad, hparams.tgt_pad, hparams.tgt_pad, 0.0, 0, 0))

    if hparams.num_buckets > 1:
        def key_func(unused_1, unused_2, unused_3, unused_4, src_len, tgt_len):
            bucket_width = (hparams.max_len + hparams.num_buckets - 1) // hparams.num_buckets
            bucket_id = tf.maximum(src_len // bucket_width, tgt_len // bucket_width)
            return tf.cast(tf.minimum(hparams.num_buckets, bucket_id), tf.int64)  # all extra long src go to last bucket

        def reduce_func(unused, windowed_data):
            return batch(windowed_data)

        batched_dataset = dataset.apply(tf.contrib.data.group_by_window(key_func=key_func, reduce_func=reduce_func, window_size=hparams.batch_size))

    else:
        batched_dataset = batch(dataset)

    return batched_dataset.make_initializable_iterator()
