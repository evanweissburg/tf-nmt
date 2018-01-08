import tensorflow as tf
import os
import utils

DEFAULT_BUCKET_WIDTH = 10

def get_batched_iterator(batch_size, start_token, end_token, src_padding, tgt_padding, shuffle_seed, buffer_size, buckets: int, src_max_len):
    if not (os.path.exists('primary.csv') and os.path.exists('secondary.csv')):
        utils.integerize_raw_data()

    source_dataset = tf.data.TextLineDataset('primary.csv')
    target_dataset = tf.data.TextLineDataset('secondary.csv')
    dataset = tf.data.Dataset.zip((source_dataset, target_dataset))
    dataset = dataset.shuffle(buffer_size, seed=shuffle_seed)

    dataset = dataset.map(lambda source, target: (tf.string_to_number(tf.string_split([source], delimiter=',').values, tf.int32),
                                                  tf.string_to_number(tf.string_split([target], delimiter=',').values, tf.int32)))
    dataset = dataset.map(lambda source, target: (source, tf.concat(([start_token], target), axis=0), tf.concat((target, [end_token]), axis=0)))
    dataset = dataset.map(lambda source, target_in, target_out: (source, target_in, target_out, tf.size(source), tf.size(target_in)))

    def batch(x):
        return x.padded_batch(batch_size,
                              padded_shapes=(tf.TensorShape([None]), tf.TensorShape([None]), tf.TensorShape([None]), tf.TensorShape([]), tf.TensorShape([])),  # targets, lengths
                              padding_values=(src_padding, tgt_padding, tgt_padding, 0, 0)) # pad with 0, do not pad int, pad with 0, pad with 0, do not pad int

    if buckets > 1:
        def key_func(unused_1, unused_2, unused_3, src_len, tgt_len):
            if src_max_len:
               bucket_width = (src_max_len + buckets - 1) // buckets  # if max src length is known, width of buckets will be equal to (max src len / num buckets) + 1
            else:
               bucket_width = DEFAULT_BUCKET_WIDTH

            bucket_id = tf.maximum(src_len // bucket_width, tgt_len // bucket_width)
            return tf.cast(tf.minimum(buckets, bucket_id), tf.int64)  # all extra long src go to last bucket

        def reduce_func(unused, windowed_data):
            return batch(windowed_data)

        batched_dataset = dataset.apply(tf.contrib.data.group_by_window(key_func=key_func, reduce_func=reduce_func, window_size=batch_size))

    else:
        batched_dataset = batch(dataset)

    return batched_dataset.make_initializable_iterator()
