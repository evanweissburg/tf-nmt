import tensorflow as tf
import os
import utils


def get_batched_iterator(batch_size, start_token, end_token, src_padding, tgt_padding, shuffle_seed, buffer_size):
    if not (os.path.exists('primary.csv') and os.path.exists('secondary.csv')):
        utils.integerize_raw_data()

    source_dataset = tf.data.TextLineDataset('primary.csv')
    target_dataset = tf.data.TextLineDataset('secondary.csv')
    dataset = tf.data.Dataset.zip((source_dataset, target_dataset))
    dataset = dataset.shuffle(buffer_size, seed=shuffle_seed)

    dataset = dataset.map(lambda source, target: (tf.string_to_number(tf.string_split([source], delimiter=',').values, tf.int32),
                                                  tf.string_to_number(tf.string_split([target], delimiter=',').values, tf.int32)))
    dataset = dataset.map(lambda source, target: (source, tf.concat(([start_token], target), axis=0), tf.concat((target, [end_token]), axis=0)))
    dataset = dataset.map(lambda source, target_in, target_out: (source, tf.size(source), target_in, target_out, tf.size(target_in)))

    batched_dataset = dataset.padded_batch(batch_size,
                                           padded_shapes=(tf.TensorShape([None]), tf.TensorShape([]), tf.TensorShape([None]), tf.TensorShape([None]), tf.TensorShape([])),  # targets, lengths
                                           padding_values=(src_padding, 0, tgt_padding, tgt_padding, 0)) # pad with 0, do not pad int, pad with 0, pad with 0, do not pad int

    return batched_dataset.make_initializable_iterator()
