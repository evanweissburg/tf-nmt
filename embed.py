import tensorflow as tf
import numpy as np
import math
import utils

vocabulary_size = 20
embedding_size = 5
batch_size = 5
num_sampled = vocabulary_size

embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))

nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

# Placeholders for inputs
train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

embed = tf.nn.embedding_lookup(embeddings, train_inputs)

# Compute the NCE loss, using a sample of the negative labels each time.
loss = tf.reduce_mean(
    tf.nn.nce_loss(weights=nce_weights,
                   biases=nce_biases,
                   labels=train_labels,
                   inputs=embed,
                   num_sampled=num_sampled,
                   num_classes=vocabulary_size))

# We use the SGD optimizer.
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0).minimize(loss)

primary, secondary = utils.read_integerized_input()


def generate_batch(batch_size, sequences, window_size=5):
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    for i in range(batch_size):
        sequence = sequences[np.random.randint(0, len(sequences))]
        data_index = np.random.randint(3, len(sequence))
        compare_index = np.random.randint(int(data_index-window_size/2), int(data_index+window_size/2))
        print(sequence[data_index])

    return batch, labels


with tf.Session() as sess:
    for inputs, labels in generate_batch(batch_size, primary):
        feed_dict = {train_inputs: inputs, train_labels: labels}
        _, cur_loss = sess.run([optimizer, loss], feed_dict=feed_dict)
