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


def generate_batch(sequence, batch_size, window_radius=1, repetition=1):
    index = 0
    print(batch_size                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            )
    assert batch_size % repetition == 0, "Batch size must be divisible by repetition coefficient (in this implementation)!"
    for i in range(batch_size // repetition):
        span = sequence[max(index-window_radius, 0):max(index+window_radius-1, len(sequence))]
        print(span)
    # This function needs to be finished


generate_batch(primary[0], 10)

with tf.Session() as sess:
    for inputs, labels in generate_batch(batch_size, primary):
        feed_dict = {train_inputs: inputs, train_labels: labels}
        _, cur_loss = sess.run([optimizer, loss], feed_dict=feed_dict)
