import tensorflow as tf
import math
import random
import numpy as np


def generate_embeddings(sequences, batch_size, epochs, embedding_size, learning_rate, vocabulary_size, window_radius, repetition):
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
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

    def generate_batch(sequences, batch_size, window_radius, repetition):
        batch = list()
        labels = list()
        assert window_radius >= repetition, 'Window radius must not be less than repetition coefficient!'
        assert batch_size % repetition == 0, 'Batch size must be divisible by repetition!'

        for sequence in sequences:
            for index in range(len(sequence)):
                samples = random.sample([k-window_radius for k in range(window_radius * 2 + 1)
                                        if k-window_radius != 0 and 0 <= index + k-window_radius < len(sequence)],
                                        repetition if repetition < len(sequence) else len(sequence))
                for j in samples:
                    labels.append(sequence[index])
                    batch.append(sequence[index + j])
                    if len(batch) == batch_size:
                        return np.asarray(batch), np.asarray(labels).reshape([-1, 1])
        return None

    with tf.Session() as sess:
        print('Generating embeddings...')
        sess.run(tf.global_variables_initializer())
        for epoch in range(epochs):
            inputs, labels = generate_batch(sequences, batch_size, window_radius, repetition)
            feed_dict = {train_inputs: inputs, train_labels: labels}
            _, cur_loss = sess.run([optimizer, loss], feed_dict=feed_dict)
        print('Embeddings complete with %s loss.' % cur_loss)
