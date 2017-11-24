import tensorflow as tf
import math
import random
import numpy as np
import utils


def generate_embeddings(sequences, batch_size, epochs, embedding_size, learning_rate, vocabulary_size, window_radius, repetition):
    embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

    embed = tf.nn.embedding_lookup(embeddings, train_inputs)

    loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights,
                                         biases=nce_biases,
                                         labels=train_labels,
                                         inputs=embed,
                                         num_sampled=vocabulary_size,
                                         num_classes=vocabulary_size))

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
            _, curr_loss = sess.run([optimizer, loss], feed_dict={train_inputs: inputs, train_labels: labels})
        final_embeddings = embeddings.eval()
        print('Embeddings complete with %s loss.' % curr_loss)
    print(final_embeddings)

    # Code to create the embedding images
    # Sourced from github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/examples/tutorials/word2vec/word2vec_basic.py)

    # Function to draw visualization of distance between embeddings.
    def plot_with_labels(low_dim_embs, labels, filename):
        assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
        plt.figure(figsize=(18, 18))  # in inches
        for i, label in enumerate(labels):
            x, y = low_dim_embs[i, :]
            plt.scatter(x, y)
            plt.annotate(label,
                         xy=(x, y),
                         xytext=(5, 2),
                         textcoords='offset points',
                         ha='right',
                         va='bottom')

        plt.savefig(filename)

    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    tsne = TSNE(perplexity=embedding_size/2, n_components=2, init='pca', n_iter=5000, method='exact')
    low_dim_embs = tsne.fit_transform(final_embeddings)
    labels = [utils.integer_to_fasta(i) for i in range(vocabulary_size)]
    plot_with_labels(low_dim_embs, labels, 'tsne' + str(vocabulary_size) + '.png')
