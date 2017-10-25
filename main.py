import tensorflow as tf

# Create inputs as (single batch of) tensor constants
sample_input = tf.constant([[1, 2, 3]], dtype=tf.float32)

# Set cell size (size of hidden layer)
LSTM_CELL_SIZE = 2

# Initialize lstm_cell object
lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(LSTM_CELL_SIZE, state_is_tuple=True)

# Setup state as a tuple of states for each hidden layer
state = (tf.zeros([1, LSTM_CELL_SIZE]),)*2

# Run lstm_cell with inputs (above) and update state
output, state_new = lstm_cell(sample_input, state)

# Run graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(output))
