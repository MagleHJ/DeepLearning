import tensorflow as tf

'''Hyper-parameters'''
num_steps = 5
batch_size = 200
num_classes = 2
state_size = 4
learning_rate = 0.1

'''Define RNN cell'''
with tf.variable_scope('rnn_cell'):
    W = tf.get_variable('W', [num_classes + state_size, state_size])
    b = tf.get_variable('b', [state_size], initializer=tf.constant_initializer(0.0))

'''Create RNN cell'''
def rnn_cell(rnn_input, state):
    '''
    Return: tanh([x(t), h(t-1)] * [U, W] + b)
    '''
    with tf.variable_scope('rnn_cell', reuse=True):
        W = tf.get_variable('W', [num_classes + state_size, state_size])
        b = tf.get_variable('b', [state_size],initializer=tf.constant_initializer(0.0))
    return tf.tanh(tf.matmul(tf.concat([rnn_input, state], 1), W) + b)

