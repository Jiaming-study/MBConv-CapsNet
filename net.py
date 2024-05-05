from __future__ import division, print_function, unicode_literals
import tensorflow as tf
from module.SKAttention import SKAttention
from module.MBConv import MBConvBlock

from utils import _conv2d_wrapper
from layer import capsules_init, capsule_flatten, capsule_conv_layer, capsule_fc_layer,
import tensorflow.contrib.slim as slim

def baseline_model_cnn(X, num_classes):
    nets = _conv2d_wrapper(
        X, shape=[3, 300, 1, 32], strides=[1, 1, 1, 1], padding='VALID',
        add_bias=False, activation_fn=tf.nn.relu, name='conv1'
        )
    nets = slim.flatten(nets)
    tf.logging.info('flatten shape: {}'.format(nets.get_shape()))
    nets = slim.fully_connected(nets, 128, scope='relu_fc3', activation_fn=tf.nn.relu)
    tf.logging.info('fc shape: {}'.format(nets.get_shape()))

    activations = tf.sigmoid(slim.fully_connected(nets, num_classes, scope='final_layer', activation_fn=None))
    tf.logging.info('fc shape: {}'.format(activations.get_shape()))
    return tf.zeros([0]), activations

def baseline_model_kimcnn(X, max_sent, num_classes):
    pooled_outputs = []
    for i, filter_size in enumerate([3,4,5]):
        with tf.name_scope("conv-maxpool-%s" % filter_size):
            filter_shape = [filter_size, 300, 1, 100]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[100]), name="b")
            conv = tf.nn.conv2d(X, W, strides=[1, 1, 1, 1], padding="VALID", name="conv")
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
            pooled = tf.nn.max_pool(
                h,
                ksize=[1, max_sent - filter_size + 1, 1, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name="pool")
            pooled_outputs.append(pooled)
    num_filters_total = 100 * 3
    h_pool = tf.concat(pooled_outputs, 3)
    h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
    activations = tf.sigmoid(slim.fully_connected(h_pool_flat, num_classes, scope='final_layer', activation_fn=None))
    return tf.zeros([0]), activations
def clstm_model(X, max_sent, num_classes, lstm_units=64):
    # Convolutional layers
    pooled_outputs = []
    for i, filter_size in enumerate([3, 4, 5]):
        with tf.name_scope("conv-maxpool-%s" % filter_size):
            filter_shape = [filter_size, 300, 1, 100]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[100]), name="b")
            conv = tf.nn.conv2d(X, W, strides=[1, 1, 1, 1], padding="VALID", name="conv")
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
            pooled = tf.nn.max_pool(
                h,
                ksize=[1, max_sent - filter_size + 1, 1, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name="pool"
            )
            pooled_outputs.append(pooled)

    # LSTM layer
    with tf.name_scope("lstm_layer"):
        lstm_input = tf.concat(pooled_outputs, axis=3)
        lstm_input = tf.squeeze(lstm_input, axis=2)
        lstm_input = tf.unstack(lstm_input, axis=1)
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(lstm_units)
        lstm_outputs, _ = tf.nn.static_rnn(lstm_cell, lstm_input, dtype=tf.float32)

    # Fully connected layer
    with tf.name_scope("fully_connected_layer"):
        logits = slim.fully_connected(lstm_outputs[-1], num_classes, activation_fn=None)

    return tf.zeros([0]), logits

def capsule_model_A(X, num_classes):
    print('X', X)
    with tf.variable_scope('capsule_' + str(3)):
        nets = _conv2d_wrapper(
            X, shape=[3, 300, 1, 32], strides=[1, 1, 1, 1], padding='VALID',
            add_bias=True, activation_fn=tf.nn.relu, name='conv1'
        )
        tf.logging.info('output shape: {}'.format(nets.get_shape()))
        nets = capsules_init(nets, shape=[1, 1, 32, 16], strides=[1, 1, 1, 1],
                             padding='VALID', pose_shape=16, add_bias=True, name='primary')
        nets = capsule_conv_layer(nets, shape=[3, 1, 16, 16], strides=[1, 1, 1, 1], iterations=3, name='conv2')
        nets = capsule_flatten(nets)
        poses, activations = capsule_fc_layer(nets, num_classes, 3, 'fc2')
    return poses, activations


def capsule_model_at(X, num_classes):
    print('X', X)
    with tf.variable_scope('capsule_' + str(3)):
        nets = _conv2d_wrapper(
            X, shape=[3, 300, 1, 32], strides=[1, 1, 1, 1], padding='VALID',
            add_bias=True, activation_fn=tf.nn.relu, name='conv1'
        )
        print('nets', nets)
        tf.logging.info('output shape: {}'.format(nets.get_shape()))
        nets = SKAttention(channel=32 , G=8)(nets)
        nets = capsules_init(nets, shape=[1, 1, 32, 16], strides=[1, 1, 1, 1],
                             padding='VALID', pose_shape=16, add_bias=True, name='primary')
        print('nets', nets)
        nets = capsule_conv_layer(nets, shape=[3, 1, 16, 16], strides=[1, 1, 1, 1], iterations=3, name='conv2')
        print('nets', nets)
        nets = capsule_flatten(nets)
        print('nets', nets)
        poses, activations = capsule_fc_layer(nets, num_classes, 3, 'fc2')
    return poses, activations
def capsule_model_conv(X, num_classes):
    print('X', X)
    X = MBConvBlock(ksize=3, input_filters=1, output_filters=1)(X)
    print('X', X)
    with tf.variable_scope('capsule_' + str(3)):
        nets = _conv2d_wrapper(
            X, shape=[3, 300, 1, 32], strides=[1, 1, 1, 1], padding='VALID',
            add_bias=True, activation_fn=tf.nn.relu, name='conv1'
        )
        tf.logging.info('output shape: {}'.format(nets.get_shape()))
        nets = capsules_init(nets, shape=[1, 1, 32, 16], strides=[1, 1, 1, 1],
                             padding='VALID', pose_shape=16, add_bias=True, name='primary')
        nets = capsule_conv_layer(nets, shape=[3, 1, 16, 16], strides=[1, 1, 1, 1], iterations=3, name='conv2')
        nets = capsule_flatten(nets)
        poses, activations = capsule_fc_layer(nets, num_classes, 3, 'fc2')
    return poses, activations

def capsule_model_ALL(X, num_classes):
    print('X', X)
    X = MBConvBlock(ksize=3, input_filters=1, output_filters=1)(X)
    with tf.variable_scope('capsule_' + str(3)):
        nets = _conv2d_wrapper(
            X, shape=[3, 300, 1, 32], strides=[1, 1, 1, 1], padding='VALID',
            add_bias=True, activation_fn=tf.nn.relu, name='conv1'
        )
        nets = SKAttention(channel=32, reduction=8)(nets)
        tf.logging.info('output shape: {}'.format(nets.get_shape()))

        nets = capsules_init(nets, shape=[1, 1, 32, 16], strides=[1, 1, 1, 1],
                             padding='VALID', pose_shape=16, add_bias=True, name='primary')
        nets = capsule_conv_layer(nets, shape=[3, 1, 16, 16], strides=[1, 1, 1, 1], iterations=3, name='conv2')
        nets = capsule_flatten(nets)
        poses, activations = capsule_fc_layer(nets, num_classes, iterations=3, name='fc2')
    return poses, activations

