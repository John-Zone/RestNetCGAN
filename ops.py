from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import embedding_ops

import tensorflow.contrib.slim as slim
import pdb


def conv1d(batch_input, kernel=3, output_channel=64, stride=1, use_bias=False, scope='conv1d'):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        if use_bias:
            return slim.conv1d(batch_input, output_channel, kernel, stride, 'SAME', data_format='NHWC',
                               activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer())
        else:
            return slim.conv1d(batch_input, output_channel, kernel, stride, 'SAME', data_format='NHWC',
                               activation_fn=None)


# Define the convolution block
def conv2d(batch_input, kernel=3, output_channel=64, stride=1, use_bias=False, scope='conv'):
    # kernel: An integer specifying the width and height of the 2D convolution window
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        if use_bias:
            return slim.conv2d(batch_input, output_channel, [kernel, kernel], stride, 'SAME', data_format='NHWC',
                               activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer())
        else:
            return slim.conv2d(batch_input, output_channel, [kernel, kernel], stride, 'SAME', data_format='NHWC',
                               activation_fn=None)


# define the transpose convolutional layer
def deconv2d(input, output_channel, kernel, stride, use_bias=False, scope='deconv'):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        if use_bias:
            return slim.conv2d_transpose(input, output_channel, kernel, stride, 'VALID', activation_fn=None,
                                         weights_initializer=tf.contrib.layers.xavier_initializer())
        else:
            return slim.conv2d_transpose(input, output_channel, kernel, stride, 'VALID', activation_fn=None)


def conv3d(batch_input, depth, height, width, output_channel, stride, use_bias=False, scope='cov3d'):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        if use_bias:
            return slim.conv3d(batch_input, output_channel, [depth, height, width], stride, 'SAME',
                               data_format="NDHWC",
                               activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer())
        else:
            return slim.conv3d(batch_input, output_channel, [depth, height, width], stride, 'SAME',
                               activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer(),
                               data_format="NDHWC",
                               biases_initializer=None)


def maxpool3d(batch_input, depth, height, width, scope='maxpool3d'):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        return slim.max_pool3d(batch_input, [depth, height, width], 2, 'SAME')


# Define our tensorflow version PRelu
def prelu_tf(inputs, name='Prelu'):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        alphas = tf.get_variable('alpha', inputs.get_shape()[-1], initializer=tf.zeros_initializer(), dtype=tf.float32)
    pos = tf.nn.relu(inputs)
    neg = alphas * (inputs - abs(inputs)) * 0.5

    return pos + neg


# Define our leaky_relu
def lrelu(inputs, alpha):
    return tf.nn.leaky_relu(inputs, alpha=alpha)


# Define the bathnorm layers
def batchnorm(inputs, is_training):
    return slim.batch_norm(inputs, decay=0.9, epsilon=0.001, updates_collections=tf.GraphKeys.UPDATE_OPS,
                           scale=False, fused=True, is_training=is_training)


# Our dense layer
def denselayer(inputs, output_size):
    output = tf.layers.dense(inputs, output_size, activation=None,
                             kernel_initializer=tf.contrib.layers.xavier_initializer())
    return output

# The operation used to print out the configuration
def print_configuration_op(FLAGS):
    print('[Configurations]:')
    # pdb.set_trace()
    for name, value in FLAGS.__flags.items():
        if type(value) == float:
            print('\t%s: %f' % (name, value))
        elif type(value) == int:
            print('\t%s: %d' % (name, value))
        elif type(value) == str:
            print('\t%s: %s' % (name, value))
        elif type(value) == bool:
            print('\t%s: %s' % (name, value))
        else:
            print('\t%s: %s' % (name, value))

    print('End of configuration')
