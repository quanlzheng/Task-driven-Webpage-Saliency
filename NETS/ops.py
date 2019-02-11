import tensorflow as tf
from math import ceil
import numpy as np
import logging
import tensorflow.contrib.slim as slim


def _conv_layer(bottom, out_dim, name, stride=1, k_h=3, k_w=3):
    with tf.variable_scope(name) as scope:
        shape = [k_h, k_w, bottom.get_shape()[-1], out_dim]

        filt = tf.get_variable(name="weight", initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                               shape=shape)
        conv = tf.nn.conv2d(bottom, filt, [1, stride, stride, 1], padding='SAME')

        conv_biases = tf.get_variable(name="biases", initializer=tf.constant_initializer(0.0), shape=[out_dim])
        bias = tf.nn.bias_add(conv, conv_biases)

        relu = tf.nn.relu(bias)
        # Add summary to Tensorboard

        return relu



def _deconv_layer(inputT, shape, input_dim, output_dim, name, stride=2, ksize=4,reuse=False):
    with tf.variable_scope(name) as scope:
        in_features = inputT.get_shape()[3].value
        # if in_features is None:
        #     in_features = 1
        f_shape = [ksize, ksize, output_dim, input_dim]
        if shape is None:
            # Compute shape out of Bottom
            in_shape = tf.shape(inputT)

            h = ((in_shape[1] - 1) * stride) + 1
            w = ((in_shape[2] - 1) * stride) + 1
            new_shape = [in_shape[0], h, w, output_dim]
        else:
            new_shape = [shape[0], shape[1], shape[2], output_dim]
        output_shape = tf.stack(new_shape)
        # output_shape = new_shape
        print('Layer name: %s' % name)

        strides = [1, stride, stride, 1]

        try:
            weights = get_deconv_filter(f_shape)
        except ValueError:
            scope.reuse_variables()
            weights = get_deconv_filter(f_shape)
        deconv = tf.nn.conv2d_transpose(inputT, weights, output_shape,
                                        strides=strides, padding='SAME')
        _activation_summary(deconv)
    return deconv

def get_deconv_filter(f_shape):
    """
      reference: https://github.com/MarvinTeichmann/tensorflow-fcn
    """
    width = f_shape[0]
    heigh = f_shape[0]
    f = ceil(width/2.0)
    c = (2 * f - 1 - f % 2) / (2.0 * f)
    bilinear = np.zeros([f_shape[0], f_shape[1]])
    for x in range(width):
        for y in range(heigh):
            value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
            bilinear[x, y] = value
    weights = np.zeros(f_shape)
    for i in range(min(f_shape[2],f_shape[3])):
        weights[:, :, i, i] = bilinear

    init = tf.constant_initializer(value=weights,
                                   dtype=tf.float32)


    VV = tf.get_variable(name="up_filter", initializer=init,
                        shape=weights.shape)

    return VV






def _activation_summary(x):
    """Helper to create summaries for activations.

    Creates a summary that provides a histogram of activations.
    Creates a summary that measure the sparsity of activations.

    Args:
      x: Tensor
    Returns:
      nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = x.op.name
    # tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def _variable_summaries(var):
    """Attach a lot of summaries to a Tensor."""
    if not tf.get_variable_scope().reuse:
        name = var.op.name
        logging.info("Creating Summary for: %s" % name)
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar(name + '/mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
            tf.summary.scalar(name + '/sttdev', stddev)
            tf.summary.scalar(name + '/max', tf.reduce_max(var))
            tf.summary.scalar(name + '/min', tf.reduce_min(var))
            tf.summary.histogram(name, var)



def he_initializer(shape):
    nl = tf.cast(shape[1]*shape[2]*shape[0],tf.float32)

    w = tf.truncated_normal_initializer(stddev=tf.sqrt(2./nl))

    return w

def prelu(_x, scope='prelu'):
    with tf.variable_scope(name_or_scope=scope):
        alphas = tf.get_variable('alpha', _x.get_shape()[-1],
                                 initializer=tf.constant_initializer(0.0),
                                 dtype=tf.float32)
        pos = tf.nn.relu(_x)
        neg = alphas * (_x - abs(_x)) * 0.5

        return pos + neg

def batch_norm_conv2d(bottom, input_dim, out_dim, name, stride=1, k_h=3, k_w=3, train=True):

    with tf.variable_scope(name) as scope:
        shape = [k_h, k_w, input_dim, out_dim]

        filt = tf.get_variable(name="weight", initializer=he_initializer(shape),
                               shape=shape)
        conv = tf.nn.conv2d(bottom, filt, [1, stride, stride, 1], padding='SAME')

        conv_biases = tf.get_variable(name="biases", initializer=tf.constant_initializer(0.0), shape=[out_dim])
        bias = tf.nn.bias_add(conv, conv_biases)


        # Add summary to Tensorboard
        # batch_norm_params = {'is_training': train, 'center': True, 'scale': True,
        #                      'updates_collections': None, 'activation_fn': tf.nn.relu}
        relu = slim.batch_norm(bias, scale=True, is_training=train, activation_fn=tf.nn.relu, scope='batch_norm')

        return relu
