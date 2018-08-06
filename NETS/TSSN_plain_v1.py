import logging

import tensorflow.contrib.slim as slim

from NETS import LLSN
from NETS import fcn8_vgg
from tools.ops import _deconv_layer, batch_norm_conv2d
from tools.utils import *


class Merge():
    def __init__(self):
        print('starting to merge')

    def Merge_layout_task_LL(self, batch_images, task, train=True):
        batch_norm_params = {'is_training': train, 'center': True, 'scale': True,
                             'updates_collections': None, 'activation_fn': tf.nn.relu}
        #
        with slim.arg_scope([slim.conv2d], activation_fn=None, stride=1, padding='SAME',
                            weights_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                            biases_initializer=tf.constant_initializer(0.0),
                            normalizer_fn=slim.batch_norm, normalizer_params=batch_norm_params):
        # #
            self.batch_images = batch_images

            self.Lgg_fcn = fcn8_vgg.FCN8VGG()
            with tf.name_scope("content_vgg"):
                self.Lgg_fcn.build(self.batch_images, num_classes=5, train=train, debug=True)

            self.mask = self.Lgg_fcn.upscore32
            # self.mask = batch_images

            
            self.mask_conv1 = slim.conv2d(self.mask, 16, [3, 3], stride=2, scope='Added_mask_conv1')

            self.mask_conv2 = slim.conv2d(self.mask_conv1, 32, [3, 3], stride=2, scope='Added_mask_conv2')

            self.mask_conv3 = slim.conv2d(self.mask_conv2, 64, [3, 3], stride=2, scope='Added_mask_conv3')
            
            # self.mask_conv1 = batch_norm_conv2d(self.mask, 5, 16, stride=2, name='Added_mask_conv1')
            #
            # self.mask_conv2 = batch_norm_conv2d(self.mask_conv1, 16, 32, stride=2, name='Added_mask_conv2')
            #
            # self.mask_conv3 = batch_norm_conv2d(self.mask_conv2, 32, 64, stride=2, name='Added_mask_conv3')


            # task bratch
            self.t_fc1 = self._fully_connected_layer(task, 'Added_t_fc1')
            if train:
                self.t_fc1 = tf.nn.dropout(self.t_fc1, 0.5)

            self.t_fc2 = self._fully_connected_layer(self.t_fc1, 'Added_t_fc2')
            if train:
                self.t_fc2 = tf.nn.dropout(self.t_fc2, 0.5)

            self.t_fc3 = self._fully_connected_layer(self.t_fc2, 'Added_t_fc3')
            if train:
                self.t_fc3 = tf.nn.dropout(self.t_fc3, 0.5)

            self.t_map = tf.reshape(self.t_fc3, [-1, 28, 28, 1])
            self.t_map_tile = tf.tile(self.t_map, [1, 1, 1, 64])
            # concatenate two nets
            self.concatLayer = tf.concat([self.mask_conv3, self.t_map_tile], axis=3, name='Added_mask_concatLayer')

            # self.concat_conv0 = batch_norm_conv2d(self.concatLayer, 128, 128, name='Added_concat_conv0')
            #
            # self.concat_conv = batch_norm_conv2d(self.concat_conv0, 128, 128, name='Added_concat_conv1')
            self.concat_conv = slim.repeat(self.concatLayer, 2, slim.conv2d, 128, [3, 3], scope='Added_concat_conv0')

            self.up_map2 = _deconv_layer(self.concat_conv, self.mask_conv2.get_shape(), 128, 32, stride=2,
                                         name='Added_mask_upconv2')
            self.up_map4 = _deconv_layer(self.up_map2, self.mask_conv1.get_shape(), 32, 16, stride=2,
                                         name='Added_mask_upconv4')
            self.up_map8 = _deconv_layer(self.up_map4, self.mask.get_shape(), 16, 1, stride=2,
                                         name='Added_mask_upcon8')
            self.map_sal = self.up_map8

    def _fully_connected_layer(self, inputs, name):
        with tf.variable_scope(name) as scope:
            if name == 'Added_t_fc1':
                W_fc1 = self.weight_variable([5, 1024])
                b_fc1 = self.bias_variable([1024])
                h_pool2_flat = tf.reshape(inputs, [-1, 5])
            elif name == 'Added_t_fc2':
                W_fc1 = self.weight_variable([1024, 1024])
                b_fc1 = self.bias_variable([1024])
                h_pool2_flat = tf.reshape(inputs, [-1, 1024])
            elif name == 'Added_t_fc3':
                W_fc1 = self.weight_variable([1024, 784])
                b_fc1 = self.bias_variable([784])
                h_pool2_flat = tf.reshape(inputs, [-1, 1024])


            _variable_summaries(W_fc1)
            _variable_summaries(b_fc1)
            h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
            _activation_summary(h_fc1)
        return h_fc1

    def weight_variable(self, shape):

        initial = tf.truncated_normal(shape, stddev=0.1, name='weights')
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.0, shape=shape, name='biases')
        return tf.Variable(initial)

    def _summary_reshape(self, fweight, shape, num_new):
        """ Produce weights for a reduced fully-connected layer.

        FC8 of VGG produces 1000 classes. Most semantic segmentation
        task require much less classes. This reshapes the original weights
        to be used in a fully-convolutional layer which produces num_new
        classes. To archive this the average (mean) of n adjanced classes is
        taken.

        Consider reordering fweight, to perserve semantic meaning of the
        weights.

        Args:
          fweight: original weights
          shape: shape of the desired fully-convolutional layer
          num_new: number of new classes


        Returns:
          Filter weights for `num_new` classes.
        """
        num_orig = shape[3]
        shape[3] = num_new
        assert (num_new < num_orig)
        n_averaged_elements = num_orig // num_new
        avg_fweight = np.zeros(shape)
        for i in range(0, num_orig, n_averaged_elements):
            start_idx = i
            end_idx = start_idx + n_averaged_elements
            avg_idx = start_idx // n_averaged_elements
            if avg_idx == num_new:
                break
            avg_fweight[:, :, :, avg_idx] = np.mean(
                fweight[:, :, :, start_idx:end_idx], axis=3)
        return avg_fweight

    def _variable_with_weight_decay(self, shape, stddev, wd, decoder=False):
        """Helper to create an initialized Variable with weight decay.

        Note that the Variable is initialized with a truncated normal
        distribution.
        A weight decay is added only if one is specified.

        Args:
          name: name of the variable
          shape: list of ints
          stddev: standard deviation of a truncated Gaussian
          wd: add L2Loss weight decay multiplied by this float. If None, weight
              decay is not added for this Variable.

        Returns:
          Variable Tensor
        """

        initializer = tf.truncated_normal_initializer(stddev=stddev)
        var = tf.get_variable('weights', shape=shape,
                              initializer=initializer)

        collection_name = tf.GraphKeys.REGULARIZATION_LOSSES
        if wd and (not tf.get_variable_scope().reuse):
            weight_decay = tf.multiply(
                tf.nn.l2_loss(var), wd, name='weight_loss')
            tf.add_to_collection(collection_name, weight_decay)
        _variable_summaries(var)
        return var

    def _add_wd_and_summary(self, var, wd, collection_name=None):
        if collection_name is None:
            collection_name = tf.GraphKeys.REGULARIZATION_LOSSES
        if wd and (not tf.get_variable_scope().reuse):
            weight_decay = tf.multiply(
                tf.nn.l2_loss(var), wd, name='weight_loss')
            tf.add_to_collection(collection_name, weight_decay)
        _variable_summaries(var)
        return var

    def _bias_variable(self, shape, constant=0.0):
        initializer = tf.constant_initializer(constant)
        var = tf.get_variable(name='biases', shape=shape,
                              initializer=initializer)
        _variable_summaries(var)
        return var

    def get_fc_weight_reshape(self, name, shape, num_classes=None):
        print('Layer name: %s' % name)
        print('Layer shape: %s' % shape)
        weights = self.data_dict[name][0]
        weights = weights.reshape(shape)
        if num_classes is not None:
            weights = self._summary_reshape(weights, shape,
                                            num_new=num_classes)
        init = tf.constant_initializer(value=weights,
                                       dtype=tf.float32)
        var = tf.get_variable(name="weights", initializer=init, shape=shape)
        return var


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





