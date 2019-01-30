from math import ceil
import tensorflow as tf
from NETS import fcn8_vgg
from NETS import TSSN_plain_v1
import tensorflow.contrib.slim as slim
import logging
import numpy as np
import os
import sys
# final merge
class merge_net():
    def __init__(self):
        print('starting to merge_net')
        self.wd = 5e-4

        path = sys.modules[self.__class__.__module__].__file__
        # print path
        path = os.path.abspath(os.path.join(path, os.pardir))
        # print path
        path = os.path.join(path, "vgg16.npy")
        vgg16_npy_path = path
        logging.info("Load npy file from '%s'.", vgg16_npy_path)
        self.data_dict = np.load(vgg16_npy_path, encoding='latin1').item()

        print("npy file loaded")
    def bulid_merge(self, batch_images, task, alpha = 0.5, train=True, debug=False):

        # TSSN
        self.TSSN = TSSN_plain_v1.Merge()
        # with tf.name_scope('content_TSSN'):
        self.TSSN.Merge_layout_task_LL(batch_images, task, train=train)
        # self.TSSN_map = slim.batch_norm(self.TSSN.up_map8,activation_fn=tf.nn.relu,is_training=train,scope='norm_TSSN')
        self.TSSN_map = self.TSSN.up_map8


        # LLSN
        with tf.name_scope("content-LLSN"):
            self.score_fr = self._fc_layer(self.TSSN.Lgg_fcn.fc7, "LLSN_score_fr",
                                           num_classes=1,
                                           relu=False)
            self.upscore2 = self._upscore_layer(self.score_fr,
                                                shape=self.TSSN.Lgg_fcn.pool4.get_shape(),
                                                num_classes=1,
                                                debug=debug, name='LLSN_upscore2',
                                                ksize=4, stride=2)
            self.score_pool4 = self._score_layer(self.TSSN.Lgg_fcn.pool4, "LLSN_score_pool4",
                                                 num_classes=1)
            self.fuse_pool4 = tf.add(self.upscore2, self.score_pool4)

            self.upscore4 = self._upscore_layer(self.fuse_pool4,
                                                shape=self.TSSN.Lgg_fcn.pool3.get_shape(),
                                                num_classes=1,
                                                debug=debug, name='LLSN_upscore4',
                                                ksize=4, stride=2)
            self.score_pool3 = self._score_layer(self.TSSN.Lgg_fcn.pool3, "LLSN_score_pool3",
                                                 num_classes=1)
            self.fuse_pool3 = tf.add(self.upscore4, self.score_pool3)

            self.upscore32 = self._upscore_layer(self.fuse_pool3,
                                                 shape=batch_images.get_shape(),
                                                 num_classes=1,
                                                 debug=debug, name='LLSN_upscore32',
                                                 ksize=16, stride=8)
            self.LLSN = self.upscore32
        # self.saliency_map = slim.batch_norm(self.LLSN, activation_fn=tf.nn.relu, is_training=train, scope='norm_LLSN')
        self.saliency_map = self.LLSN



        self.map_sal = tf.add(self.saliency_map, self.TSSN_map, name='Added_merge')
        # self.map_sal = tf.multiply(self.saliency_map, self.TSSN_map, name='Added_merge')
    def _upscore_layer(self, bottom, shape,
                       num_classes, name, debug,
                       ksize=4, stride=2):
        strides = [1, stride, stride, 1]
        with tf.variable_scope(name):
            in_features = bottom.get_shape()[3].value
            if in_features is None:
                in_features = 1

            if shape is None:
                # Compute shape out of Bottom
                in_shape = tf.shape(bottom)

                h = ((in_shape[1] - 1) * stride) + 1
                w = ((in_shape[2] - 1) * stride) + 1
                new_shape = [in_shape[0], h, w, num_classes]
            else:
                new_shape = [shape[0], shape[1], shape[2], num_classes]
            output_shape = tf.stack(new_shape)

            logging.debug("Layer: %s, Fan-in: %d" % (name, in_features))
            f_shape = [ksize, ksize, num_classes, in_features]

            # create
            num_input = ksize * ksize * in_features / stride
            stddev = (2 / num_input) ** 0.5

            weights = self.get_deconv_filter(f_shape)

            deconv = tf.nn.conv2d_transpose(bottom, weights, output_shape,
                                            strides=strides, padding='SAME')

            if debug:
                deconv = tf.Print(deconv, [tf.shape(deconv)],
                                  message='Shape of %s' % name,
                                  summarize=4, first_n=1)


        return deconv

    def _score_layer(self, bottom, name, num_classes):

        with tf.variable_scope(name) as scope:
            # get number of input channels
            in_features = bottom.get_shape()[3].value
            shape = [1, 1, in_features, num_classes]
            # He initialization Sheme
            if name == "LLSN_score_fr":
                num_input = in_features
                stddev = (2 / num_input)**0.5
            elif name == "LLSN_score_pool4":
                stddev = 0.001
            elif name == "LLSN_score_pool3":
                stddev = 0.0001
            # Apply convolution
            w_decay = self.wd

            weights = self._variable_with_weight_decay(shape, stddev, w_decay)
            conv = tf.nn.conv2d(bottom, weights, [1, 1, 1, 1], padding='SAME')
            # Apply bias
            conv_biases = self._bias_variable([num_classes], constant=0.0)
            bias = tf.nn.bias_add(conv, conv_biases)



            return bias


    def get_deconv_filter(self, f_shape):
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
        for i in range(f_shape[2]):
            weights[:, :, i, i] = bilinear

        init = tf.constant_initializer(value=weights,
                                       dtype=tf.float32)
        var = tf.get_variable(name="up_filter", initializer=init,
                              shape=weights.shape)
        return var

    def get_conv_filter(self, name):
        init = tf.constant_initializer(value=self.data_dict[name][0],
                                       dtype=tf.float32)
        shape = self.data_dict[name][0].shape
        print('Layer name: %s' % name)
        print('Layer shape: %s' % str(shape))
        var = tf.get_variable(name="filter", initializer=init, shape=shape)
        if not tf.get_variable_scope().reuse:
            weight_decay = tf.multiply(tf.nn.l2_loss(var), self.wd,
                                       name='weight_loss')
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,
                                 weight_decay)

        return var

    def get_bias(self, name, num_classes=None):
        bias_wights = self.data_dict[name][1]
        shape = self.data_dict[name][1].shape
        if name == 'fc8':
            bias_wights = self._bias_reshape(bias_wights, shape[0],
                                             num_classes)
            shape = [num_classes]
        init = tf.constant_initializer(value=bias_wights,
                                       dtype=tf.float32)
        var = tf.get_variable(name="biases", initializer=init, shape=shape)

        return var

    def get_fc_weight(self, name):
        init = tf.constant_initializer(value=self.data_dict[name][0],
                                       dtype=tf.float32)
        shape = self.data_dict[name][0].shape
        var = tf.get_variable(name="weights", initializer=init, shape=shape)
        if not tf.get_variable_scope().reuse:
            weight_decay = tf.multiply(tf.nn.l2_loss(var), self.wd,
                                       name='weight_loss')
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,
                                 weight_decay)

        return var

    def _bias_reshape(self, bweight, num_orig, num_new):
        """ Build bias weights for filter produces with `_summary_reshape`

        """
        n_averaged_elements = num_orig//num_new
        avg_bweight = np.zeros(num_new)
        for i in range(0, num_orig, n_averaged_elements):
            start_idx = i
            end_idx = start_idx + n_averaged_elements
            avg_idx = start_idx//n_averaged_elements
            if avg_idx == num_new:
                break
            avg_bweight[avg_idx] = np.mean(bweight[start_idx:end_idx])
        return avg_bweight


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
        assert(num_new < num_orig)
        n_averaged_elements = num_orig//num_new
        avg_fweight = np.zeros(shape)
        for i in range(0, num_orig, n_averaged_elements):
            start_idx = i
            end_idx = start_idx + n_averaged_elements
            avg_idx = start_idx//n_averaged_elements
            if avg_idx == num_new:
                break
            avg_fweight[:, :, :, avg_idx] = np.mean(
                fweight[:, :, :, start_idx:end_idx], axis=3)
        return avg_fweight

    def _variable_with_weight_decay(self, shape, stddev, wd):
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

        if wd and (not tf.get_variable_scope().reuse):
            weight_decay = tf.multiply(
                tf.nn.l2_loss(var), wd, name='weight_loss')
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,
                                 weight_decay)
        return var

    def _bias_variable(self, shape, constant=0.0):
        initializer = tf.constant_initializer(constant)
        return tf.get_variable(name='biases', shape=shape,
                               initializer=initializer)

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
        return tf.get_variable(name="weights", initializer=init, shape=shape)


    def _fc_layer(self, bottom, name, num_classes=None,
                  relu=True, debug=False):
        with tf.variable_scope(name) as scope:
            shape = bottom.get_shape().as_list()

            if name == 'LLSN_fc6':
                filt = self.get_fc_weight_reshape(name[5:], [7, 7, 512, 4096])
            elif name == 'LLSN_score_fr':
                name = 'LLSN_fc8'  # Name of score_fr layer in VGG Model
                filt = self.get_fc_weight_reshape(name[5:], [1, 1, 4096, 1000],
                                                  num_classes=num_classes)
            else:
                filt = self.get_fc_weight_reshape(name[5:], [1, 1, 4096, 4096])
            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            conv_biases = self.get_bias(name[5:], num_classes=num_classes)
            bias = tf.nn.bias_add(conv, conv_biases)

            if relu:
                bias = tf.nn.relu(bias)


            if debug:
                bias = tf.Print(bias, [tf.shape(bias)],
                                message='Shape of %s' % name,
                                summarize=4, first_n=1)
            return bias