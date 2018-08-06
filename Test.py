import argparse
import os
import time
import Task_driven_model
import matplotlib.pylab as plt
import numpy as np
import scipy.misc
import tensorflow as tf
import math

from utils import *


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
parser = argparse.ArgumentParser(
    description='Monodepth TensorFlow implementation.')


parser.add_argument('--save_path', type=str, help='directory to save checkpoints and summaries',
                    default='')
args = parser.parse_args()

def test():
    with tf.Graph().as_default():
        # img_gt = misc.imread()
        batch_img = readtestdata(['data/Benchmark_webpage.tfrecords'])
        batch_imgs = tf.placeholder('float32', shape=[1, 224, 224, 3])
        batch_labs = tf.placeholder('float32', shape=[1, 5])

        Final_mergeNet = Task_driven_model.merge_net()
        Final_mergeNet.bulid_merge(batch_imgs, batch_labs, alpha = args.alpha, train=False)

        config = tf.ConfigProto()
        config.allow_soft_placement = True
        config.gpu_options.allow_growth = True  # True
        config.log_device_placement = True
        sess = tf.Session(config=config)
        train_saver = tf.train.Saver()

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        coordinator = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)
        train_saver.restore(sess, 'model/model-5000')

        print('Running the Network')

        print(args.save_path)
        if not os.path.exists(args.save_path):
            print('done~')
            os.makedirs(args.save_path)

        for step in range(0, 200):
            lab = np.zeros((1, 5))
            lab[:, 1] = 1.0

            [imgs_gt] = sess.run([batch_img])
            img_gt_s = np.squeeze(imgs_gt)

            title_name = ['signing-up', 'filling in information', 'reading product', 'shopping', 'whether to join']

            for j in range(0, 5):
                lab = np.zeros((1, 5))
                lab[:, j] = 1.0
                up_sal = sess.run([Final_mergeNet.map_sal],
                                                    feed_dict={batch_imgs: imgs_gt, batch_labs: lab})
                pred_color = up_sal[0, :, :, 0]

                misc.imsave(os.path.join(args.save_path, "{}_task_{}.png".format('image_' + str(step * 1).zfill(3), str(j + 1))),
                            pred_color)

        coordinator.request_stop()
        coordinator.join(threads)

def main(_):

    test()


if __name__ == '__main__':
    tf.app.run()
