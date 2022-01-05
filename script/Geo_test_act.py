import os

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import scipy.io as scio
from cir_net_FOV import *

from OriNet_CVACT.input_data_ACT_test_polar import InputData

import tensorflow as tf
from tensorflow.python.ops.gen_math_ops import *
import numpy as np

import argparse

parser = argparse.ArgumentParser(description='TensorFlow implementation.')

parser.add_argument('--network_type',              type=str,
                    choices=['threebranch_notshare_low', 'threebranch_notshare_high',
                             'threebranch_share_low', 'threebranch_share_high',
                             'twobranch_polar', 'twobranch_a2g', 'twobranch_a2gbottom'], default='threebranch_notshare_low')

parser.add_argument('--train_grd_noise',           type=int,   help='0~360',    default=360)
parser.add_argument('--test_grd_noise',            type=int,   help='0~360',    default=0)

parser.add_argument('--train_grd_FOV',             type=int,   help='70, 90, 100, 120, 180, 360',   default=360)
parser.add_argument('--test_grd_FOV',              type=int,   help='70, 90, 100, 120, 180, 360',   default=360)

args = parser.parse_args()

# --------------  configuration parameters  -------------- #
network_type = args.network_type

train_grd_noise = args.train_grd_noise
test_grd_noise = args.test_grd_noise

train_grd_FOV = args.train_grd_FOV
test_grd_FOV = args.test_grd_FOV


data_type = 'CVACT'

loss_type = 'l1'

batch_size = 32
is_training = True
loss_weight = 10.0

learning_rate_val = 1e-5
keep_prob_val = 0.8
# -------------------------------------------------------- #


if __name__ == '__main__':

    tf.reset_default_graph()

    # import data
    input_data = InputData()

    # define placeholders
    width = int(test_grd_FOV / 360 * 512)
    grd = tf.placeholder(tf.float32, [None, 128, width, 3], name='grd')
    sat = tf.placeholder(tf.float32, [None, 256, 256, 3], name='sat')
    # g2a = tf.placeholder(tf.float32, [None, 256, 256, 3], name='g2a')
    a2g = tf.placeholder(tf.float32, [None, 128, 512, 3], name='a2g')
    sat_polar = tf.placeholder(tf.float32, [None, 128, 512, 3])

    # grd_orien = tf.placeholder(tf.int32, [None], name='grd_orien')

    utms_x = tf.placeholder(tf.float32, [None, None, 1], name='utms')

    keep_prob = tf.placeholder(tf.float32)
    learning_rate = tf.placeholder(tf.float32)

    # build model
    if 'threebranch' in network_type:
        sat_matrix, grd_matrix, distance_global, distance_local1, distance_local2, pred_orien, corr_orien1, corr_orien2 \
            = model_threebranch(grd, sat_polar, a2g, keep_prob, is_training, dimension=8,
                                loss_method='2', network_type=network_type)

    elif 'twobranch' in network_type:
        if network_type == 'twobranch_polar':
            sat_matrix, grd_matrix, distance, pred_orien = model_twobranch(grd, sat_polar, keep_prob, is_training)
        elif network_type == 'twobranch_a2g':
            sat_matrix, grd_matrix, distance, pred_orien = model_twobranch(grd, a2g, keep_prob, is_training)
        elif network_type == 'twobranch_a2gbottom':
            grd_bottom = tf.concat([tf.zeros([batch_size, 64, width, 3]), grd[:, 64:, :, :]], axis=1)
            sat_matrix, grd_matrix, distance, pred_orien = model_twobranch(grd_bottom, a2g, keep_prob, is_training)

    s_height, s_width, s_channel = sat_matrix.get_shape().as_list()[1:]
    g_height, g_width, g_channel = grd_matrix.get_shape().as_list()[1:]
    sat_global_matrix = np.zeros([input_data.get_test_dataset_size(), s_height, s_width, s_channel])
    grd_global_matrix = np.zeros([input_data.get_test_dataset_size(), g_height, g_width, g_channel])

    saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)

    # run model
    print('run model...')
    config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 1
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        print('load model...')

        load_model_path = '../ModelFOV/' + data_type + '/' + network_type \
                          + '/2_train_noise_' + str(train_grd_noise) + '_train_FOV_' + str(train_grd_FOV) \
                          + '/97/model.ckpt'
        saver.restore(sess, load_model_path)

        # ---------------------- validation ----------------------

        print('validate...')
        print('   compute global descriptors')
        input_data.reset_scan()

        val_i = 0
        while True:
            print('      progress %d' % val_i)
            batch_grd, batch_sat, batch_a2g, batch_sat_polar, batch_dis_utm = input_data.next_batch_scan(batch_size)
            if batch_sat is None:
                break
            feed_dict = {sat: batch_sat, grd: batch_grd, a2g: batch_a2g, sat_polar: batch_sat_polar
                , keep_prob: 1}
            sat_matrix_val, grd_matrix_val = \
                sess.run([sat_matrix, grd_matrix], feed_dict=feed_dict)

            # sat_global_val, grd_global_val = \
            #     sess.run([sat_matrix, grd_matrix], feed_dict=feed_dict)

            sat_global_matrix[val_i: val_i + sat_matrix_val.shape[0], ...] = sat_matrix_val
            grd_global_matrix[val_i: val_i + grd_matrix_val.shape[0], ...] = grd_matrix_val
            val_i += sat_matrix_val.shape[0]

        sat_global_descriptor = np.reshape(sat_global_matrix, [-1, s_height*s_width*s_channel])
        grd_global_descriptor = np.reshape(grd_global_matrix, [-1, g_height*g_width*g_channel])

        data_file = '../ResultFOV/CVACT/Geo_test/' + network_type + '_grd_global_descriptor.mat'
        scio.savemat(data_file, {'grd_global_descriptor': grd_global_descriptor})

        data_file = '../ResultFOV/CVACT/Geo_test/' + network_type + 'sat_global_descriptor.mat'
        scio.savemat(data_file, {'sat_global_descriptor': sat_global_descriptor})


        print(network_type, ' done...')
