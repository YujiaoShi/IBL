import os

# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from cir_net_FOV import *
from distance import *
from OriNet_CVACT.input_data_CVACT import InputData
import tensorflow as tf
import numpy as np
import argparse
from tensorflow.python.ops.gen_math_ops import *
import scipy.io as scio

parser = argparse.ArgumentParser(description='TensorFlow implementation.')

parser.add_argument('--network_type',              type=str,
                    choices=['threebranch',
                             'twobranch_polar', 'twobranch_a2g', 'twobranch_a2gbottom'], default='threebranch')

parser.add_argument('--start_epoch', type=int, help='from epoch', default=25)
parser.add_argument('--number_of_epoch', type=int, help='number_of_epoch', default=100)
parser.add_argument('--polar', type=int, help='0 or 1', default=1)

parser.add_argument('--train_grd_noise', type=int, help='0~360', default=360)
parser.add_argument('--test_grd_noise', type=int, help='0~360', default=360)

parser.add_argument('--train_grd_FOV', type=int, help='70, 90, 180, 360', default=360)
parser.add_argument('--test_grd_FOV', type=int, help='70, 90, 180, 360', default=360)

args = parser.parse_args()

# --------------  configuration parameters  -------------- #
network_type = args.network_type

start_epoch = args.start_epoch
polar = args.polar

train_grd_noise = args.train_grd_noise
test_grd_noise = args.test_grd_noise

train_grd_FOV = args.train_grd_FOV
test_grd_FOV = args.test_grd_FOV

number_of_epoch = args.number_of_epoch

data_type = 'CVACT'

loss_type = 'l1'

batch_size = 32
is_training = False
loss_weight = 10.0
# number_of_epoch = 100

learning_rate_val = 1e-5
keep_prob_val = 0.8

dimension = 4


# -------------------------------------------------------- #

if __name__ == '__main__':
    tf.reset_default_graph()

    # import data
    input_data = InputData(polar)

    # define placeholders
    width = int(test_grd_FOV / 360 * 512)
    grd = tf.placeholder(tf.float32, [None, 128, width, 3], name='grd')
    sat = tf.placeholder(tf.float32, [None, 256, 256, 3], name='sat')
    g2a = tf.placeholder(tf.float32, [None, 256, 256, 3], name='g2a')
    a2g = tf.placeholder(tf.float32, [None, 128, 512, 3], name='a2g')
    sat_polar = tf.placeholder(tf.float32, [None, 128, 512, 3])

    grd_orien = tf.placeholder(tf.int32, [None], name='grd_orien')

    utms_x = tf.placeholder(tf.float32, [None, None, 1], name='utms')

    keep_prob = tf.placeholder(tf.float32)
    learning_rate = tf.placeholder(tf.float32)

    # build model
    if 'threebranch' in network_type:
        sat_matrix, grd_matrix, distance_global, distance_local1, distance_local2, pred_orien, corr_orien1, corr_orien2 \
            = model_threebranch(grd, sat_polar, a2g, keep_prob, is_training, dimension=8)

    elif network_type == 'twobranch_polar':
        sat_matrix, grd_matrix, distance, pred_orien = model_twobranch(grd, sat_polar, keep_prob, is_training)

    elif network_type == 'twobranch_a2g':
        sat_matrix, grd_matrix, distance, pred_orien = model_twobranch(grd, a2g, keep_prob, is_training)

    elif network_type == 'twobranch_a2gbottom':
        grd_bottom = tf.concat([tf.zeros_like(grd[:, 64:, :, :]), grd[:, 64:, :, :]], axis=1)
        sat_matrix, grd_matrix, distance, pred_orien = model_twobranch(grd_bottom, a2g, keep_prob, is_training)


    s_height, s_width, s_channel = sat_matrix.get_shape().as_list()[1:]
    g_height, g_width, g_channel = grd_matrix.get_shape().as_list()[1:]
    sat_global_matrix = np.zeros([input_data.get_test_dataset_size(), s_height, s_width, s_channel])
    grd_global_matrix = np.zeros([input_data.get_test_dataset_size(), g_height, g_width, g_channel])
    orientation_gth = np.zeros([input_data.get_test_dataset_size()])
    pred_orientation = np.zeros([input_data.get_test_dataset_size()])

    pred_loc1 = np.zeros([input_data.get_test_dataset_size()])

    saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)

    global_vars = tf.global_variables()

    # run model
    print('run model...')
    config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 1
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        print('load model...')

        load_model_path = '../ModelFOV/' + data_type + '/' + network_type \
                          + '/' \
                          + 'train_noise_' + str(train_grd_noise) + '_train_FOV_' + str(train_grd_FOV) \
                          + '/' + str(start_epoch - 1) + '/model.ckpt'
        saver.restore(sess, load_model_path)

        print("   Model loaded from: %s" % load_model_path)
        print('load model...FINISHED')

        # ---------------------- validation ----------------------
        print('validate...')
        print('   compute global descriptors')
        input_data.reset_scan()
        np.random.seed(2019)

        val_i = 0
        while True:
            print('      progress %d' % val_i)
            batch_grd, batch_sat, batch_g2a, batch_a2g, batch_sat_polar, batch_dis_utm, batch_orien \
                = input_data.next_batch_scan(batch_size, grd_noise=test_grd_noise, FOV=test_grd_FOV)
            if batch_sat is None:
                break
            feed_dict = {sat: batch_sat, grd: batch_grd, g2a: batch_g2a, a2g: batch_a2g, sat_polar: batch_sat_polar
                , keep_prob: 1}
            sat_matrix_val, grd_matrix_val = \
                sess.run([sat_matrix, grd_matrix], feed_dict=feed_dict)

            sat_global_matrix[val_i: val_i + sat_matrix_val.shape[0], :] = sat_matrix_val
            grd_global_matrix[val_i: val_i + grd_matrix_val.shape[0], :] = grd_matrix_val
            orientation_gth[val_i: val_i + grd_matrix_val.shape[0]] = batch_orien
            val_i += sat_matrix_val.shape[0]

        print('   compute accuracy')
        grd_descriptor = grd_global_matrix
        sat_descriptor = sat_global_matrix

        descriptor_dir = '../ResultFOV/CVACT/Descriptor/'
        if not os.path.exists(descriptor_dir):
            os.makedirs(descriptor_dir)

        file = descriptor_dir \
               + 'train_grd_noise_' + str(train_grd_noise) + '_train_grd_FOV_' + str(train_grd_FOV) \
               + 'test_grd_noise_' + str(test_grd_noise) + '_test_grd_FOV_' + str(test_grd_FOV) \
               + '_' + network_type + '.mat'
        scio.savemat(file, {'grd_descriptor': grd_descriptor, 'sat_descriptor': sat_descriptor,
                            'orientation_gth': orientation_gth})

        data_amount = grd_descriptor.shape[0]
        top1_percent = int(data_amount * 0.01) + 1

        if test_grd_noise==0:
            sat_descriptor = np.reshape(sat_global_matrix[:, :, :g_width, :], [-1, g_height * g_width * g_channel])
            sat_descriptor = sat_descriptor / np.linalg.norm(sat_descriptor, axis=-1, keepdims=True)

            grd_descriptor = np.reshape(grd_global_matrix, [-1, g_height * g_width * g_channel])

            dist_array = 2 - 2 * np.matmul(grd_descriptor, np.transpose(sat_descriptor))
            gt_dist = dist_array.diagonal()
            prediction = np.sum(dist_array < gt_dist.reshape(-1, 1), axis=-1)
            loc_acc = np.sum(prediction.reshape(-1, 1) < np.arange(top1_percent), axis=0) / data_amount

            scio.savemat(file, {'loc_acc': loc_acc,
                                'grd_descriptor': grd_descriptor, 'sat_descriptor': sat_descriptor})

        else:

            sat_fft = fft.fft(sat_descriptor.transpose([0, 3, 1, 2]))[:, np.newaxis, ...]

            loc_acc = np.zeros(top1_percent)
            for i in range(400):
                print(i)
                batch_start = int(data_amount * i / 400)
                if i < 399:
                    batch_end = int(data_amount * (i + 1) / 400)
                else:
                    batch_end = data_amount

                dist_array, pred_orien = corr_distance_FOV_np(grd_descriptor[batch_start: batch_end, :], sat_descriptor, sat_fft)
                gt_dist = np.array(
                    [dist_array[index, batch_start + index] for index in range(batch_end - batch_start)]).reshape(
                    [batch_end - batch_start, 1])

                # record indexes of ground images which have been correctly localized
                pred_loc1[batch_start:batch_end] = np.min(dist_array, axis=-1) == gt_dist[:, 0]

                pred_orientation[batch_start:batch_end] = \
                    ((pred_orien[batch_start:batch_end, :].diagonal()) * 360 / 64 + test_grd_FOV / 2) % 360 - 180

                prediction = np.sum(dist_array < gt_dist, axis=-1)

                loc_acc += np.sum(prediction.reshape(-1, 1) < np.arange(top1_percent).reshape(1, -1), axis=0)

            loc_acc = loc_acc / data_amount


            print('top-1:', loc_acc[1])
            print('top-5:', loc_acc[5])
            print('top-10:', loc_acc[10])
            print('top-1%:', loc_acc[-1])

            idx = (pred_loc1 == 1).squeeze()

            dis = pred_orientation[idx] - orientation_gth[idx]
            idx1 = dis > 180
            idx2 = dis < -180
            dis1 = dis.copy()
            dis1[idx1] = 360 - dis[idx1]
            dis1[idx2] = dis[idx2] + 360

            orien_median = np.median(np.abs(dis1))
            orien_acc = np.mean(np.abs(dis1) < 36)

            print('orientation median value: ', orien_median)
            print('orientation accuracy: ', orien_acc)

            scio.savemat(file, {'orientation_gth': orientation_gth,
                                'pred_orientation': pred_orientation,
                                'idx': idx,
                                'pred_loc1': pred_loc1,
                                'loc_acc': loc_acc,
                                'orien_acc': orien_acc,
                                'orien_median': orien_median,
                                'grd_descriptor': grd_descriptor,
                                'sat_descriptor': sat_descriptor})

