import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from cir_net_FOV import *
from input_data_CVUSA import InputData

import tensorflow as tf
import numpy as np

import argparse

parser = argparse.ArgumentParser(description='TensorFlow implementation.')

parser.add_argument('--network_type',              type=str,
                    choices=['threebranch',
                             'twobranch_polar', 'twobranch_a2g', 'twobranch_a2gbottom'], default='threebranch')

parser.add_argument('--start_epoch', type=int, help='from epoch', default=0)
parser.add_argument('--number_of_epoch', type=int, help='number_of_epoch', default=100)

parser.add_argument('--train_grd_noise', type=int, help='0~360', default=360)
parser.add_argument('--test_grd_noise', type=int, help='0~360', default=0)

parser.add_argument('--train_grd_FOV', type=int, help='70, 90, 180, 360', default=360)
parser.add_argument('--test_grd_FOV', type=int, help='70, 90, 180, 360', default=360)

args = parser.parse_args()

# --------------  configuration parameters  -------------- #
network_type = args.network_type

start_epoch = args.start_epoch

train_grd_noise = args.train_grd_noise
test_grd_noise = args.test_grd_noise

train_grd_FOV = args.train_grd_FOV
test_grd_FOV = args.test_grd_FOV

number_of_epoch = args.number_of_epoch

data_type = 'CVUSA'

batch_size = 32
is_training = True
loss_weight = 10.0

learning_rate_val = 1e-5
keep_prob_val = 0.8

dimension = 4

# -------------------------------------------------------- #

def validate(dist_array, topK):
    accuracy = 0.0
    data_amount = 0.0

    for i in range(dist_array.shape[0]):
        gt_dist = dist_array[i, i]
        prediction = np.sum(dist_array[i, :] < gt_dist)
        if prediction < topK:
            accuracy += 1.0
        data_amount += 1.0
    accuracy /= data_amount

    return accuracy


def compute_loss(dist_array):

    with tf.name_scope('weighted_soft_margin_triplet_loss'):

        pos_dist = tf.diag_part(dist_array)

        pair_n = batch_size * (batch_size - 1.0)

        # satellite to ground
        triplet_dist_g2s = pos_dist - dist_array
        loss_g2s = tf.reduce_sum(tf.log(1 + tf.exp(triplet_dist_g2s * loss_weight))) / pair_n

        # ground to satellite
        triplet_dist_s2g = tf.expand_dims(pos_dist, 1) - dist_array
        loss_s2g = tf.reduce_sum(tf.log(1 + tf.exp(triplet_dist_s2g * loss_weight))) / pair_n

        loss = (loss_g2s + loss_s2g) / 2.0

    return loss


def train(start_epoch=0):
    '''
    Train the network and do the test
    :param start_epoch: the epoch id start to train. The first epoch is 0.
    '''

    # import data
    input_data = InputData()

    width = int(test_grd_FOV / 360 * 512)

    grd = tf.placeholder(tf.float32, [None, 128, width, 3], name='grd')
    sat = tf.placeholder(tf.float32, [None, 256, 256, 3], name='sat')
    g2a = tf.placeholder(tf.float32, [None, 256, 256, 3], name='g2a')
    a2g = tf.placeholder(tf.float32, [None, 128, 512, 3], name='a2g')
    sat_polar = tf.placeholder(tf.float32, [None, 128, 512, 3])

    grd_orien = tf.placeholder(tf.int32, [None], name='grd_orien')

    keep_prob = tf.placeholder(tf.float32)
    learning_rate = tf.placeholder(tf.float32)

    # build model
    if 'threebranch' in network_type:
        sat_matrix, grd_matrix, distance_global, distance_local1, distance_local2, pred_orien, corr_orien1, corr_orien2 \
            = model_threebranch(grd, sat_polar, a2g, keep_prob, is_training, dimension=8)
        loss_global = compute_loss(distance_global)
        loss_local1 = compute_loss(distance_local1)
        loss_local2 = compute_loss(distance_local2)

        loss = (loss_global + loss_local1 + loss_local2)

    elif network_type=='twobranch_polar':
        sat_matrix, grd_matrix, distance, pred_orien = model_twobranch(grd, sat_polar, keep_prob, is_training)
        loss = compute_loss(distance)
    elif network_type=='twobranch_a2g':
        sat_matrix, grd_matrix, distance, pred_orien = model_twobranch(grd, a2g, keep_prob, is_training)
        loss = compute_loss(distance)
    elif network_type=='twobranch_a2gbottom':
        grd_bottom = tf.concat([tf.zeros_like(grd[:, 64:, :, :]), grd[:, 64:, :, :]], axis=1)
        sat_matrix, grd_matrix, distance, pred_orien = model_twobranch(grd_bottom, a2g, keep_prob, is_training)

        loss = compute_loss(distance)


    s_height, s_width, s_channel = sat_matrix.get_shape().as_list()[1:]
    g_height, g_width, g_channel = grd_matrix.get_shape().as_list()[1:]
    sat_global_matrix = np.zeros([input_data.get_test_dataset_size(), s_height, s_width, s_channel])
    grd_global_matrix = np.zeros([input_data.get_test_dataset_size(), g_height, g_width, g_channel])
    orientation_gth = np.zeros([input_data.get_test_dataset_size()])

    # set training
    global_step = tf.Variable(0, trainable=False)

    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(learning_rate, 0.9, 0.999).minimize(loss, global_step=global_step)

    saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)

    global_vars = tf.global_variables()

    # Compute the amount of trainable parameters

    trainable_params = tf.trainable_variables()
    total = 0
    for var in trainable_params:
        total += np.prod(var.get_shape().as_list())
    print(total)

    # ===================================== LOAD MODEL ======================================

    var_dict1 = {}
    for var in global_vars:
        if 'VGG_grd' in var.op.name and 'Adam' not in var.op.name:
            var_dict1[var.op.name] = var

    var_dict2 = {}
    for var in global_vars:
        if 'VGG_sat' in var.op.name and 'Adam' not in var.op.name:
            var_dict2[var.op.name] = var

    var_dict3 = {}
    for var in global_vars:
        if 'VGG_a2g' in var.op.name and 'Adam' not in var.op.name:
            var_dict3[var.op.name.replace('VGG_a2g', 'VGG_grd')] = var

    var_dict4 = {}
    for var in global_vars:
        if 'VGG_polar' in var.op.name and 'Adam' not in var.op.name:
            var_dict4[var.op.name.replace('VGG_polar', 'VGG_grd')] = var

    if len(var_dict1) > 0:
        saver1 = tf.train.Saver(var_dict1)
    if len(var_dict2) > 0:
        saver2 = tf.train.Saver(var_dict2)
    if len(var_dict3) > 0:
        saver3 = tf.train.Saver(var_dict3)
    if len(var_dict4) > 0:
        saver4 = tf.train.Saver(var_dict4)

    # ======================================================================================

    # run model
    print('run model...')
    config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    print('open session ...')
    with tf.Session(config=config) as sess:
        print('initialize...')
        sess.run(tf.global_variables_initializer())

        print('load model...')

        if start_epoch == 0:
            load_model_path = '../Model/CVM-Net_model/CVM-NET-I/CVM-NET-I_init/model.ckpt'
            if len(var_dict1) > 0:
                saver1.restore(sess, load_model_path)
                print('saver1: load done...')
            if len(var_dict2) > 0:
                saver2.restore(sess, load_model_path)
                print('saver2: load done...')
            if len(var_dict3) > 0:
                saver3.restore(sess, load_model_path)
                print('saver3: load done...')
            if len(var_dict4) > 0:
                saver4.restore(sess, load_model_path)
                print('saver4: load done...')
        else:

            load_model_path = '../Model/' + data_type + '/' + network_type \
                              + '/' \
                              + 'train_noise_' + str(train_grd_noise) + '_train_FOV_' + str(train_grd_FOV) \
                              + '/' + str(start_epoch - 1) + '/model.ckpt'

            saver.restore(sess, load_model_path)

        print("   Model loaded from: %s" % load_model_path)
        print('load model...FINISHED')

        # Train
        for epoch in range(start_epoch, start_epoch + number_of_epoch):
            iter = 0
            while True:
                batch_grd, batch_sat, batch_g2a, batch_a2g, batch_sat_polar, batch_orien \
                    = input_data.next_pair_batch(batch_size, grd_noise=train_grd_noise, FOV=train_grd_FOV)
                if batch_sat is None:
                    break

                global_step_val = tf.train.global_step(sess, global_step)

                feed_dict = {sat: batch_sat, grd: batch_grd, g2a: batch_g2a, a2g: batch_a2g, sat_polar: batch_sat_polar,
                             learning_rate: learning_rate_val, keep_prob: keep_prob_val}
                if iter % 20 == 0:
                    _, loss_val = sess.run([train_step, loss], feed_dict=feed_dict)
                    if 'threebranch' in network_type:
                        loss_global_val, loss_local1_val, loss_local2_val = \
                            sess.run([loss_global, loss_local1, loss_local2], feed_dict=feed_dict)
                        print(
                            'global %d, epoch %d, iter %d: triplet_loss : %.4f, '
                            'loss_global : %.4f, loss_local1 : %.4f, loss_local2 : %.4f ' %
                            (global_step_val, epoch, iter, loss_val, loss_global_val, loss_local1_val, loss_local2_val))
                    else:
                        print('global %d, epoch %d, iter %d: triplet_loss : %.4f ' %
                              (global_step_val, epoch, iter, loss_val))

                else:
                    sess.run(train_step, feed_dict=feed_dict)

                iter += 1


            model_dir = '../Model/' + data_type + '/' + network_type \
                        + '/' \
                        + '_train_noise_' + str(train_grd_noise) + '_train_FOV_' + str(train_grd_FOV) \
                        + '/' + str(epoch) + '/'

            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            save_path = saver.save(sess, model_dir + 'model.ckpt')
            print("Model saved in file: %s" % save_path)

            # ---------------------- validation ----------------------

            print('validate...')
            print('   compute global descriptors')
            input_data.reset_scan()

            val_i = 0
            while True:
                # print('      progress %d' % val_i)
                batch_grd, batch_sat, batch_g2a, batch_a2g, batch_sat_polar, _ \
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
            sat_descriptor = np.reshape(sat_global_matrix[:, :, :g_width, :], [-1, g_height*g_width*g_channel])
            sat_descriptor = sat_descriptor / np.linalg.norm(sat_descriptor, axis=-1, keepdims=True)
            grd_descriptor = np.reshape(grd_global_matrix, [-1, g_height*g_width*g_channel])

            # test_grd_noise == 0:
            dist_array = 2 - 2 * np.matmul(grd_descriptor, sat_descriptor.transpose())
            val_accuracy = validate(dist_array, 1)

            result_dir = '../Result/' + data_type + '/'

            if not os.path.exists(result_dir):
                os.makedirs(result_dir)

            print('   %d: accuracy = %.1f%%' % (epoch, val_accuracy * 100.0))

            with open(result_dir + str(network_type)
                      + '_train_noise_' + str(train_grd_noise) + '_train_FOV_' + str(train_grd_FOV)
                      + '_test_noise_' + str(0) + '_test_FOV_' + str(test_grd_FOV)
                      + '.txt', 'a') as file:
                file.write(str(epoch) + ' ' + str(iter) + ' : ' + str(val_accuracy) + '\n')


if __name__ == '__main__':
    train(start_epoch)
